#!/usr/bin/env python3
"""
Harmonize Replogle 2022, LINCS L1000 and LINCS GMT files to the challenge
gene vocabulary.

Principle: data VALUES are left exactly as they are. Only names are made
consistent, and annotation is added. Outputs are standard .h5ad / .gmt files,
so they work with any method.

Conventions in every output .h5ad
---------------------------------
  X                        copied unchanged from the source (dtype, values, NaN/inf)
  var_names                challenge symbol if the gene is in gene_names.csv,
                           otherwise the source symbol (made unique)
  var['gene_symbol_orig']  symbol as in the source
  var['ensembl_id']        versionless Ensembl ID ('' if unknown)
  var['challenge_idx']     column in challenge gene order, -1 if not a challenge gene
  var['match']             symbol | ensembl | alias | none
  obs['target_gene']       perturbation target, harmonized the same way
  obs['target_gene_orig']  as in the source
  obs['is_control']        Replogle controls are labelled 'non-targeting', as in the challenge
  (all original obs / var columns are kept)

Outputs (under --out)
---------------------
  replogle/<name>.h5ad
  lincs/<gctx name>.h5ad
  lincs/<gmt name>.harmonized.gmt     same sets, member genes renamed
  lincs/<gmt name>.sets.parquet       one row per set: cell line, time, target ...
  gene_coverage.csv                   challenge genes x datasets (measured?)
  pert_coverage.csv                   challenge perturbations x datasets (present?)

Usage
-----
  python harmonize_data.py replogle     # bulk + raw single-cell by default
  python harmonize_data.py lincs        # level5 trt_xpr + ctl by default
  python harmonize_data.py gmt
  python harmonize_data.py coverage
  python harmonize_data.py all

Existing outputs are skipped unless --overwrite, so re-running is cheap.
Requires genes.py in the same folder.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from genes import (harmonize_genes, harmonize_labels, load_reference,  # noqa: E402
                   strip_version, summarize_labels, var_ids)

DATA = Path("/data/han/projects/VCC/data")
REF = DATA / "ref/challenge_genes.tsv"
PERTS = DATA / "ref/challenge_perts.txt"
HGNC = DATA / "ref/hgnc_complete_set.txt"
REPLOGLE = Path("/data/share/han/Spatial3D/data_Replogle_2022")
LINCS = Path("/data/share/han/Spatial3D/data_LINCS")
OUT = DATA / "processed"

CTRL = "non-targeting"
GENETIC = {"trt_xpr", "trt_sh", "trt_sh.cgs", "trt_sh.css", "trt_oe", "trt_si"}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def unique_names(names, keep_first: np.ndarray) -> pd.Index:
    """Make names unique with -1, -2 suffixes. Rows flagged keep_first
    (challenge genes) always keep the clean name."""
    s = pd.Series(np.asarray(names, dtype=object)).astype(str)
    order = np.lexsort((np.arange(len(s)), ~keep_first))
    so = s.iloc[order]
    cc = np.empty(len(s), dtype=int)
    cc[order] = so.groupby(so.values).cumcount().values
    return pd.Index(np.where(cc > 0, s + "-" + cc.astype(str), s))


def gene_table(symbols, ensembl, ref) -> tuple[pd.DataFrame, dict]:
    t, info = harmonize_genes(symbols, ensembl, ref)
    var = pd.DataFrame({"gene_symbol_orig": t["gene_symbol_orig"].values,
                        "ensembl_id": t["gene_id"].values,
                        "challenge_idx": t["target_idx"].values,
                        "match": t["match"].values})
    var.index = unique_names(t["gene_symbol"].values, t["target_idx"].values >= 0)
    return var, info


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame writable to h5ad: text-like columns -> str."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "category":
            df[c] = df[c].astype(object).where(df[c].notna(), "").astype(str)
    df.columns = [str(c) for c in df.columns]
    df.index = df.index.astype(str)
    return df


def fix_x_encoding(x) -> None:
    """Older h5ad files lack the encoding attrs newer anndata expects."""
    import h5py
    if "encoding-type" in x.attrs:
        return
    if isinstance(x, h5py.Dataset):
        x.attrs["encoding-type"], x.attrs["encoding-version"] = "array", "0.2.0"
    elif "h5sparse_format" in x.attrs:
        fmt = x.attrs["h5sparse_format"]
        fmt = fmt.decode() if isinstance(fmt, bytes) else str(fmt)
        x.attrs["encoding-type"] = f"{fmt}_matrix"
        x.attrs["encoding-version"] = "0.1.0"
        x.attrs["shape"] = x.attrs["h5sparse_shape"]


def write_h5ad(dst: Path, obs: pd.DataFrame, var: pd.DataFrame, uns: dict, fill_x) -> None:
    """Write obs/var/uns with anndata, then add X directly with h5py."""
    import anndata as ad
    import h5py
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.h5ad")
    ad.AnnData(obs=clean_frame(obs), var=clean_frame(var), uns=uns).write_h5ad(tmp)
    with h5py.File(tmp, "a") as f:
        if "X" in f:
            del f["X"]
        fill_x(f)
        fix_x_encoding(f["X"])
    tmp.replace(dst)


def check_disk(where: Path, need: int) -> None:
    where.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(where).free
    if need > free * 0.95:
        sys.exit(f"  Not enough disk in {where}: need {need / 1e9:.1f} GB, "
                 f"free {free / 1e9:.1f} GB")


def expand(folder: Path, patterns: list[str]) -> list[Path]:
    return sorted({p for pat in patterns for p in folder.glob(pat)})


def label_line(info: dict) -> str:
    return (f"  genes: {info['n_matched']:,}/{info['n_reference_genes']:,} challenge genes "
            f"(exact {info['matched_by_symbol']:,}, Ensembl {info['matched_by_ensembl']}, "
            f"alias {info['matched_by_alias']})")


def uns_info(info: dict) -> dict:
    return {k: v for k, v in info.items() if k != "renamed_examples"}


# --------------------------------------------------------------------------- #
# Replogle
# --------------------------------------------------------------------------- #
def parse_replogle_index(names: pd.Index) -> pd.DataFrame:
    """'10023_ZC3H18_P1P2_ENSG00000158545' -> gene, transcript, ensembl."""
    rows = []
    for s in names.astype(str):
        p = s.split("_")
        if len(p) >= 4 and p[-1].startswith("ENS"):
            rows.append(("_".join(p[1:-2]), p[-2], p[-1]))
        else:
            rows.append((s, "", ""))
    return pd.DataFrame(rows, columns=["gene", "transcript", "ensembl"])


def do_replogle(a, ref, perts) -> None:
    import anndata as ad
    import h5py
    out = a.out / "replogle"
    for path in expand(a.replogle, a.replogle_files):
        dst = out / path.name.replace("_01.h5ad", ".h5ad")
        if dst.exists() and not a.overwrite:
            print(f"[skip] {dst.name} exists")
            continue
        print(f"\n=== {path.name}  ({path.stat().st_size / 1e9:.1f} GB)")
        check_disk(out, int(path.stat().st_size * 1.02))

        src = ad.read_h5ad(path, backed="r")
        obs, var_src = src.obs.copy(), src.var.copy()
        src.file.close()

        # genes
        sym, ens = var_ids(var_src)
        var, info = gene_table(sym, ens, ref)
        extra = var_src.reset_index(drop=True)
        extra.index = var.index
        var = pd.concat([var, extra.drop(columns=[c for c in extra.columns
                                                  if c in var.columns])], axis=1)
        print(label_line(info))

        # perturbation targets
        if "gene" in obs.columns:                      # single-cell
            orig = obs["gene"].astype(str)
            t_ens = obs["gene_id"].astype(str).values if "gene_id" in obs.columns else None
        else:                                          # bulk: target is in the row name
            p = parse_replogle_index(obs.index)
            orig = pd.Series(p["gene"].values, index=obs.index)
            t_ens = p["ensembl"].values
            obs["target_transcript"] = p["transcript"].values
            obs["target_ensembl"] = p["ensembl"].values
        ctrl = np.array(orig.str.lower().str.contains("non-targeting").values, dtype=bool)
        if "core_control" in obs.columns:
            ctrl = ctrl | np.asarray(obs["core_control"].astype(bool).values, dtype=bool)

        new, rep = harmonize_labels(orig.values, t_ens, ref, extra_symbols=perts,
                                    keep=set(orig[ctrl]))
        obs.insert(0, "target_gene", np.where(ctrl, CTRL, new))
        obs.insert(1, "target_gene_orig", orig.values)
        obs.insert(2, "is_control", ctrl)
        print(summarize_labels(rep))
        if perts:
            have = set(obs.loc[~ctrl, "target_gene"])
            print(f"  challenge perturbations present: {sum(p in have for p in perts)}"
                  f"/{len(perts)}   controls: {int(ctrl.sum()):,}")

        uns = {"source": str(path), "gene_harmonization": uns_info(info),
               "note": "X copied unchanged from source"}

        def fill_x(f, path=path):
            with h5py.File(path, "r") as s:
                s.copy(s["X"], f, name="X")

        write_h5ad(dst, obs, var, uns, fill_x)
        rep.to_csv(dst.with_suffix(".target_harmonization.csv"), index=False)
        print(f"  wrote {dst}")


# --------------------------------------------------------------------------- #
# LINCS GCTX
# --------------------------------------------------------------------------- #
def load_geneinfo(folder: Path) -> pd.DataFrame:
    gi = pd.read_csv(folder / "geneinfo_beta.txt", sep="\t", dtype=str)
    gi["gene_id"] = gi["gene_id"].str.strip()
    return gi


def gctx_ids(f):
    dec = lambda v: v.decode().strip() if isinstance(v, bytes) else str(v).strip()
    cid = [dec(v) for v in f["0/META/COL/id"][:]]
    rid = [dec(v) for v in f["0/META/ROW/id"][:]]
    M = f["0/DATA/0/matrix"]
    if M.shape == (len(cid), len(rid)):
        return cid, rid, M, False
    if M.shape == (len(rid), len(cid)):
        return cid, rid, M, True
    sys.exit(f"Unexpected GCTX matrix shape {M.shape} for {len(cid)} cols x {len(rid)} rows")


def do_lincs(a, ref, perts) -> None:
    import h5py
    out = a.out / "lincs"
    gi = load_geneinfo(a.lincs)
    by_id = gi.set_index("gene_id")
    sym2ens = gi.drop_duplicates("gene_symbol").set_index("gene_symbol").get("ensembl_id")

    meta = {}
    for fname, key in (("siginfo_beta.txt", "sig_id"), ("instinfo_beta.txt", "sample_id")):
        if (a.lincs / fname).exists():
            meta[key] = pd.read_csv(a.lincs / fname, sep="\t", low_memory=False)

    for path in expand(a.lincs, a.gctx):
        dst = out / path.name.replace(".gctx", ".h5ad")
        if dst.exists() and not a.overwrite:
            print(f"[skip] {dst.name} exists")
            continue
        print(f"\n=== {path.name}")
        with h5py.File(path, "r") as f:
            cid, rid, M, transposed = gctx_ids(f)
            nbytes = len(cid) * len(rid) * M.dtype.itemsize
        check_disk(out, int(nbytes * 1.02))

        # obs: join signature / instance metadata on column id
        obs = pd.DataFrame(index=pd.Index(cid, name=None))
        for key, df in meta.items():
            if df[key].astype(str).isin(set(cid[:5000])).mean() > 0:
                m = df.drop_duplicates(key).set_index(df.drop_duplicates(key)[key].astype(str))
                obs = m.reindex(cid).drop(columns=[key])
                print(f"  metadata from {key}: {obs.notna().any(axis=1).mean():.1%} of columns matched")
                break
        pt = obs["pert_type"].astype(str) if "pert_type" in obs.columns else pd.Series("", index=obs.index)
        orig = obs["cmap_name"].astype(str) if "cmap_name" in obs.columns else pd.Series("", index=obs.index)
        genetic = pt.isin(GENETIC).values
        ctrl = pt.str.startswith("ctl").values

        target = orig.values.astype(object).copy()
        if genetic.any():
            lab = orig.values[genetic]
            t_ens = [sym2ens.get(s, "") if sym2ens is not None else "" for s in lab]
            new, rep = harmonize_labels(lab, t_ens, ref, extra_symbols=perts)
            target[genetic] = new
            print(summarize_labels(rep))
            rep.to_csv(dst.with_suffix(".target_harmonization.csv"), index=False)
        obs.insert(0, "target_gene", target)
        obs.insert(1, "target_gene_orig", orig.values)
        obs.insert(2, "is_control", ctrl)

        # var: Entrez row ids -> symbol + Ensembl via geneinfo
        g = by_id.reindex(rid)
        sym = g["gene_symbol"].fillna(pd.Series(rid, index=g.index)).values
        ens = g["ensembl_id"].fillna("").values if "ensembl_id" in g.columns else None
        var, info = gene_table(sym, ens, ref)
        var["entrez_id"] = rid
        if "feature_space" in g.columns:
            var["feature_space"] = g["feature_space"].fillna("").values
        print(label_line(info))
        if perts:
            have = set(target[genetic & ~ctrl])
            print(f"  challenge perturbations present: {sum(p in have for p in perts)}/{len(perts)}")

        uns = {"source": str(path), "gene_harmonization": uns_info(info),
               "note": "X copied unchanged from source (signatures x genes)"}

        def fill_x(f, path=path, n=len(cid), g_=len(rid), transposed=transposed):
            with h5py.File(path, "r") as s:
                src = s["0/DATA/0/matrix"]
                X = f.create_dataset("X", shape=(n, g_), dtype=src.dtype,
                                     chunks=(min(64, n), g_))
                step = 8192
                for i in range(0, n, step):
                    j = min(i + step, n)
                    X[i:j] = src[:, i:j].T if transposed else src[i:j]

        write_h5ad(dst, obs, var, uns, fill_x)
        print(f"  wrote {dst}  ({len(cid):,} x {len(rid):,})")


# --------------------------------------------------------------------------- #
# LINCS GMT
# --------------------------------------------------------------------------- #
DOSE_RE = re.compile(r"^[\d.]+(?:e-?\d+)?\s*(?:[munp]?M|ug/ml|ng/ml|%)?$", re.I)


def parse_set_name(name: str) -> dict:
    """'HAHN001_A549_96H_A03_AKT1 up' -> fields. Unparseable names keep set_name only."""
    row = {"set_name": name}
    base, direction = (name.rsplit(" ", 1) + [""])[:2] if " " in name else (name, "")
    t = base.split("_")
    if len(t) >= 5:
        rest, dose = t[4:], ""
        if len(rest) >= 2 and DOSE_RE.match(rest[-1]):
            dose, rest = rest[-1], rest[:-1]
        row.update(signature=base, plate=t[0], cell_line=t[1], time=t[2], well=t[3],
                   pert="_".join(rest), dose=dose, direction=direction)
    return row


def do_gmt(a, ref, perts) -> None:
    out = a.out / "lincs"
    out.mkdir(parents=True, exist_ok=True)
    gi = load_geneinfo(a.lincs)
    sym2ens = gi.drop_duplicates("gene_symbol").set_index("gene_symbol")["ensembl_id"]

    for path in expand(a.lincs, a.gmt):
        dst = out / path.name.replace(".gmt", ".harmonized.gmt")
        if dst.exists() and not a.overwrite:
            print(f"[skip] {dst.name} exists")
            continue
        print(f"\n=== {path.name}")

        names, members = [], set()
        with open(path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                names.append(parts[0])
                members.update(g for g in parts[2:] if g)
        uniq = sorted(members)
        t, info = harmonize_genes(uniq, [sym2ens.get(s, "") for s in uniq], ref)
        rename = dict(zip(uniq, t["gene_symbol"]))
        print(f"  {len(names):,} sets, {len(uniq):,} distinct member genes")
        print(label_line(info))

        with open(path) as fh, open(dst, "w") as fo:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                fo.write("\t".join(parts[:2] + [rename.get(g, g) for g in parts[2:] if g]) + "\n")

        sets = pd.DataFrame([parse_set_name(n) for n in names])
        genetic = "xpr" in path.name or "sh" in path.name or "oe" in path.name
        if genetic and "pert" in sets.columns:
            lab = sets["pert"].fillna("").astype(str).values
            new, rep = harmonize_labels(lab, [sym2ens.get(s, "") for s in lab], ref,
                                        extra_symbols=perts)
            sets.insert(1, "target_gene", new)
            print(summarize_labels(rep))
            if perts:
                print(f"  challenge perturbations present: "
                      f"{sum(p in set(new) for p in perts)}/{len(perts)}")
        sets.to_parquet(dst.with_suffix(".sets.parquet").with_name(
            path.name.replace(".gmt", ".sets.parquet")))
        print(f"  wrote {dst}")


# --------------------------------------------------------------------------- #
# coverage tables
# --------------------------------------------------------------------------- #
def do_coverage(a, ref, perts) -> None:
    import anndata as ad
    genes = pd.DataFrame({"gene_symbol": ref.symbols, "ensembl_id": ref.ensembl})
    pc = pd.DataFrame({"target_gene": perts or []})

    for f in sorted((a.out / "replogle").glob("*.h5ad")) + sorted((a.out / "lincs").glob("*.h5ad")):
        h = ad.read_h5ad(f, backed="r")
        name = f"{f.parent.name}:{f.stem}"
        idx = h.var["challenge_idx"].to_numpy()
        m = np.zeros(len(ref), dtype=bool)
        m[idx[idx >= 0]] = True
        genes[name] = m
        if "feature_space" in h.var.columns:
            lm = np.zeros(len(ref), dtype=bool)
            sel = (idx >= 0) & (h.var["feature_space"].to_numpy() == "landmark")
            lm[idx[sel]] = True
            genes[f"{name}:landmark"] = lm
        if perts:
            o = h.obs[["target_gene", "is_control"]]
            tg = o.loc[~o["is_control"].astype(bool), "target_gene"].astype(str)
            pc[name] = [p in set(tg) for p in perts]
            if "cell_iname" in h.obs.columns:
                n_lines = h.obs.loc[tg.index].groupby("target_gene", observed=True)[
                    "cell_iname"].nunique()
                pc[f"{name}:n_cell_lines"] = [int(n_lines.get(p, 0)) for p in perts]
        h.file.close()

    for f in sorted((a.out / "lincs").glob("*.sets.parquet")):
        s = pd.read_parquet(f)
        if "target_gene" in s.columns and perts:
            name = f"gmt:{f.name.replace('.sets.parquet', '')}"
            pc[name] = [p in set(s["target_gene"]) for p in perts]

    a.out.mkdir(parents=True, exist_ok=True)
    genes.to_csv(a.out / "gene_coverage.csv", index=False)
    pc.to_csv(a.out / "pert_coverage.csv", index=False)

    cols = [c for c in genes.columns if c not in ("gene_symbol", "ensembl_id")]
    print(f"\nchallenge genes ({len(ref):,}) measured in:")
    for c in cols:
        print(f"  {c:<55} {int(genes[c].sum()):>7,}")
    if perts:
        print(f"\nchallenge perturbations ({len(perts)}) present in:")
        for c in pc.columns[1:]:
            v = pc[c]
            s = f"{int((v > 0).sum())}" if v.dtype != bool else f"{int(v.sum())}"
            print(f"  {c:<55} {s:>7}")
        present = pc[[c for c in pc.columns[1:] if pc[c].dtype == bool]]
        if len(present.columns):
            print(f"  {'in at least one dataset':<55} {int(present.any(axis=1).sum()):>7}")
    print(f"\nWrote {a.out / 'gene_coverage.csv'} and {a.out / 'pert_coverage.csv'}")


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["replogle", "lincs", "gmt", "coverage", "all"])
    p.add_argument("--ref", type=Path, default=REF)
    p.add_argument("--perts", type=Path, default=PERTS)
    p.add_argument("--hgnc", type=Path, default=HGNC)
    p.add_argument("--replogle", type=Path, default=REPLOGLE)
    p.add_argument("--lincs", type=Path, default=LINCS)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--replogle-files", nargs="+",
                   default=["*bulk*.h5ad", "*raw_singlecell*.h5ad"],
                   help="glob(s) in --replogle (add '*normalized_singlecell*' if wanted)")
    p.add_argument("--gctx", nargs="+",
                   default=["level5_beta_trt_xpr*.gctx", "level5_beta_ctl*.gctx"],
                   help="glob(s) in --lincs")
    p.add_argument("--gmt", nargs="+", default=["l1000_xpr.gmt", "l1000_cp.gmt"])
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    ref = load_reference(a.ref, a.hgnc if a.hgnc and a.hgnc.exists() else None)
    perts = ([l.strip() for l in open(a.perts) if l.strip()]
             if a.perts and a.perts.exists() else None)

    steps = ["replogle", "lincs", "gmt", "coverage"] if a.step == "all" else [a.step]
    for s in steps:
        {"replogle": do_replogle, "lincs": do_lincs,
         "gmt": do_gmt, "coverage": do_coverage}[s](a, ref, perts)
    return 0


if __name__ == "__main__":
    sys.exit(main())