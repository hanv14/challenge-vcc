#!/usr/bin/env python3
"""
Build a small example of the processed data, for GitHub and for testing the
prototype end to end in the cloud.

Same folder layout, file formats and column names as the real data — only
fewer genes, cells and perturbations. Run on the server, after
`harmonize_data.py all` and `phase_data.py all`.

What is kept
------------
  genes          all LINCS panel genes + the selected challenge perturbations
                 + a random sample of other genes (re-indexed; vocabulary is
                 deliberately NOT 18,533 so no code can hardcode it)
  perturbations  --n-perts challenge targets, stratified by where they appear
                 (Replogle genome-wide, LINCS, both, neither)
  challenge      --cells-per-guide control cells per ntc_id guide, per context;
                 obs['total_counts'] keeps each cell's FULL library size
  Replogle       K562 genome-wide + RPE1 raw single-cell: controls, cells of the
                 selected challenge targets, and a sample of other targets;
                 matching rows of the bulk files
  LINCS          Phase 1 rows for the selected targets + a random sample
  phases/        rebuilt from the mini inputs with the same phase_data.py code

Layout written to --out
-----------------------
  ref/        challenge_genes.tsv  challenge_perts.txt  hgnc_subset.txt
  vcc/        manifest.json  gene_names.csv  pert_counts.csv  context_{A,B,C}.h5ad
  processed/  replogle/*.h5ad  pert_coverage.csv  gene_coverage.csv
              phases/{gene_split.csv, phase1_lincs.h5ad, phase2/<ctx>/, phase3/}

Usage
-----
  python make_mini_data.py --out /data/han/projects/VCC/repo/mini_data
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from genes import load_reference  # noqa: E402
from phase_data import do_phase2, do_phase3  # noqa: E402

DATA = Path("/data/han/projects/VCC/data")
VCC = Path("/data/share/han/VCC")


def _is_new_string(dtype) -> bool:
    """pandas' nullable / Arrow string dtypes (not the classic object dtype)."""
    return pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype)


def _obj(values) -> np.ndarray:
    return np.array(["" if pd.isna(v) else str(v) for v in values], dtype=object)


def plain_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert nullable/Arrow strings (index, columns, categories) to classic
    object strings. anndata < 0.12 refuses to write them by default, and files
    written with them cannot be read by older anndata — so the example data
    stays readable by any version."""
    df = df.copy()
    if _is_new_string(df.index.dtype):
        df.index = pd.Index(_obj(df.index), dtype=object, name=df.index.name)
    for c in df.columns:
        col = df[c]
        if isinstance(col.dtype, pd.CategoricalDtype):
            if _is_new_string(col.cat.categories.dtype):
                cats = pd.Index(_obj(col.cat.categories), dtype=object)
                df[c] = pd.Categorical.from_codes(col.cat.codes.to_numpy(), categories=cats)
        elif _is_new_string(col.dtype):
            df[c] = pd.Series(_obj(col), index=df.index, dtype=object)
    return df


def write_h5ad(adata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.obs = plain_strings(adata.obs)
    adata.var = plain_strings(adata.var)
    adata.write_h5ad(path, compression="gzip")
    print(f"  {path.relative_to(path.parents[2]) if len(path.parents) > 2 else path}"
          f"  {adata.shape}  {path.stat().st_size / 1e6:.1f} MB")


def read_rows(path: Path, rows: np.ndarray):
    """Read selected rows of X without loading the whole (possibly 65 GB) file."""
    import anndata as ad
    import h5py
    rows = np.sort(np.asarray(rows))
    with h5py.File(path, "r") as f:
        if isinstance(f["X"], h5py.Dataset):
            return sp.csr_matrix(f["X"][rows.tolist()].astype(np.float32))
    a = ad.read_h5ad(path, backed="r")
    X = a.X[rows]
    a.file.close()
    return sp.csr_matrix(X, dtype=np.float32)


def remap_var(var: pd.DataFrame, old2new: dict[int, int]) -> tuple[np.ndarray, pd.DataFrame]:
    """Columns whose challenge gene survives, with challenge_idx re-indexed."""
    ci = var["challenge_idx"].to_numpy()
    cols = np.flatnonzero(np.isin(ci, list(old2new)))
    v = var.iloc[cols].copy()
    v["challenge_idx"] = [old2new[int(c)] for c in ci[cols]]
    return cols, v


def pick_perts(perts, cov: pd.DataFrame | None, n: int, rng) -> list[str]:
    """Stratify by where each challenge target appears, so all code paths get exercised."""
    if cov is None:
        return list(rng.choice(perts, min(n, len(perts)), replace=False))
    cov = cov.set_index("target_gene").reindex(perts)
    bool_cols = [c for c in cov.columns if cov[c].dtype == bool or set(cov[c].dropna().unique()) <= {True, False}]
    rep = cov[[c for c in bool_cols if "gwps" in c and "singlecell" in c]].fillna(False).any(axis=1)
    lin = cov[[c for c in bool_cols if "trt_xpr" in c]].fillna(False).any(axis=1)
    group = rep.astype(int) * 2 + lin.astype(int)          # 0 neither, 1 LINCS, 2 Replogle, 3 both
    chosen = []
    for g, members in group.groupby(group):
        k = max(1, round(n * len(members) / len(perts)))
        chosen += list(rng.choice(members.index, min(k, len(members)), replace=False))
    chosen = chosen[:n]
    print(f"  challenge targets: {len(chosen)} "
          f"(neither/LINCS/Replogle/both: {[int((group[chosen] == g).sum()) for g in range(4)]})")
    return chosen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA, help="has ref/ and processed/")
    p.add_argument("--vcc", type=Path, default=VCC)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-perts", type=int, default=30)
    p.add_argument("--n-rest-genes", type=int, default=1000)
    p.add_argument("--cells-per-guide", type=int, default=10)
    p.add_argument("--cells-per-pert", type=int, default=50,
                   help="cells_per_pert written to the mini manifest (keeps test predictions small)")
    p.add_argument("--replogle", nargs="+",
                   default=["K562_gwps_raw_singlecell", "rpe1_raw_singlecell"])
    p.add_argument("--bulk", nargs="+",
                   default=["K562_gwps_raw_bulk", "K562_gwps_normalized_bulk", "rpe1_raw_bulk"])
    p.add_argument("--n-controls", type=int, default=1000)
    p.add_argument("--cells-challenge-target", type=int, default=60)
    p.add_argument("--n-other-targets", type=int, default=150,
                   help="non-challenge targets per Replogle file (includes the shared ones)")
    p.add_argument("--n-shared-targets", type=int, default=80,
                   help="targets present in every Replogle file, for cross-context rehearsal")
    p.add_argument("--cells-other-target", type=int, default=25)
    p.add_argument("--n-phase1-extra", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import anndata as ad
    rng = np.random.default_rng(a.seed)
    proc, ph = a.data / "processed", a.data / "processed" / "phases"
    out = a.out
    print(f"Building example data in {out}\n")

    # ---- 1. perturbations and gene vocabulary ------------------------------
    perts_all = [l.strip() for l in open(a.data / "ref/challenge_perts.txt") if l.strip()]
    cov_path = proc / "pert_coverage.csv"
    cov = pd.read_csv(cov_path) if cov_path.exists() else None
    perts = pick_perts(perts_all, cov, a.n_perts, rng)

    split = pd.read_csv(ph / "gene_split.csv", keep_default_na=False)
    panel = set(split.loc[split["split"] == "panel", "gene_symbol"])
    rest_pool = split.loc[(split["split"] == "rest") & ~split["gene_symbol"].isin(perts), "gene_symbol"]
    rest = set(rng.choice(rest_pool.values, min(a.n_rest_genes, len(rest_pool)), replace=False))
    keep = split[split["gene_symbol"].isin(panel | rest | set(perts))].copy()
    old2new = {int(o): i for i, o in enumerate(keep["challenge_idx"])}
    keep["challenge_idx"] = np.arange(len(keep))
    symbols = keep["gene_symbol"].tolist()
    print(f"  genes: {len(symbols):,} (panel {len(panel):,}, rest sample {len(rest):,}, "
          f"targets {len(perts)})")

    (out / "ref").mkdir(parents=True, exist_ok=True)
    keep[["gene_symbol", "ensembl_id"]].to_csv(out / "ref/challenge_genes.tsv", sep="\t",
                                               header=False, index=False)
    (out / "ref/challenge_perts.txt").write_text("\n".join(perts) + "\n")
    (out / "processed/phases").mkdir(parents=True, exist_ok=True)
    keep.to_csv(out / "processed/phases/gene_split.csv", index=False)

    hgnc = a.data / "ref/hgnc_complete_set.txt"
    if hgnc.exists():
        cols = ["hgnc_id", "symbol", "name", "locus_group", "locus_type", "gene_group",
                "gene_group_id", "ensembl_gene_id", "prev_symbol", "alias_symbol"]
        h = pd.read_csv(hgnc, sep="\t", dtype=str, low_memory=False,
                        usecols=lambda c: c in cols)
        ens = set(keep["ensembl_id"]) - {""}
        h = h[h["symbol"].isin(symbols) | h["ensembl_gene_id"].isin(ens)]
        h.to_csv(out / "ref/hgnc_subset.txt", sep="\t", index=False)
        print(f"  HGNC subset: {len(h):,} rows")

    # ---- 2. challenge release ----------------------------------------------
    print("\nchallenge release")
    (out / "vcc").mkdir(parents=True, exist_ok=True)
    for f in sorted(a.vcc.glob("context_*.h5ad")):
        h = ad.read_h5ad(f)
        C = sp.csr_matrix(h.X, dtype=np.float32)
        total = np.asarray(C.sum(1)).ravel()
        g = h.obs["ntc_id"].astype(str) if "ntc_id" in h.obs.columns else pd.Series("all", index=h.obs_names)
        rows = np.sort(np.concatenate([
            rng.choice(np.flatnonzero((g == k).values), min(a.cells_per_guide, int((g == k).sum())),
                       replace=False) for k in g.unique()]))
        col = pd.Index(h.var_names).get_indexer(symbols)
        if (col < 0).any():
            sys.exit(f"{f.name}: {int((col < 0).sum())} vocabulary genes missing")
        obs = h.obs.iloc[rows].copy()
        obs["total_counts"] = total[rows]
        write_h5ad(ad.AnnData(X=C[rows][:, col], obs=obs, var=pd.DataFrame(index=symbols)),
                   out / "vcc" / f.name)
    pd.DataFrame({"gene_name": symbols}).to_csv(out / "vcc/gene_names.csv", index=False)
    pd.DataFrame({"target_gene": perts}).to_csv(out / "vcc/pert_counts.csv", index=False)
    man = json.loads((a.vcc / "manifest.json").read_text()) if (a.vcc / "manifest.json").exists() else {}
    n_ctx_cells = a.cells_per_guide * int(man.get("per_context", {}).get("A", {}).get("n_ntc_ids", 46))
    man.update({"n_genes": len(symbols), "n_constructs": len(perts),
                "cells_per_pert": a.cells_per_pert, "mini_example": True})
    for c in man.get("per_context", {}):
        man["per_context"][c].update(n_perturbations=len(perts), control_cells=n_ctx_cells,
                                     ground_truth_cells=len(perts) * a.cells_per_pert + n_ctx_cells)
    (out / "vcc/manifest.json").write_text(json.dumps(man, indent=2))

    # ---- 3. Replogle single-cell + bulk ------------------------------------
    print("\nReplogle")
    # targets present in EVERY context file, so the cross-context rehearsal
    # (learn in K562, predict in RPE1) has shared targets to score on
    obs_of = {}
    for name in a.replogle:
        src = proc / "replogle" / f"{name}.h5ad"
        if src.exists():
            h = ad.read_h5ad(src, backed="r")
            obs_of[name] = h.obs[["target_gene", "is_control"]].copy()
            h.file.close()
    sets = [set(o.loc[~o["is_control"].astype(bool), "target_gene"].astype(str))
            for o in obs_of.values()]
    common = sorted(set.intersection(*sets) - set(perts)) if len(sets) > 1 else []
    shared = list(rng.choice(common, min(a.n_shared_targets, len(common)), replace=False)) \
        if common else []
    print(f"  targets shared by all {len(sets)} contexts: {len(common):,} (keeping {len(shared)})")

    chosen_targets: dict[str, set] = {}
    for name in a.replogle:
        src = proc / "replogle" / f"{name}.h5ad"
        if name not in obs_of:
            print(f"  [skip] {src} not found")
            continue
        h = ad.read_h5ad(src, backed="r")
        obs, var = h.obs.copy(), h.var.copy()
        h.file.close()
        tg = obs["target_gene"].astype(str).values
        ctrl = obs["is_control"].astype(bool).values
        rows = [rng.choice(np.flatnonzero(ctrl), min(a.n_controls, int(ctrl.sum())), replace=False)]
        present = [t for t in perts if (tg == t).any()]
        pool = sorted(set(tg[~ctrl]) - set(perts) - set(shared))
        n_extra = max(a.n_other_targets - len(shared), 0)
        others = list(shared) + list(rng.choice(pool, min(n_extra, len(pool)), replace=False))
        for t, k in [(t, a.cells_challenge_target) for t in present] + \
                    [(t, a.cells_other_target) for t in others]:
            r = np.flatnonzero(tg == t)
            rows.append(rng.choice(r, min(k, len(r)), replace=False))
        rows = np.sort(np.concatenate(rows))
        cols, v = remap_var(var, old2new)
        X = read_rows(src, rows)[:, cols]
        o = obs.iloc[rows].copy()
        write_h5ad(ad.AnnData(X=X, obs=o, var=v, uns={"mini_example": True}),
                   out / "processed/replogle" / f"{name}.h5ad")
        chosen_targets[name.split("_raw")[0].split("_normalized")[0]] = set(present) | set(others)
        print(f"    challenge targets present: {len(present)}, other targets: {len(others)}, "
              f"controls: {int(o['is_control'].sum())}")

    for name in a.bulk:
        src = proc / "replogle" / f"{name}.h5ad"
        if not src.exists():
            print(f"  [skip] {src} not found")
            continue
        b = ad.read_h5ad(src)
        key = name.split("_raw")[0].split("_normalized")[0]
        want = chosen_targets.get(key, set(perts))
        m = b.obs["is_control"].astype(bool).values | b.obs["target_gene"].astype(str).isin(want).values
        cols, v = remap_var(b.var, old2new)
        write_h5ad(ad.AnnData(X=np.asarray(b.X[np.flatnonzero(m)][:, cols], dtype=np.float32),
                              obs=b.obs[m].copy(), var=v), out / "processed/replogle" / f"{name}.h5ad")

    # ---- 4. Phase 1 (LINCS) ------------------------------------------------
    print("\nLINCS phase 1")
    p1 = ad.read_h5ad(ph / "phase1_lincs.h5ad")
    tg = p1.obs["target_gene"].astype(str).values
    sel = np.flatnonzero(np.isin(tg, perts))
    other = np.flatnonzero(~np.isin(tg, perts))
    sel = np.sort(np.concatenate([sel, rng.choice(other, min(a.n_phase1_extra, len(other)), replace=False)]))
    cols, v = remap_var(p1.var, old2new)
    sub = p1[sel][:, cols].copy()
    sub.var = v
    sub.obs["is_challenge_pert"] = sub.obs["target_gene"].astype(str).isin(perts).values
    write_h5ad(sub, out / "processed/phases/phase1_lincs.h5ad")

    # ---- 5. coverage tables ------------------------------------------------
    if cov is not None:
        cov[cov["target_gene"].isin(perts)].to_csv(out / "processed/pert_coverage.csv", index=False)
    gc = proc / "gene_coverage.csv"
    if gc.exists():
        g = pd.read_csv(gc, keep_default_na=False)
        g = g[g["gene_symbol"].isin(symbols)].copy()
        g.to_csv(out / "processed/gene_coverage.csv", index=False)

    # ---- 6. rebuild Phase 2 and Phase 3 with the same code as the real data --
    print("\nrebuilding phases 2 and 3 from the mini inputs")
    ns = Namespace(processed=out / "processed", out=out / "processed/phases", vcc=out / "vcc",
                   phase2_files="*raw_singlecell.h5ad", panel="landmark")
    ref = load_reference(out / "ref/challenge_genes.tsv", None, verbose=False)
    msplit = pd.read_csv(out / "processed/phases/gene_split.csv", keep_default_na=False)
    do_phase2(ns, ref, perts, msplit)
    do_phase3(ns, ref, perts, msplit)

    # ---- 7. size report ----------------------------------------------------
    files = [f for f in out.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    big = [f for f in files if f.stat().st_size > 25e6]
    print(f"\nTotal {total / 1e6:.1f} MB in {len(files)} files")
    for f in big:
        print(f"  [warn] {f.relative_to(out)} is {f.stat().st_size / 1e6:.1f} MB "
              f"(GitHub warns above 50 MB) — lower the --n-* / --cells-* options")
    if total > 90e6:
        print("  [warn] total above 90 MB — consider smaller options")
    return 0


if __name__ == "__main__":
    sys.exit(main())