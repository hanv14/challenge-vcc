#!/usr/bin/env python3
"""
Build the data each phase of the method needs, from the harmonized data.

All three phases share ONE gene coordinate system: challenge genes in
challenge order (gene_names.csv). Column i always means the same gene.

Gene split (all phases)                              phases/gene_split.csv
    panel = LINCS landmark genes that are challenge genes  (--panel all: + inferred)
    rest  = every other challenge gene

Phase 1 — LINCS                                      phases/phase1_lincs.h5ad
    one row per (cell_line, target_gene, time_h), CRISPR only; var = panel genes
    layers['ctrl']  control expression   Level 3, same-plate control wells, mean
    layers['pert']  perturbed expression Level 3, treated wells, mean
    layers['sig']   signature            Level 5 z-scores, mean over the key's signatures
    layers['gmt']   GMT signature        (#up-sets - #down-sets) / #signatures, in [-1, 1]
    X = layers['pert']
    obs: n_pert_wells, n_ctrl_wells, ctrl_source, n_sigs, has_sig, n_gmt_signatures,
         has_gmt, cc_q75_median, is_challenge_pert
    units: Level 3 = log2 normalized expression; Level 5 = robust z vs plate controls

Phase 2 — Replogle                                   phases/phase2/<context>/
    genes.csv        challenge genes measured here: split (panel/rest), source_col
    cells.parquet    one row per cell: row, target_gene, is_control, gem_group, library_size
    pseudobulk.h5ad  one row per target (+ non-targeting): mean log1p(CP10K);
                     var ctrl_mean / ctrl_std from control cells
    Cells are NOT copied. ReplogleCells (below) reads them from the harmonized
    .h5ad and returns (panel, rest) matrices, normalized, on request.

Phase 3 — challenge                                  phases/phase3/
    genes.csv            all challenge genes with split
    controls_<ctx>.h5ad  control cells: X = log1p(CP10K) (sparse), layers['counts'] = raw,
                         var ctrl_mean / ctrl_std, obs library_size
    targets.csv          context x target_gene to predict, n_cells, and which
                         training sources contain the target

Normalization (Replogle and challenge): log1p(counts / library_size * 1e4)
    library_size = Replogle obs['UMI_count'] (all UMIs of the cell) if present,
    otherwise the row sum (challenge: sum over all 18,533 genes).

Usage
-----
    python phase_data.py split
    python phase_data.py phase1
    python phase_data.py phase2
    python phase_data.py phase3
    python phase_data.py all

Using Phase 2 cells
-------------------
    from phase_data import ReplogleCells
    rc = ReplogleCells("/data/han/projects/VCC/data/processed/phases/phase2/K562_gwps")
    rng = np.random.default_rng(0)
    ctrl_panel, ctrl_rest = rc.controls(256, rng)
    kd_panel,   kd_rest   = rc.cells("ADNP", 256, rng)

Requires genes.py and harmonize_data.py in the same folder, and the outputs of
`harmonize_data.py all`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from genes import harmonize_labels, load_reference  # noqa: E402
from harmonize_data import clean_frame, gctx_ids, gene_table, load_geneinfo  # noqa: E402

DATA = Path("/data/han/projects/VCC/data")
PROC = DATA / "processed"
LINCS = Path("/data/share/han/Spatial3D/data_LINCS")
VCC = Path("/data/share/han/VCC")
CTRL = "non-targeting"
SCALE = 1e4


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #
def log1p_cp10k(counts: np.ndarray, lib: np.ndarray) -> np.ndarray:
    lib = np.where(lib > 0, lib, 1.0).astype(np.float64)
    return np.log1p(counts * (SCALE / lib)[:, None]).astype(np.float32)


def hours(values) -> np.ndarray:
    """96, '96', 96.0, '96H', '96 h' -> 96; '8D' -> 192 (days); unknown -> -1.

    A bare number is taken as hours (LINCS pert_time). A string needs an
    explicit unit: H/h/hr = hours, D/d/day = days. Anything else (e.g. 'XH',
    the LINCS placeholder) is unknown.
    """
    s = pd.Series(values).astype(str).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    m = s.str.extract(r"^([\d.]+)\s*([A-Za-z]*)$")
    val = pd.to_numeric(m[0], errors="coerce")
    unit = m[1].fillna("").str.lower()
    mult = unit.map({"": 1.0, "h": 1.0, "hr": 1.0, "hrs": 1.0, "hour": 1.0, "hours": 1.0,
                     "d": 24.0, "day": 24.0, "days": 24.0})
    num = num.fillna(val * mult)
    return num.fillna(-1).round().astype(int).values


def lincs_cell(values) -> pd.Series:
    """'A375.311' -> 'A375': GMT names carry a Cas9-clone suffix that
    siginfo/instinfo cell_iname do not."""
    return pd.Series(values).astype(str).str.replace(r"\.\d+$", "", regex=True)


def write(path: Path, X, obs, var, layers=None, uns=None) -> None:
    import anndata as ad
    path.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=X, obs=clean_frame(obs), var=clean_frame(var),
               layers=layers or {}, uns=uns or {}).write_h5ad(path, compression="gzip")
    print(f"  wrote {path}")


def gene_split(a, ref) -> pd.DataFrame:
    """Challenge genes (challenge order) with panel/rest split and LINCS Entrez ID."""
    gi = load_geneinfo(a.lincs)
    ens = gi["ensembl_id"].fillna("").values if "ensembl_id" in gi.columns else None
    var, _ = gene_table(gi["gene_symbol"].values, ens, ref)
    var["entrez_id"] = gi["gene_id"].values
    var["feature_space"] = gi["feature_space"].values if "feature_space" in gi.columns else ""
    hit = var[var["challenge_idx"] >= 0].drop_duplicates("challenge_idx")

    split = pd.DataFrame({"gene_symbol": ref.symbols, "ensembl_id": ref.ensembl,
                          "challenge_idx": np.arange(len(ref))})
    split["lincs_entrez"] = ""
    split["lincs_feature_space"] = ""
    split.loc[hit["challenge_idx"].values, "lincs_entrez"] = hit["entrez_id"].values
    split.loc[hit["challenge_idx"].values, "lincs_feature_space"] = hit["feature_space"].values
    in_panel = (split["lincs_feature_space"] == "landmark") if a.panel == "landmark" \
        else (split["lincs_entrez"] != "")
    split["split"] = np.where(in_panel, "panel", "rest")
    return split


def do_split(a, ref, perts) -> pd.DataFrame:
    s = gene_split(a, ref)
    out = a.out / "gene_split.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out, index=False)
    print(f"\ngene split ({a.panel}): panel {int((s['split'] == 'panel').sum()):,}, "
          f"rest {int((s['split'] == 'rest').sum()):,}  ->  {out}")
    return s


# --------------------------------------------------------------------------- #
# Phase 1 — LINCS
# --------------------------------------------------------------------------- #
def gctx_group_means(path: Path, col_group: dict[str, int], n_groups: int,
                     gene_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over GCTX columns (wells/signatures) per group, for selected genes."""
    import h5py
    from tqdm import tqdm
    with h5py.File(path, "r") as f:
        cid, rid, M, transposed = gctx_ids(f)
        g_all = np.array([col_group.get(c, -1) for c in cid], dtype=np.int64)
        big = n_groups * len(gene_rows) > 2e8
        sums = np.zeros((n_groups, len(gene_rows)), dtype=np.float32 if big else np.float64)
        counts = np.bincount(g_all[g_all >= 0], minlength=n_groups)
        step = 8192
        for i in tqdm(range(0, len(cid), step), desc=f"  {path.name[:40]}", unit="blk"):
            j = min(i + step, len(cid))
            g = g_all[i:j]
            v = g >= 0
            if not v.any():
                continue
            B = (M[:, i:j].T if transposed else M[i:j])[v][:, gene_rows]
            S = sp.csr_matrix((np.ones(v.sum()), (g[v], np.arange(v.sum()))),
                              shape=(n_groups, v.sum()))
            sums += S @ B.astype(sums.dtype)
    means = sums / np.maximum(counts, 1)[:, None]
    return means.astype(np.float32), counts


def do_phase1(a, ref, perts, split) -> None:
    import h5py
    lincs = a.lincs
    find = lambda pat: next(iter(sorted(lincs.glob(pat))), None)
    l3_trt, l3_ctl, l5_trt = (find("level3_beta_trt_xpr*.gctx"),
                              find("level3_beta_ctl*.gctx"), find("level5_beta_trt_xpr*.gctx"))
    for n, p in (("Level 3 trt_xpr", l3_trt), ("Level 3 ctl", l3_ctl), ("Level 5 trt_xpr", l5_trt)):
        if p is None:
            sys.exit(f"missing {n} .gctx in {lincs}")
    print(f"\n=== Phase 1 (LINCS)\n  {l3_trt.name}\n  {l3_ctl.name}\n  {l5_trt.name}")

    gi = load_geneinfo(lincs)
    sym2ens = gi.drop_duplicates("gene_symbol").set_index("gene_symbol")["ensembl_id"] \
        if "ensembl_id" in gi.columns else pd.Series(dtype=str)

    def harm(names) -> np.ndarray:
        names = np.asarray(names, dtype=object)
        new, _ = harmonize_labels(names, [sym2ens.get(s, "") for s in names], ref,
                                  extra_symbols=perts)
        return np.asarray(new, dtype=object)

    # ---- panel genes -> GCTX rows (Entrez) ---------------------------------
    panel = split[split["split"] == "panel"]
    with h5py.File(l3_trt, "r") as f:
        _, rid, _, _ = gctx_ids(f)
    rpos = {r: j for j, r in enumerate(rid)}
    panel = panel[panel["lincs_entrez"].isin(rpos)]
    gene_rows = np.array([rpos[e] for e in panel["lincs_entrez"]], dtype=np.int64)
    print(f"  panel genes: {len(panel):,}")

    # ---- instances: treated (trt_xpr) and controls --------------------------
    inst = pd.read_csv(lincs / "instinfo_beta.txt", sep="\t", dtype=str, low_memory=False)
    plate_col = next((c for c in ("det_plate", "rna_plate", "plate") if c in inst.columns), None)
    tcol = "pert_time" if "pert_time" in inst.columns else "pert_itime"
    print(f"  control matching by: {'cell line + ' + plate_col if plate_col else 'cell line only'}")

    ti = inst[inst["pert_type"] == "trt_xpr"].copy()
    ti["target_gene"] = harm(ti["cmap_name"].values)
    ti["time_h"] = hours(ti[tcol])
    ti["key"] = ti["cell_iname"] + "|" + ti["target_gene"] + "|" + ti["time_h"].astype(str)
    keys = pd.Index(sorted(ti["key"].unique()))
    k_of = pd.Series(np.arange(len(keys)), index=keys)
    ti["k"] = k_of[ti["key"]].values

    ci = inst[inst["pert_type"].str.startswith("ctl")].copy()
    ci["cg"] = ci["cell_iname"] + "|" + (ci[plate_col] if plate_col else "")
    cgroups = pd.Index(sorted(ci["cg"].unique()))
    cg_of = pd.Series(np.arange(len(cgroups)), index=cgroups)

    # ---- Level 3 means ------------------------------------------------------
    pert_mean, n_pert = gctx_group_means(l3_trt, dict(zip(ti["sample_id"], ti["k"])),
                                         len(keys), gene_rows)
    ctrl_mean_g, n_ctrl_g = gctx_group_means(
        l3_ctl, dict(zip(ci["sample_id"], cg_of[ci["cg"]].values)), len(cgroups), gene_rows)

    # ---- pair each key with its plates' controls ----------------------------
    with h5py.File(l3_trt, "r") as f:
        present = set(gctx_ids(f)[0])
    tp = ti[ti["sample_id"].isin(present)].copy()
    tp["cg"] = tp["cell_iname"] + "|" + (tp[plate_col] if plate_col else "")
    tp = tp[tp["cg"].isin(cg_of.index)]
    W = sp.csr_matrix((np.ones(len(tp)), (tp["k"].values, cg_of[tp["cg"]].values)),
                      shape=(len(keys), len(cgroups)))
    W = W.multiply((n_ctrl_g > 0)[None, :]).tocsr()          # ignore plates without controls
    tot = np.asarray(W.sum(1)).ravel()
    ctrl = (sp.diags(1 / np.maximum(tot, 1)) @ W) @ ctrl_mean_g
    n_ctrl = np.asarray(((W > 0).astype(np.float64) @ n_ctrl_g[:, None])).ravel()
    source = np.where(tot > 0, "plate" if plate_col else "cell_line", "none").astype(object)

    # fallback: keys without same-plate controls get their cell line's control mean
    cell_of_cg = pd.Series([c.split("|")[0] for c in cgroups])
    miss = np.flatnonzero(tot == 0)
    if len(miss):
        cells = pd.Series([k.split("|")[0] for k in keys])
        for cell in cells.iloc[miss].unique():
            m = (cell_of_cg == cell).values & (n_ctrl_g > 0)
            if m.any():
                w = n_ctrl_g[m] / n_ctrl_g[m].sum()
                rows = miss[(cells.iloc[miss] == cell).values]
                ctrl[rows] = w @ ctrl_mean_g[m]
                n_ctrl[rows] = n_ctrl_g[m].sum()
                source[rows] = "cell_line"
    ctrl = np.asarray(ctrl, dtype=np.float32)

    # ---- Level 5 signatures -------------------------------------------------
    si = pd.read_csv(lincs / "siginfo_beta.txt", sep="\t", dtype=str, low_memory=False)
    si = si[si["pert_type"] == "trt_xpr"].copy()
    si["key"] = (si["cell_iname"] + "|" + harm(si["cmap_name"].values) + "|"
                 + hours(si["pert_time" if "pert_time" in si.columns else "pert_itime"]).astype(str))
    si = si[si["key"].isin(k_of.index)]
    si["k"] = k_of[si["key"]].values
    sig, n_sig = gctx_group_means(l5_trt, dict(zip(si["sig_id"], si["k"])), len(keys), gene_rows)
    cc = (pd.to_numeric(si["cc_q75"], errors="coerce").groupby(si["k"]).median()
          if "cc_q75" in si.columns else pd.Series(dtype=float))

    # ---- GMT signatures -----------------------------------------------------
    gmt = np.zeros((len(keys), len(panel)), dtype=np.float32)
    n_gmt = np.zeros(len(keys), dtype=int)
    gmt_file = a.processed / "lincs" / "l1000_xpr.harmonized.gmt"
    sets_file = a.processed / "lincs" / "l1000_xpr.sets.parquet"
    if gmt_file.exists() and sets_file.exists():
        sets = pd.read_parquet(sets_file)
        skey = (lincs_cell(sets["cell_line"]).values + "|" + sets["target_gene"].astype(str).values
                + "|" + hours(sets["time"]).astype(str))
        skey = pd.Series(skey, index=sets.index)
        print(f"  GMT sets matched to a Phase 1 row: {skey.isin(k_of.index).mean():.1%}")
        sk = skey.map(k_of).fillna(-1).astype(int).values
        sgn = sets["direction"].map({"up": 1, "dn": -1, "down": -1}).fillna(0).values
        col = {g: j for j, g in enumerate(panel["gene_symbol"])}
        with open(gmt_file) as fh:
            for i, line in enumerate(fh):
                if sk[i] < 0 or sgn[i] == 0:
                    continue
                cols = [col[g] for g in line.rstrip("\n").split("\t")[2:] if g in col]
                gmt[sk[i], cols] += sgn[i]
        n_gmt = (sets.assign(k=sk)[sk >= 0].groupby("k")["signature"].nunique()
                 .reindex(range(len(keys)), fill_value=0).values)
        gmt /= np.maximum(n_gmt, 1)[:, None]
    else:
        print(f"  [note] no harmonized xpr GMT in {a.processed / 'lincs'}; layers['gmt'] left empty")

    # ---- assemble -----------------------------------------------------------
    keep = n_pert > 0
    parts = [k.split("|") for k in keys]
    obs = pd.DataFrame({
        "cell_line": [p[0] for p in parts], "target_gene": [p[1] for p in parts],
        "time_h": [int(p[2]) for p in parts],
        "n_pert_wells": n_pert, "n_ctrl_wells": n_ctrl.astype(int), "ctrl_source": source,
        "n_sigs": n_sig, "has_sig": n_sig > 0,
        "n_gmt_signatures": n_gmt, "has_gmt": n_gmt > 0,
        "cc_q75_median": cc.reindex(range(len(keys))).values if len(cc) else np.nan,
    }, index=keys)
    obs["is_challenge_pert"] = obs["target_gene"].isin(set(perts or []))
    var = panel.set_index("gene_symbol")[["challenge_idx", "ensembl_id", "lincs_entrez"]]

    obs, sl = obs[keep], np.flatnonzero(keep)
    write(a.out / "phase1_lincs.h5ad", X=pert_mean[sl], obs=obs, var=var,
          layers={"ctrl": ctrl[sl], "pert": pert_mean[sl], "sig": sig[sl], "gmt": gmt[sl]},
          uns={"panel": a.panel, "units": {"ctrl": "LINCS level3 log2", "pert": "LINCS level3 log2",
                                           "sig": "LINCS level5 z", "gmt": "signed set fraction"}})
    print(f"  {len(obs):,} (cell_line, target, time) rows; {obs['target_gene'].nunique():,} targets "
          f"in {obs['cell_line'].nunique()} cell lines")
    print(f"  with signature {int(obs['has_sig'].sum()):,}, with GMT {int(obs['has_gmt'].sum()):,}; "
          f"controls: {obs['ctrl_source'].value_counts().to_dict()}")
    if perts:
        cp = obs[obs["is_challenge_pert"]]
        print(f"  challenge perturbations covered: {cp['target_gene'].nunique()}/{len(perts)}")


# --------------------------------------------------------------------------- #
# Phase 2 — Replogle
# --------------------------------------------------------------------------- #
def do_phase2(a, ref, perts, split) -> None:
    import anndata as ad
    from tqdm import tqdm
    in_panel = (split["split"] == "panel").values
    for path in sorted((a.processed / "replogle").glob(a.phase2_files)):
        ctx = path.stem.replace("_raw_singlecell", "").replace("_singlecell", "")
        out = a.out / "phase2" / ctx
        print(f"\n=== Phase 2: {path.name} -> {out}")
        h = ad.read_h5ad(path, backed="r")

        cidx = h.var["challenge_idx"].to_numpy()
        cols = np.flatnonzero(cidx >= 0)
        cols = cols[np.argsort(cidx[cols])]                   # challenge order
        genes = pd.DataFrame({"gene_symbol": h.var_names[cols], "challenge_idx": cidx[cols],
                              "source_col": cols,
                              "split": np.where(in_panel[cidx[cols]], "panel", "rest")})

        obs = h.obs
        cells = pd.DataFrame({"row": np.arange(h.n_obs),
                              "target_gene": obs["target_gene"].astype(str).values,
                              "is_control": obs["is_control"].astype(bool).values})
        if "gem_group" in obs.columns:
            cells["gem_group"] = obs["gem_group"].values
        use_umi = "UMI_count" in obs.columns
        lib = obs["UMI_count"].to_numpy(dtype=np.float64) if use_umi else np.zeros(h.n_obs)

        targets = [CTRL] + sorted(set(cells["target_gene"]) - {CTRL})
        code = pd.Categorical(cells["target_gene"], categories=targets).codes
        sums = np.zeros((len(targets), len(cols)), dtype=np.float64)
        csum = np.zeros(len(cols)); csq = np.zeros(len(cols))
        ctrl = cells["is_control"].to_numpy()

        step = 16384
        for i in tqdm(range(0, h.n_obs, step), desc="  streaming", unit="blk"):
            j = min(i + step, h.n_obs)
            B = h.X[i:j]
            B = B.toarray() if hasattr(B, "toarray") else np.asarray(B)
            if not use_umi:
                lib[i:j] = B.sum(1)
            L = log1p_cp10k(B[:, cols], lib[i:j])
            S = sp.csr_matrix((np.ones(j - i), (code[i:j], np.arange(j - i))),
                              shape=(len(targets), j - i))
            sums += S @ L
            c = ctrl[i:j]
            csum += L[c].sum(0); csq += (L[c].astype(np.float64) ** 2).sum(0)
        h.file.close()

        n = np.bincount(code, minlength=len(targets))
        nc = max(int(ctrl.sum()), 1)
        mu = csum / nc
        genes["ctrl_mean"] = mu.astype(np.float32)
        genes["ctrl_std"] = np.sqrt(np.maximum(csq / nc - mu ** 2, 0)).astype(np.float32)
        cells["library_size"] = lib.astype(np.float32)

        out.mkdir(parents=True, exist_ok=True)
        genes.to_csv(out / "genes.csv", index=False)
        cells.to_parquet(out / "cells.parquet")
        pobs = pd.DataFrame({"n_cells": n, "is_control": [t == CTRL for t in targets],
                             "is_challenge_pert": [t in set(perts or []) for t in targets]},
                            index=targets)
        write(out / "pseudobulk.h5ad", X=(sums / np.maximum(n, 1)[:, None]).astype(np.float32),
              obs=pobs, var=genes.set_index("gene_symbol").drop(columns=["source_col"]),
              uns={"units": "mean log1p(CP10K)"})
        meta = {"source": str(path), "source_rel": os.path.relpath(path, out),
                "context": ctx, "n_cells": int(len(cells)),
                "n_controls": int(ctrl.sum()), "n_targets": len(targets) - 1,
                "n_panel": int((genes["split"] == "panel").sum()),
                "n_rest": int((genes["split"] == "rest").sum()),
                "library_size": "obs UMI_count" if use_umi else "row sum",
                "normalization": "log1p(counts / library_size * 1e4)"}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  {meta['n_cells']:,} cells, {meta['n_targets']:,} targets, {meta['n_controls']:,} "
              f"controls; genes: panel {meta['n_panel']:,}, rest {meta['n_rest']:,}; "
              f"library size from {meta['library_size']}")


class ReplogleCells:
    """Read Phase 2 cells on demand, returned as (panel, rest) log1p(CP10K) arrays."""

    def __init__(self, ctx_dir):
        import h5py
        d = Path(ctx_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.genes = pd.read_csv(d / "genes.csv")
        self.obs = pd.read_parquet(d / "cells.parquet")
        src = Path(self.meta["source"])
        if not src.exists() and self.meta.get("source_rel"):
            src = (d / self.meta["source_rel"]).resolve()
        self.source = src
        self._f = h5py.File(src, "r")
        self._X = self._f["X"]
        self._dense = isinstance(self._X, h5py.Dataset)
        self._lib = self.obs["library_size"].to_numpy(np.float64)
        self._cols = self.genes["source_col"].to_numpy()
        self._panel = (self.genes["split"] == "panel").to_numpy()
        self._rows = self.obs.groupby("target_gene")["row"].apply(np.asarray).to_dict()
        if not self._dense:
            import anndata as ad
            self._ad = ad.read_h5ad(src, backed="r")

    @property
    def targets(self) -> list[str]:
        return [t for t in self._rows if t != CTRL]

    def get(self, rows) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(rows)
        order = np.argsort(rows)
        srt = rows[order]
        B = self._X[srt.tolist()] if self._dense else self._ad.X[srt]
        B = B.toarray() if hasattr(B, "toarray") else np.asarray(B)
        L = log1p_cp10k(B[:, self._cols], self._lib[srt])
        L = L[np.argsort(order)]                              # back to requested order
        return L[:, self._panel], L[:, ~self._panel]

    def cells(self, target: str, n: int | None = None, rng=None):
        r = self._rows[target]
        if n is not None and n < len(r):
            r = (rng or np.random.default_rng()).choice(r, n, replace=False)
        return self.get(r)

    def controls(self, n: int | None = None, rng=None):
        return self.cells(CTRL, n, rng)


# --------------------------------------------------------------------------- #
# Phase 3 — challenge
# --------------------------------------------------------------------------- #
def do_phase3(a, ref, perts, split) -> None:
    import anndata as ad
    out = a.out / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Phase 3 (challenge) -> {out}")
    split[["gene_symbol", "challenge_idx", "split"]].to_csv(out / "genes.csv", index=False)
    man = json.loads((a.vcc / "manifest.json").read_text()) if (a.vcc / "manifest.json").exists() else {}
    n_per = int(man.get("cells_per_pert", 400))

    contexts = []
    for path in sorted(a.vcc.glob("context_*.h5ad")):
        h = ad.read_h5ad(path)
        ctx = (str(h.obs["context"].iloc[0]) if "context" in h.obs.columns
               else path.stem.replace("context_", ""))
        contexts.append(ctx)
        if list(h.var_names) != list(ref.symbols):
            missing = set(ref.symbols) - set(h.var_names)
            if missing:
                sys.exit(f"{path.name}: {len(missing)} challenge genes missing, e.g. {list(missing)[:5]}")
            h = h[:, ref.symbols].copy()
        C = sp.csr_matrix(h.X, dtype=np.float32)
        lib = (h.obs["total_counts"].to_numpy(np.float64) if "total_counts" in h.obs.columns
               else np.asarray(C.sum(1)).ravel())
        L = (sp.diags((SCALE / np.maximum(lib, 1)).astype(np.float32)) @ C).tocsr()
        L.data = np.log1p(L.data)
        mu = np.asarray(L.mean(0)).ravel()
        sd = np.sqrt(np.maximum(np.asarray(L.multiply(L).mean(0)).ravel() - mu ** 2, 0))

        obs = h.obs.copy()
        obs["library_size"] = lib
        var = split.set_index("gene_symbol")[["challenge_idx", "split"]].copy()
        var["ctrl_mean"], var["ctrl_std"] = mu.astype(np.float32), sd.astype(np.float32)
        write(out / f"controls_{ctx}.h5ad", X=L, obs=obs, var=var, layers={"counts": C},
              uns={"units": "log1p(CP10K); layers['counts'] raw", "context": ctx})
        print(f"  context {ctx}: {h.n_obs:,} control cells, median library {np.median(lib):,.0f}")

    t = pd.DataFrame([(c, p) for c in contexts for p in (perts or [])],
                     columns=["context", "target_gene"])
    t["n_cells"] = n_per
    cov = a.processed / "pert_coverage.csv"
    if cov.exists():
        t = t.merge(pd.read_csv(cov), on="target_gene", how="left")
    p1 = a.out / "phase1_lincs.h5ad"
    if p1.exists():
        o = ad.read_h5ad(p1, backed="r").obs
        agg = o.groupby("target_gene", observed=True).agg(phase1_n_cell_lines=("cell_line", "nunique"),
                                           phase1_has_sig=("has_sig", "any"),
                                           phase1_has_gmt=("has_gmt", "any"))
        t = t.merge(agg, left_on="target_gene", right_index=True, how="left")
    t.to_csv(out / "targets.csv", index=False)
    print(f"  targets: {len(t):,} rows ({len(contexts)} contexts x {len(perts or [])} perturbations, "
          f"{n_per} cells each) -> {out / 'targets.csv'}")


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["split", "phase1", "phase2", "phase3", "all"])
    p.add_argument("--panel", choices=["landmark", "all"], default="landmark",
                   help="LINCS panel: 978 measured landmarks (default) or all 12,328 genes")
    p.add_argument("--ref", type=Path, default=DATA / "ref/challenge_genes.tsv")
    p.add_argument("--perts", type=Path, default=DATA / "ref/challenge_perts.txt")
    p.add_argument("--hgnc", type=Path, default=DATA / "ref/hgnc_complete_set.txt")
    p.add_argument("--lincs", type=Path, default=LINCS)
    p.add_argument("--vcc", type=Path, default=VCC)
    p.add_argument("--processed", type=Path, default=PROC)
    p.add_argument("--out", type=Path, default=PROC / "phases")
    p.add_argument("--phase2-files", default="*raw_singlecell.h5ad",
                   help="glob in processed/replogle")
    a = p.parse_args()

    ref = load_reference(a.ref, a.hgnc if a.hgnc.exists() else None)
    perts = [l.strip() for l in open(a.perts) if l.strip()] if a.perts.exists() else []

    split = do_split(a, ref, perts)
    steps = {"split": [], "phase1": [do_phase1], "phase2": [do_phase2],
             "phase3": [do_phase3], "all": [do_phase1, do_phase2, do_phase3]}[a.step]
    for fn in steps:
        fn(a, ref, perts, split)
    return 0


if __name__ == "__main__":
    sys.exit(main())