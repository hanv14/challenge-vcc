#!/usr/bin/env python
"""Measurement 2: data inventory for the redesign.

    python scripts/measure/data_inventory.py <data_root> <vcc_root>

The manifest (real cells per target), the challenge targets' coverage by
each Replogle screen and by LINCS, how strong their knockdowns are in
Replogle, and how the challenge contexts' control profiles compare with each
other, with themselves (split halves) and with the training cell lines.
Read-only.
"""
import glob, json, os, sys
import numpy as np, pandas as pd, anndata as ad
import scipy.sparse as sp

DATA, VCC = sys.argv[1], sys.argv[2]
P = lambda *a: os.path.join(DATA, *a)

m = json.load(open(os.path.join(VCC, "manifest.json")))
print("== manifest:", {k: v for k, v in m.items() if k != "per_context"})
for ctx, d in m["per_context"].items():
    print(f"   {ctx}: {d}  -> real cells/target ~ {d['ground_truth_cells'] / d['n_perturbations']:.0f}")

targets = pd.read_csv(os.path.join(VCC, "pert_counts.csv"))["target_gene"].astype(str).tolist()
T = set(targets)
print(f"\n== {len(T)} challenge targets")

# ---- Replogle coverage and effect strength
screens = sorted(os.path.basename(d) for d in glob.glob(P("processed/phases/phase2/*")))
all_scr = {}
for s in screens:
    obs = ad.read_h5ad(P("processed/phases/phase2", s, "pseudobulk.h5ad"), backed="r").obs
    pert = obs[~obs["is_control"].astype(bool)]
    all_scr[s] = set(pert.index.astype(str))
    n = pert["n_cells"]
    print(f"   {s}: {len(all_scr[s])} targets; covers {len(T & all_scr[s])}/{len(T)} challenge targets; "
          f"cells/target median {n.median():.0f} (challenge targets {n[n.index.isin(T)].median() if (n.index.isin(T)).any() else float('nan'):.0f})")
if screens:
    anyr = set().union(*all_scr.values())
    print(f"   in any Replogle screen: {len(T & anyr)}/{len(T)}")
    for a in screens:
        for b in screens:
            if a < b:
                print(f"   shared targets {a} & {b}: {len(all_scr[a] & all_scr[b])} (challenge: {len(T & all_scr[a] & all_scr[b])})")

for f in sorted(glob.glob(P("processed/replogle/*_bulk.h5ad"))):
    if "normalized" in f and os.path.exists(f.replace("normalized", "raw")):
        continue
    obs = ad.read_h5ad(f, backed="r").obs
    obs = obs[~obs["is_control"].astype(bool)].copy()
    g = obs.groupby(obs["target_gene"].astype(str)).agg(
        ad_counts=("anderson_darling_counts", "median"), fold_expr=("fold_expr", "median"))
    inT = g.index.isin(T)
    q = lambda x: np.nanpercentile(x, [25, 50, 75]).round(3).tolist() if len(x) else []
    print(f"\n   {os.path.basename(f)}: {len(g)} genes")
    print(f"     anderson_darling_counts  all {q(g.ad_counts)}  challenge {q(g.ad_counts[inT])}")
    print(f"     fold_expr                all {q(g.fold_expr)}  challenge {q(g.fold_expr[inT])}")
    if inT.any():
        pct = (g.ad_counts.rank(pct=True)[inT]).median()
        print(f"     challenge targets' median percentile of anderson_darling among all targets: {pct:.2f}")

# ---- LINCS coverage
lobs = ad.read_h5ad(P("processed/phases/phase1_lincs.h5ad"), backed="r").obs
lines_per_t = lobs.groupby(lobs["target_gene"].astype(str), observed=True)["cell_line"].nunique()
print(f"\n== LINCS: {lobs['cell_line'].nunique()} lines, {lines_per_t.size} targets, {len(lobs)} rows")
print("   lines per target (all):", lines_per_t.describe()[["50%", "75%", "max"]].to_dict(),
      "| targets in >=3 lines:", int((lines_per_t >= 3).sum()), "| >=10 lines:", int((lines_per_t >= 10).sum()))
lt = lines_per_t[lines_per_t.index.isin(T)]
print(f"   challenge targets in LINCS: {lt.size}/{len(T)}; in >=3 lines: {int((lt >= 3).sum())}")
print("   targets per line:", lobs.groupby("cell_line", observed=True)["target_gene"].nunique().sort_values(ascending=False).to_dict())

# ---- context similarity: challenge controls vs Replogle controls vs LINCS controls
genes = pd.read_csv(os.path.join(VCC, "gene_names.csv"))["gene_name"].astype(str).tolist()
ctx_prof = {}
half_rng = np.random.default_rng(0)
print("\n== challenge controls: library size, and the split-half self-correlation of the mean profile")
for f in sorted(glob.glob(P("processed/phases/phase3/controls_*.h5ad"))):
    a = ad.read_h5ad(f)
    X = a.X
    mean = np.asarray(X.mean(axis=0)).ravel() if sp.issparse(X) else X.mean(axis=0)
    name = os.path.basename(f)[9:-5]
    ctx_prof[name] = pd.Series(mean, index=a.var_names.astype(str))
    order = half_rng.permutation(a.n_obs)
    h1, h2 = order[: a.n_obs // 2], order[a.n_obs // 2:]
    m1 = np.asarray(X[h1].mean(axis=0)).ravel()
    m2 = np.asarray(X[h2].mean(axis=0)).ravel()
    lib = a.obs["library_size"].to_numpy(float) if "library_size" in a.obs else np.full(a.n_obs, np.nan)
    xmax = float(X.max()) if a.n_obs else float("nan")
    print(f"   {name}: {a.n_obs} cells, library median {np.nanmedian(lib):.0f}, "
          f"X max {xmax:.2f} (log1p CP10K is < ~9.3), half/half Pearson {np.corrcoef(m1, m2)[0, 1]:.4f}, "
          f"genes with mean > 0: {(mean > 0).sum()}")
rep_prof = {}
for s in screens:
    pb = ad.read_h5ad(P("processed/phases/phase2", s, "pseudobulk.h5ad"))
    row = pb[pb.obs["is_control"].astype(bool)].X
    rep_prof[s] = pd.Series(np.asarray(row).ravel(), index=pb.var_names.astype(str))
prof = pd.DataFrame({**ctx_prof, **rep_prof})
common = prof.dropna()
print(f"\n== control profiles, mean log1p(CP10K), {len(common)} genes measured everywhere")
print("   Pearson:\n" + common.corr().round(3).to_string())
print("   Spearman:\n" + common.corr(method="spearman").round(3).to_string())
dev = common.sub(common.mean(axis=1), axis=0)
print("   Pearson of deviations from the across-context mean:\n" + dev.corr().round(3).to_string())

L = ad.read_h5ad(P("processed/phases/phase1_lincs.h5ad"))
ctrl = pd.DataFrame(L.layers["ctrl"], index=L.obs_names, columns=L.var_names.astype(str))
lmean = ctrl.groupby(L.obs["cell_line"].astype(str).values).mean().T
panel = [g for g in lmean.index if g in prof.index and prof.loc[g, list(ctx_prof)].notna().all()]
ldev = lmean.loc[panel].sub(lmean.loc[panel].mean(axis=1), axis=0)
cdev = prof.loc[panel, list(ctx_prof)]
cdev = cdev.sub(cdev.mean(axis=1), axis=0)
print(f"\n   challenge-context deviation vs LINCS line deviation, Spearman over {len(panel)} panel genes (top 5 per context):")
for c in cdev:
    r = ldev.apply(lambda col: cdev[c].corr(col, method="spearman")).sort_values(ascending=False)
    print(f"   {c}: " + ", ".join(f"{k} {v:+.2f}" for k, v in r.head(5).items()))
