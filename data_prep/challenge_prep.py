#!/usr/bin/env python3
"""
Build the reference files every other script maps into, from the challenge release.

Reads   /data/share/han/VCC/{gene_names.csv, pert_counts.csv, context_*.h5ad}
Writes  <out>/challenge_genes.tsv     symbol<TAB>ensembl, in OFFICIAL gene_names.csv order
        <out>/challenge_perts.txt     perturbations to predict, one symbol per line
        <out>/challenge_summary.json  everything this script found

The gene order comes from gene_names.csv, not from the .h5ad, because that is
the order a submission must use: column i of every processed dataset then
equals column i of the final prediction.

Usage
-----
    python challenge_prep.py
    python challenge_prep.py --vcc /data/share/han/VCC --out /data/han/projects/VCC/data/ref
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from genes import strip_version, var_ids  # noqa: E402

VCC = Path("/data/share/han/VCC")
OUT = Path("/data/han/projects/VCC/data/ref")
HEADER_WORDS = {"gene", "genes", "gene_name", "gene_names", "gene_symbol", "symbol",
                "target", "target_gene", "perturbation", "pert", "name", "gene_id",
                "ensembl", "ensembl_id", "count", "counts", "n_cells", "n"}
CONTROLS = {"non-targeting", "non_targeting", "nontargeting", "control", "ntc"}


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV whether or not it has a header row."""
    raw = pd.read_csv(path, header=None, dtype=str)
    first = [str(v).strip().lower() for v in raw.iloc[0]]
    if any(v in HEADER_WORDS or v.startswith("unnamed") for v in first):
        return pd.read_csv(path, dtype=str)
    raw.columns = [f"col{i}" for i in range(raw.shape[1])]
    return raw


def frac(values, test) -> float:
    s = pd.Series(values).dropna().astype(str)
    return float(s.map(test).mean()) if len(s) else 0.0


def is_num(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def symbol_and_ens_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    ens = next((c for c in df.columns if frac(df[c], lambda v: v.startswith("ENS")) > 0.5), None)
    sym = next((c for c in df.columns if c != ens
                and frac(df[c], is_num) < 0.5
                and frac(df[c], lambda v: v.startswith("ENS")) < 0.5), None)
    if sym is None:
        sys.exit(f"Could not find a symbol column. Columns: {list(df.columns)}\n{df.head()}")
    return sym, ens


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vcc", type=Path, default=VCC)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--n-sample", type=int, default=2000)
    p.add_argument("--hgnc", type=Path, default=None,
                   help="HGNC complete set; fills in Ensembl IDs the release lacks")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # ---- manifest -----------------------------------------------------------
    man = a.vcc / "manifest.json"
    if man.exists():
        try:
            summary["manifest"] = json.loads(man.read_text())
            print("manifest.json:")
            print("  " + json.dumps(summary["manifest"], indent=2)[:1500]
                  .replace("\n", "\n  "))
        except json.JSONDecodeError:
            print("manifest.json: not valid JSON, skipped")

    # ---- gene list (official order) -----------------------------------------
    g = read_table(a.vcc / "gene_names.csv")
    sym_col, ens_col = symbol_and_ens_columns(g)
    genes = g[sym_col].astype(str).str.strip().tolist()
    ens_csv = (g[ens_col].map(strip_version).tolist() if ens_col else [""] * len(genes))
    n_dup = len(genes) - len(set(genes))
    print(f"\ngene_names.csv: {len(genes):,} genes (column {sym_col!r}"
          f"{f', Ensembl in {ens_col!r}' if ens_col else ''}); "
          f"first {genes[:5]}; duplicates: {n_dup}")
    summary.update(n_genes=len(genes), gene_duplicates=n_dup)

    # ---- contexts -----------------------------------------------------------
    import anndata as ad
    ens_from_h5ad: dict[str, str] = {}
    ctx = {}
    rng = np.random.default_rng(0)
    for path in sorted(a.vcc.glob("context_*.h5ad")):
        h = ad.read_h5ad(path, backed="r")
        vs, ve = var_ids(h.var)
        same_order = list(vs) == genes
        same_set = set(vs) == set(genes)
        for s, e in zip(vs, ve):
            if e:
                ens_from_h5ad.setdefault(s, e)

        idx = np.sort(rng.choice(h.n_obs, min(a.n_sample, h.n_obs), replace=False))
        block = h.X[idx]
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        umi = block.sum(1)
        is_int = bool(np.allclose(block, np.round(block)))

        info = {"n_cells": h.n_obs, "n_genes": h.n_vars,
                "same_order_as_gene_names": same_order,
                "same_gene_set": same_set,
                "n_with_ensembl": int(sum(1 for e in ve if e)),
                "obs_columns": list(h.obs.columns),
                "var_columns": list(h.var.columns),
                "median_umi_per_cell": float(np.median(umi)),
                "looks_like_raw_counts": is_int}
        if "ntc_id" in h.obs.columns:
            vc = h.obs["ntc_id"].value_counts()
            info.update(n_ntc_guides=int(len(vc)),
                        cells_per_guide_min=int(vc.min()),
                        cells_per_guide_max=int(vc.max()))
        ctx[path.stem] = info
        h.file.close()

        print(f"\n{path.name}: {h.n_obs:,} cells x {h.n_vars:,} genes")
        print(f"  gene order matches gene_names.csv: {same_order}   "
              f"same set: {same_set}")
        print(f"  Ensembl IDs in var: {info['n_with_ensembl']:,}   "
              f"var columns: {info['var_columns']}")
        print(f"  obs columns: {info['obs_columns']}")
        if "n_ntc_guides" in info:
            print(f"  ntc_id: {info['n_ntc_guides']} guides, "
                  f"{info['cells_per_guide_min']}-{info['cells_per_guide_max']} cells each")
        print(f"  median UMI/cell {info['median_umi_per_cell']:,.0f}   "
              f"raw integer counts: {is_int}")
    summary["contexts"] = ctx

    # ---- reference table ----------------------------------------------------
    ens = [e or ens_from_h5ad.get(s, "") for s, e in zip(genes, ens_csv)]
    how = ["release" if e else "" for e in ens]

    if a.hgnc:
        # Fill Ensembl IDs from HGNC. Challenge symbols may be OLDER than current
        # HGNC (e.g. C1orf112 is now FIRRM), so resolve through previous symbols.
        from genes import load_alias
        hg = pd.read_csv(a.hgnc, sep="\t", dtype=str, low_memory=False,
                         usecols=lambda c: c in ("symbol", "ensembl_gene_id",
                                                 "prev_symbol", "alias_symbol"))
        sym2ens = dict(zip(hg["symbol"], hg["ensembl_gene_id"].map(strip_version)))
        alias = load_alias(a.hgnc)
        for i, s in enumerate(genes):
            if ens[i]:
                continue
            if sym2ens.get(s):
                ens[i], how[i] = sym2ens[s], "hgnc_current"
            else:
                c = alias.canon(s)
                if c != s and sym2ens.get(c):
                    ens[i], how[i] = sym2ens[c], "hgnc_previous"
        # two reference genes resolving to one Ensembl ID would be ambiguous: drop both
        dup = pd.Series(ens)[pd.Series(ens) != ""].duplicated(keep=False)
        for i in dup[dup].index:
            ens[i], how[i] = "", "ambiguous"
        vc = pd.Series(how).replace("", "none").value_counts().to_dict()
        print(f"\nEnsembl IDs filled from HGNC: {vc}")
        prev = [f"{genes[i]}->{alias.canon(genes[i])}" for i in range(len(genes))
                if how[i] == "hgnc_previous"]
        if prev:
            print(f"  resolved via previous symbol, e.g. {', '.join(prev[:8])}")
        miss = [genes[i] for i in range(len(genes)) if not ens[i]]
        if miss:
            print(f"  still without Ensembl ({len(miss):,}), e.g. {miss[:10]}")
        summary["ensembl_fill"] = vc

    n_ens = sum(1 for e in ens if e)
    ref_path = a.out / "challenge_genes.tsv"
    pd.DataFrame({"symbol": genes, "ensembl": ens}).to_csv(
        ref_path, sep="\t", header=False, index=False)
    print(f"\nWrote {ref_path}  ({len(genes):,} genes, {n_ens:,} with Ensembl IDs)")
    if n_ens == 0:
        print("  [note] no Ensembl IDs — rerun with --hgnc to fill them in")
    summary["n_genes_with_ensembl"] = n_ens

    # ---- perturbations ------------------------------------------------------
    pc = read_table(a.vcc / "pert_counts.csv")
    psym, _ = symbol_and_ens_columns(pc)
    num_cols = [c for c in pc.columns if c != psym and frac(pc[c], is_num) > 0.9]
    perts = pc[psym].astype(str).str.strip()
    ctrl = perts.str.lower().isin(CONTROLS) | perts.str.lower().str.startswith("non-target")
    plist = perts[~ctrl].tolist()
    print(f"\npert_counts.csv: columns {list(pc.columns)}; symbol column {psym!r}")
    print(pc.head(5).to_string(index=False))
    print(f"  {len(plist)} perturbations ({int(ctrl.sum())} control rows excluded)")
    if num_cols:
        counts = pd.to_numeric(pc.loc[~ctrl, num_cols[0]], errors="coerce")
        print(f"  {num_cols[0]!r}: min {counts.min():.0f}  max {counts.max():.0f}  "
              f"total {counts.sum():,.0f}")
        summary["pert_count_column"] = num_cols[0]
        summary["pert_counts_total"] = float(counts.sum())

    in_genes = [x for x in plist if x in set(genes)]
    print(f"  perturbation targets that are also measured genes: "
          f"{len(in_genes)}/{len(plist)}")
    missing = [x for x in plist if x not in set(genes)]
    if missing:
        print(f"    not in gene list: {missing[:20]}{' ...' if len(missing) > 20 else ''}")

    perts_path = a.out / "challenge_perts.txt"
    perts_path.write_text("\n".join(plist) + "\n")
    print(f"Wrote {perts_path}")
    summary.update(n_perts=len(plist), perts_in_gene_list=len(in_genes),
                   perts_not_in_gene_list=missing)

    (a.out / "challenge_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {a.out / 'challenge_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())