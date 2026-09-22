#!/usr/bin/env python3
"""
Gene-symbol harmonization against one reference vocabulary: the challenge's
gene list. Every processed dataset (Replogle, LINCS, GMT) is mapped into it,
so all outputs speak the challenge's symbols.

Why not just join on symbols? Symbols drift. HGNC renames genes (e.g. SKIV2L
became SKIC2 in 2021); Replogle 2022, LINCS 2020 and the 2026 challenge each
use a different annotation year. A plain symbol join silently drops renamed
genes — and if a renamed gene is a challenge perturbation, you'd wrongly
conclude it has no training data.

Matching order (first hit wins, each reference gene is claimed at most once):
    1. exact symbol
    2. Ensembl gene ID (stable across renames; version suffix ignored)
    3. previous-symbol / alias table (optional, e.g. HGNC complete set)

Python
------
    from genes import load_reference, harmonize_genes, harmonize_labels
    ref = load_reference("validation_controls.h5ad", alias="hgnc_complete_set.txt")
    table, info = harmonize_genes(symbols, ensembl_ids, ref)
    new_labels, report = harmonize_labels(pert_symbols, pert_ensembl, ref)

CLI — harmonize any existing .h5ad (small ones; loads into memory)
------------------------------------------------------------------
    python genes.py apply --h5ad lincs_xpr_consensus.h5ad \\
        --reference validation_controls.h5ad --alias hgnc_complete_set.txt \\
        --pert-col gene --extra-perts challenge_perts.txt \\
        --out lincs_xpr_consensus.harmonized.h5ad

    add --to-reference-space to reorder columns into the reference gene order
    (unmeasured genes become zero columns, flagged by var['measured']).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL_COLS = ["gene_name", "gene_symbol", "gene_symbols", "symbol", "feature_name"]
ENS_COLS = ["gene_id", "gene_ids", "ensembl_id", "ensembl_gene_id", "ensembl"]


# --------------------------------------------------------------------------- #
# identifiers
# --------------------------------------------------------------------------- #
def strip_version(e) -> str:
    e = "" if e is None else str(e).strip()
    if e.lower() in ("", "nan", "none"):
        return ""
    return e.split(".")[0] if e.startswith("ENS") else ""


def _ens_like(values) -> float:
    v = pd.Series(values).astype(str)
    return float(v.str.startswith("ENS").mean()) if len(v) else 0.0


def var_ids(var: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(symbols, ensembl_ids) from any var table.

    Ensembl is taken from whichever of index / known columns actually holds
    ENS* values — so a LINCS 'gene_id' column (which is Entrez) is ignored.
    """
    idx = var.index.astype(str)
    sc = next((c for c in SYMBOL_COLS if c in var.columns), None)
    sym = var[sc].astype(str).values if sc else idx.values

    ens = np.array([""] * len(var), dtype=object)
    if _ens_like(idx) > 0.5:
        ens = idx.values
    else:
        for c in ENS_COLS:
            if c in var.columns and _ens_like(var[c]) > 0.5:
                ens = var[c].astype(str).values
                break
    ens = np.array([strip_version(e) for e in ens], dtype=object)
    return np.asarray(sym, dtype=object), ens


# --------------------------------------------------------------------------- #
# alias tables
# --------------------------------------------------------------------------- #
class AliasMap(dict):
    """old-symbol -> current-symbol, plus the set of current symbols."""

    def __init__(self, *args, current: set[str] | None = None, **kw):
        super().__init__(*args, **kw)
        self.current: set[str] = current if current is not None else set(self.values())

    def canon(self, s: str) -> str:
        """Current HGNC symbol for s (s itself if already current or unknown)."""
        if s in self.current:
            return s
        return self.get(s, s)


def load_alias(path: Path | str | None) -> AliasMap:
    """Load old-symbol -> current-symbol.

    Accepts the HGNC complete set (columns symbol, prev_symbol, alias_symbol;
    pipe-separated lists) or a plain two-column TSV (old <tab> new).
    Ambiguous entries (one old symbol -> several new) are dropped.
    Previous symbols take priority over aliases.
    """
    if path is None:
        return AliasMap()
    path = Path(path)
    head = pd.read_csv(path, sep="\t", nrows=0).columns
    maps: list[dict[str, str]] = []
    current: set[str] = set()

    def build(pairs) -> dict[str, str]:
        m, bad = {}, set()
        for old, new in pairs:
            old, new = str(old).strip(), str(new).strip()
            if not old or not new or old == new or old.lower() == "nan":
                continue
            if old in m and m[old] != new:
                bad.add(old)
            m[old] = new
        for b in bad:
            m.pop(b, None)
        return m

    if "symbol" in head and ({"prev_symbol", "alias_symbol"} & set(head)):
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False,
                         usecols=[c for c in ("symbol", "prev_symbol", "alias_symbol")
                                  if c in head])
        current = set(df["symbol"].dropna())
        for col in ("alias_symbol", "prev_symbol"):          # prev applied last = wins
            if col in df.columns:
                pairs = [(o, new) for new, olds in zip(df["symbol"], df[col])
                         if isinstance(olds, str)
                         for o in olds.strip('"').split("|")]
                maps.append(build(pairs))
    else:
        df = pd.read_csv(path, sep="\t", dtype=str, header=None, usecols=[0, 1])
        maps.append(build(zip(df[0], df[1])))

    merged: dict[str, str] = {}
    for m in maps:
        merged.update(m)
    # an "alias" that is some other gene's current symbol is not safe to remap
    for s in current:
        merged.pop(s, None)
    return AliasMap(merged, current=current or None)


# --------------------------------------------------------------------------- #
# reference
# --------------------------------------------------------------------------- #
@dataclass
class Reference:
    symbols: list[str]
    ensembl: list[str]
    alias: AliasMap = field(default_factory=AliasMap)

    def __post_init__(self):
        if not isinstance(self.alias, AliasMap):
            self.alias = AliasMap(self.alias)
        self.sym2idx: dict[str, int] = {}
        for i, s in enumerate(self.symbols):
            self.sym2idx.setdefault(s, i)
        self.ens2idx: dict[str, int] = {}
        for i, e in enumerate(self.ensembl):
            if e:
                self.ens2idx.setdefault(e, i)
        # current-HGNC form of each reference symbol. Matching on this works
        # whether the reference or the source uses the older annotation.
        self.canon2idx: dict[str, int] = {}
        if self.alias:
            for i, s in enumerate(self.symbols):
                self.canon2idx.setdefault(self.alias.canon(s), i)

    def canon_index(self, s: str) -> int | None:
        return self.canon2idx.get(self.alias.canon(s)) if self.alias else None

    def __len__(self) -> int:
        return len(self.symbols)


def load_reference(path: Path | str | None,
                   alias: Path | str | None = None,
                   verbose: bool = True) -> Reference | None:
    """Reference gene vocabulary from a challenge .h5ad (uses var) or a .txt
    (one symbol per line, or 'symbol<TAB>ensembl')."""
    if path is None:
        return None
    path = Path(path)
    if path.suffix == ".h5ad":
        import anndata as ad
        a = ad.read_h5ad(path, backed="r")
        sym, ens = var_ids(a.var)
        a.file.close()
    else:
        rows = [l.rstrip("\n").split("\t") for l in open(path) if l.strip()]
        sym = np.array([r[0].strip() for r in rows], dtype=object)
        ens = np.array([strip_version(r[1]) if len(r) > 1 else "" for r in rows],
                       dtype=object)
    ref = Reference(list(sym), list(ens), load_alias(alias))
    if verbose:
        n_ens = sum(1 for e in ref.ensembl if e)
        print(f"  reference: {len(ref):,} genes from {path.name}; "
              f"{n_ens:,} with Ensembl IDs; {len(ref.alias):,} alias entries")
        if n_ens == 0:
            print("  [note] reference has no Ensembl IDs — renamed genes can only "
                  "be rescued via --alias")
    return ref


# --------------------------------------------------------------------------- #
# harmonize measured genes (var)
# --------------------------------------------------------------------------- #
def harmonize_genes(symbols, ensembl, ref: Reference | None
                    ) -> tuple[pd.DataFrame, dict]:
    """Map a dataset's genes into the reference.

    Returns a table aligned with the input order:
        gene_symbol       harmonized (reference) symbol, or original if unmatched
        gene_symbol_orig  as in the source
        gene_id           Ensembl, versionless ('' if unknown)
        target_idx        column in reference space, -1 if unmatched
        match             'symbol' | 'ensembl' | 'alias' | 'none'
    """
    # object dtype, NOT astype(str): numpy's fixed-width strings would silently
    # truncate a replacement symbol longer than the longest source symbol
    sym = np.array([str(s) for s in symbols], dtype=object)
    n = len(sym)
    ens = (np.array([strip_version(e) for e in ensembl], dtype=object)
           if ensembl is not None else np.array([""] * n, dtype=object))
    tidx = np.full(n, -1, dtype=np.int32)
    how = np.array(["none"] * n, dtype=object)

    if ref is None:
        return pd.DataFrame({"gene_symbol": sym, "gene_symbol_orig": sym,
                             "gene_id": ens, "target_idx": tidx,
                             "match": how}), {}

    taken: set[int] = set()

    def claim(j: int, i: int | None, label: str) -> None:
        if i is not None and i not in taken:
            taken.add(i)
            tidx[j], how[j] = i, label

    for j, s in enumerate(sym):
        claim(j, ref.sym2idx.get(s), "symbol")
    if ref.ens2idx:
        for j in np.flatnonzero(tidx < 0):
            claim(j, ref.ens2idx.get(ens[j]), "ensembl")
    if ref.alias:
        # compare current-HGNC forms of both sides: catches old->new AND new->old
        for j in np.flatnonzero(tidx < 0):
            claim(j, ref.canon_index(sym[j]), "alias")

    harm = sym.copy()
    m = tidx >= 0
    harm[m] = np.asarray(ref.symbols, dtype=object)[tidx[m]]
    renamed = m & (harm != sym)

    table = pd.DataFrame({"gene_symbol": harm, "gene_symbol_orig": sym,
                          "gene_id": ens, "target_idx": tidx, "match": how})
    info = {
        "n_reference_genes": len(ref),
        "n_source_genes": n,
        "n_matched": int(m.sum()),
        "frac_reference_covered": round(float(m.sum()) / max(len(ref), 1), 4),
        "matched_by_symbol": int((how == "symbol").sum()),
        "matched_by_ensembl": int((how == "ensembl").sum()),
        "matched_by_alias": int((how == "alias").sum()),
        "n_renamed": int(renamed.sum()),
        "renamed_examples": [f"{o}->{h}" for o, h in
                             zip(sym[renamed][:12], harm[renamed][:12])],
    }
    return table, info


# --------------------------------------------------------------------------- #
# harmonize perturbation labels (obs)
# --------------------------------------------------------------------------- #
def harmonize_labels(labels, label_ensembl, ref: Reference | None,
                     extra_symbols=None, keep=frozenset()
                     ) -> tuple[np.ndarray, pd.DataFrame]:
    """Map perturbation target symbols into the reference vocabulary.

    extra_symbols: additional valid symbols (e.g. the 300 challenge perts,
                   which need not all be in the measured-gene list).
    keep:          labels to leave untouched (e.g. the control label).
    """
    labels = pd.Series(np.asarray(labels, dtype=object)).astype(str)
    if ref is None:
        return labels.values, pd.DataFrame(columns=["orig", "harmonized", "match"])

    ens_of: dict[str, str] = {}
    if label_ensembl is not None:
        for s, e in zip(labels, label_ensembl):
            e = strip_version(e)
            if e:
                ens_of.setdefault(s, e)

    vocab = set(ref.symbols) | set(extra_symbols or [])
    canon_vocab: dict[str, str] = {}
    if ref.alias:
        for v in list(ref.symbols) + list(extra_symbols or []):
            canon_vocab.setdefault(ref.alias.canon(v), v)
    ens2sym = {e: ref.symbols[i] for e, i in ref.ens2idx.items()}

    decided: dict[str, tuple[str, str]] = {}
    for s in labels.unique():
        if s in keep:
            decided[s] = (s, "kept")
        elif s in vocab:
            decided[s] = (s, "symbol")
        elif ens_of.get(s) in ens2sym:
            decided[s] = (ens2sym[ens_of[s]], "ensembl")
        elif ref.alias and ref.alias.canon(s) in canon_vocab:
            decided[s] = (canon_vocab[ref.alias.canon(s)], "alias")
        else:
            decided[s] = (s, "none")

    new = labels.map(lambda s: decided[s][0]).values
    report = pd.DataFrame([(s, v[0], v[1]) for s, v in decided.items()],
                          columns=["orig", "harmonized", "match"])
    # two different source labels collapsing onto one reference symbol
    changed = report[report["orig"] != report["harmonized"]]
    clash = report["harmonized"].duplicated(keep=False) & report["harmonized"].isin(
        changed["harmonized"])
    report["collides"] = clash.values
    return new, report


def summarize_labels(report: pd.DataFrame) -> str:
    if report.empty:
        return "  perturbation labels: no reference given, unchanged"
    c = report["match"].value_counts().to_dict()
    ren = report[report["orig"] != report["harmonized"]]
    ex = ", ".join(f"{o}->{h}" for o, h in zip(ren["orig"][:8], ren["harmonized"][:8]))
    s = (f"  perturbation labels: {c.get('symbol', 0):,} exact, "
         f"{c.get('ensembl', 0)} via Ensembl, {c.get('alias', 0)} via alias, "
         f"{c.get('none', 0):,} not in reference")
    if len(ren):
        s += f"\n    renamed: {ex}{' ...' if len(ren) > 8 else ''}"
    if report.get("collides", pd.Series(dtype=bool)).any():
        s += (f"\n    [warn] {int(report['collides'].sum())} labels collide onto "
              f"the same symbol after renaming — check the report CSV")
    return s


# --------------------------------------------------------------------------- #
# CLI: apply to an existing h5ad
# --------------------------------------------------------------------------- #
def apply_h5ad(src: Path, dst: Path, ref: Reference, pert_col: str | None,
               pert_ens_col: str | None, extra: list[str] | None,
               to_reference_space: bool, control_labels: set[str]) -> None:
    import anndata as ad
    a = ad.read_h5ad(src)
    sym, ens = var_ids(a.var)
    table, info = harmonize_genes(sym, ens, ref)
    print(f"  genes: {info['n_matched']:,}/{info['n_reference_genes']:,} reference "
          f"genes covered ({info['frac_reference_covered']:.1%}); "
          f"{info['matched_by_ensembl']} via Ensembl, "
          f"{info['matched_by_alias']} via alias")
    if info["renamed_examples"]:
        print(f"    renamed e.g. {', '.join(info['renamed_examples'][:8])}")

    if to_reference_space:
        import scipy.sparse as sp
        m = table["target_idx"].to_numpy() >= 0
        src_cols = np.flatnonzero(m)
        dst_cols = table["target_idx"].to_numpy()[m]
        X = a.X
        if sp.issparse(X):
            X = X.tocsc()[:, src_cols]
            P = sp.csr_matrix((np.ones(len(src_cols)), (np.arange(len(src_cols)),
                               dst_cols)), shape=(len(src_cols), len(ref)))
            Xn = (X @ P).tocsr().astype(a.X.dtype)
        else:
            Xn = np.zeros((a.n_obs, len(ref)), dtype=np.asarray(X).dtype)
            Xn[:, dst_cols] = np.asarray(X)[:, src_cols]
        var = pd.DataFrame(index=pd.Index(ref.symbols, name="gene_symbol"))
        var["gene_id"] = ref.ensembl
        var["measured"] = False
        var.iloc[dst_cols, var.columns.get_loc("measured")] = True
        orig = pd.Series("", index=range(len(ref)), dtype=object)
        orig.iloc[dst_cols] = table["gene_symbol_orig"].to_numpy()[m]
        var["gene_symbol_orig"] = orig.values
        b = ad.AnnData(X=Xn, obs=a.obs.copy(), var=var, uns=dict(a.uns))
    else:
        b = a
        for c in ("gene_symbol", "gene_symbol_orig", "gene_id", "target_idx", "match"):
            b.var[c] = table[c].values
        b.var_names = pd.Index(table["gene_symbol"].astype(str).values)
        b.var_names_make_unique()

    if pert_col:
        if pert_col not in b.obs.columns:
            sys.exit(f"--pert-col {pert_col!r} not in obs: {list(b.obs.columns)}")
        lab = b.obs[pert_col].astype(str)
        keep = {l for l in lab.unique() if l.lower() in control_labels}
        e = b.obs[pert_ens_col] if pert_ens_col else None
        new, rep = harmonize_labels(lab.values, e, ref, extra, keep)
        b.obs[f"{pert_col}_orig"] = lab.values
        b.obs[pert_col] = new
        print(summarize_labels(rep))
        rep.to_csv(dst.with_suffix(".perturbation_harmonization.csv"), index=False)

    b.uns["gene_harmonization"] = {k: v for k, v in info.items()
                                   if k != "renamed_examples"}
    b.write_h5ad(dst, compression="gzip")
    table.to_csv(dst.with_suffix(".gene_harmonization.csv"), index=False)
    print(f"Wrote {dst}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ap = sub.add_parser("apply", help="harmonize an existing .h5ad")
    ap.add_argument("--h5ad", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True,
                    help="challenge .h5ad (uses var) or .txt of symbols")
    ap.add_argument("--alias", type=Path, default=None,
                    help="HGNC complete set, or two-column old<TAB>new TSV")
    ap.add_argument("--pert-col", default=None, help="obs column of target symbols")
    ap.add_argument("--pert-ens-col", default=None,
                    help="obs column of target Ensembl IDs (improves rescue)")
    ap.add_argument("--extra-perts", type=Path, default=None,
                    help="txt of extra valid perturbation symbols (challenge perts)")
    ap.add_argument("--to-reference-space", action="store_true",
                    help="reorder columns into reference gene order")
    ap.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    ref = load_reference(a.reference, a.alias)
    extra = ([l.strip() for l in open(a.extra_perts) if l.strip()]
             if a.extra_perts else None)
    apply_h5ad(a.h5ad, a.out, ref, a.pert_col, a.pert_ens_col, extra,
               a.to_reference_space,
               {"non-targeting", "non_targeting", "control", "ntc", "ctl"})
    return 0


if __name__ == "__main__":
    sys.exit(main())