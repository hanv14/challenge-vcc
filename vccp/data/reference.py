"""The gene and perturbation reference.

One gene coordinate system everywhere: challenge genes in challenge order
(CLAUDE.md §3.1). `challenge_idx = i` means the i-th gene of that list in
every file the pipeline touches, and genes are named by challenge symbols.

`gene_names.csv` (the challenge release) is the authority on the submission's
gene axis; `ref/challenge_genes.tsv` carries the same list with Ensembl ids.
`check-data` verifies the two agree — this module just loads them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HGNC_COLUMNS = ("symbol", "gene_group", "ensembl_gene_id", "prev_symbol", "alias_symbol")


@dataclass(frozen=True)
class GeneReference:
    """Challenge genes in challenge order."""

    symbols: tuple[str, ...]
    ensembl: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def index(self) -> dict[str, int]:
        return {symbol: i for i, symbol in enumerate(self.symbols)}

    def positions(self, symbols) -> np.ndarray:
        """Challenge indices of `symbols`, raising on anything unknown."""
        lookup = self.index
        unknown = [s for s in symbols if s not in lookup]
        if unknown:
            raise KeyError(
                f"{len(unknown)} symbol(s) are not challenge genes, e.g. {unknown[:5]}"
            )
        return np.asarray([lookup[s] for s in symbols], dtype=np.int64)


def load_gene_reference(challenge_genes: Path) -> GeneReference:
    """`ref/challenge_genes.tsv`: symbol<TAB>ensembl, no header, official order."""
    frame = pd.read_csv(
        challenge_genes, sep="\t", header=None, names=["symbol", "ensembl"], dtype=str
    )
    if frame.empty:
        raise ValueError(f"{challenge_genes}: no genes")
    duplicated = frame["symbol"][frame["symbol"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{challenge_genes}: duplicate symbols, e.g. {duplicated[:5]}")
    return GeneReference(
        symbols=tuple(frame["symbol"]),
        ensembl=tuple(frame["ensembl"].fillna("")),
    )


def load_gene_names(gene_names_csv: Path) -> list[str]:
    """`gene_names.csv` (column `gene_name`) — the submission's gene axis."""
    frame = pd.read_csv(gene_names_csv, dtype=str)
    if "gene_name" not in frame.columns:
        raise ValueError(
            f"{gene_names_csv}: expected a 'gene_name' column, found {list(frame.columns)}"
        )
    return frame["gene_name"].tolist()


def load_challenge_perts(challenge_perts: Path) -> list[str]:
    """`ref/challenge_perts.txt` — one target gene symbol per line."""
    lines = challenge_perts.read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


@dataclass(frozen=True)
class PertCounts:
    """`pert_counts.csv` — the authority on which perturbations to submit.

    `cells_per_pert` is the per-target cell count if the file carries one
    (the column name varies, so any single numeric column is taken as it),
    otherwise None and the manifest's value applies.
    """

    targets: tuple[str, ...]
    cells_per_pert: dict[str, int] | None
    count_column: str | None

    def __len__(self) -> int:
        return len(self.targets)


def load_pert_counts(pert_counts_csv: Path, pert_col: str = "target_gene") -> PertCounts:
    frame = pd.read_csv(pert_counts_csv)
    if pert_col not in frame.columns:
        raise ValueError(
            f"{pert_counts_csv}: expected a {pert_col!r} column, found {list(frame.columns)}"
        )
    targets = frame[pert_col].astype(str).tolist()
    duplicated = sorted({t for t in targets if targets.count(t) > 1})
    if duplicated:
        raise ValueError(f"{pert_counts_csv}: duplicate targets, e.g. {duplicated[:5]}")

    numeric = [
        column
        for column in frame.columns
        if column != pert_col and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(numeric) == 1:
        column = numeric[0]
        counts = {t: int(n) for t, n in zip(targets, frame[column])}
        return PertCounts(tuple(targets), counts, column)
    if len(numeric) > 1:
        raise ValueError(
            f"{pert_counts_csv}: more than one numeric column ({numeric}); "
            "cannot tell which one is the per-target cell count"
        )
    return PertCounts(tuple(targets), None, None)


def load_hgnc(hgnc_path: Path) -> pd.DataFrame:
    """The HGNC table, under either of its two file names.

    Only the columns the pipeline uses are required to be present; the file
    has many more and they are kept.
    """
    frame = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    missing = [column for column in HGNC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{hgnc_path}: missing column(s) {missing}")
    return frame
