"""Coverage tables (`processed/pert_coverage.csv`, `gene_coverage.csv`).

Which sources contain which perturbation, and which measure which gene. The
column names are source names and differ between mini_data and the server
(the server has an extra Replogle screen), so they are read rather than
listed — only the key column is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Coverage:
    """A coverage table plus the names of its source columns."""

    frame: pd.DataFrame
    key: str

    @property
    def sources(self) -> list[str]:
        return [c for c in self.frame.columns if c != self.key]

    @property
    def boolean_sources(self) -> list[str]:
        """Source columns that are true/false rather than counts."""
        return [c for c in self.sources if self.frame[c].dtype == bool]

    def counts_per_source(self) -> dict[str, int]:
        return {c: int(self.frame[c].sum()) for c in self.boolean_sources}

    def covered_by_any(self) -> pd.Series:
        columns = self.boolean_sources
        if not columns:
            return pd.Series(False, index=self.frame.index)
        return self.frame[columns].any(axis=1)


def load_pert_coverage(path: Path, key: str = "target_gene") -> Coverage:
    frame = pd.read_csv(path)
    if key not in frame.columns:
        raise ValueError(f"{path}: expected a {key!r} column, found {list(frame.columns)}")
    return Coverage(frame=frame, key=key)


def load_gene_coverage(path: Path, key: str = "gene_symbol") -> Coverage:
    frame = pd.read_csv(path)
    if key not in frame.columns:
        raise ValueError(f"{path}: expected a {key!r} column, found {list(frame.columns)}")
    return Coverage(frame=frame, key=key)
