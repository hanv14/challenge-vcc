"""The panel / rest gene split (`processed/phases/gene_split.csv`).

`panel` = LINCS landmark genes that are challenge genes; `rest` = every other
challenge gene (CLAUDE.md §3.1). The split is the model's input/output
division: the panel is what a perturbation model predicts directly, the rest
is what the shared mapping predicts from it.

Sizes are never assumed — `n_panel` and `n_rest` are whatever the file says.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "gene_symbol",
    "ensembl_id",
    "challenge_idx",
    "lincs_entrez",
    "lincs_feature_space",
    "split",
)
PANEL = "panel"
REST = "rest"


@dataclass(frozen=True)
class GeneSplit:
    """Panel/rest membership, in challenge order."""

    frame: pd.DataFrame
    #: challenge indices of the panel genes, ascending
    panel_idx: np.ndarray
    #: challenge indices of the rest genes, ascending
    rest_idx: np.ndarray

    @property
    def n_genes(self) -> int:
        return len(self.frame)

    @property
    def n_panel(self) -> int:
        return int(self.panel_idx.size)

    @property
    def n_rest(self) -> int:
        return int(self.rest_idx.size)

    @property
    def symbols(self) -> list[str]:
        return self.frame["gene_symbol"].tolist()

    @property
    def panel_symbols(self) -> list[str]:
        return self.frame["gene_symbol"].to_numpy()[self.panel_idx].tolist()

    @property
    def rest_symbols(self) -> list[str]:
        return self.frame["gene_symbol"].to_numpy()[self.rest_idx].tolist()

    def is_panel(self) -> np.ndarray:
        """Boolean mask over all challenge genes, in challenge order."""
        mask = np.zeros(self.n_genes, dtype=bool)
        mask[self.panel_idx] = True
        return mask


def load_gene_split(gene_split_csv: Path) -> GeneSplit:
    frame = pd.read_csv(gene_split_csv)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{gene_split_csv}: missing column(s) {missing}")

    unexpected = sorted(set(frame["split"].unique()) - {PANEL, REST})
    if unexpected:
        raise ValueError(
            f"{gene_split_csv}: 'split' must be {PANEL!r} or {REST!r}, found {unexpected}"
        )

    idx = frame["challenge_idx"].to_numpy()
    if not np.array_equal(idx, np.arange(len(frame))):
        raise ValueError(
            f"{gene_split_csv}: 'challenge_idx' must be 0..{len(frame) - 1} in order "
            "(the file is the gene coordinate system; it cannot be reordered)"
        )

    split = frame["split"].to_numpy()
    return GeneSplit(
        frame=frame,
        panel_idx=np.flatnonzero(split == PANEL).astype(np.int64),
        rest_idx=np.flatnonzero(split == REST).astype(np.int64),
    )
