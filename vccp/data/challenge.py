"""Phase 3 — the challenge data (`processed/phases/phase3/`, `vcc_root/`).

Control cells only: the challenge never shows a perturbed cell, which is what
makes Phase 3 an adaptation on controls (CLAUDE.md §4.7). `controls_<ctx>.h5ad`
carries both units — `X` is log1p(CP10K) and `layers['counts']` the raw
integers the submission is built from.

Cells are read in blocks. Nothing here materializes a
cells x all-genes dense matrix for more than one block at a time, so the same
code path works on the server's far larger contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

CONTROLS_VAR_COLUMNS = ("challenge_idx", "split", "ctrl_mean", "ctrl_std")
CONTROLS_OBS_COLUMNS = ("library_size", "ntc_id")
TARGETS_COLUMNS = ("context", "target_gene", "n_cells")
GENES_COLUMNS = ("gene_symbol", "challenge_idx", "split")

#: See `replogle.MIN_CTRL_STD`.
MIN_CTRL_STD = 1e-3


@dataclass(frozen=True)
class ChallengeControls:
    """One context's control cells, read in blocks from disk."""

    context: str
    path: Path
    obs: pd.DataFrame
    var: pd.DataFrame

    @property
    def n_cells(self) -> int:
        return len(self.obs)

    @property
    def n_genes(self) -> int:
        return len(self.var)

    @property
    def library_size(self) -> np.ndarray:
        return self.obs["library_size"].to_numpy(np.float64)

    def control_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """(`ctrl_mean`, `ctrl_std`) over all challenge genes, std floored."""
        mean = self.var["ctrl_mean"].to_numpy(np.float32)
        std = np.maximum(self.var["ctrl_std"].to_numpy(np.float32), MIN_CTRL_STD)
        return mean, std

    # ---- block reads -----------------------------------------------------
    def _read(self, layer: str | None, rows: np.ndarray | None) -> np.ndarray:
        import anndata as ad

        backed = ad.read_h5ad(self.path, backed="r")
        try:
            source = backed.X if layer is None else backed.layers[layer]
            block = source if rows is None else source[np.sort(np.asarray(rows))]
            dense = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        finally:
            if backed.file is not None:
                backed.file.close()
        return np.asarray(dense, dtype=np.float32)

    def counts(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Raw integer counts for `rows` (all cells when None).

        Rows come back in ascending order, which is what the sampler wants —
        it shuffles the selection itself.
        """
        return self._read("counts", rows)

    def lognorm(self, rows: np.ndarray | None = None) -> np.ndarray:
        """log1p(CP10K) values for `rows`."""
        return self._read(None, rows)

    def iter_lognorm_blocks(self, block_size: int) -> Iterator[np.ndarray]:
        """Stream the control cells in blocks, for the co-expression priors."""
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        for start in range(0, self.n_cells, block_size):
            rows = np.arange(start, min(start + block_size, self.n_cells))
            yield self.lognorm(rows)


def load_challenge_controls(controls_h5ad: Path, context: str) -> ChallengeControls:
    """Open a control file and read its metadata, not its matrix."""
    import anndata as ad

    backed = ad.read_h5ad(controls_h5ad, backed="r")
    try:
        obs, var = backed.obs.copy(), backed.var.copy()
        has_counts = "counts" in backed.layers
    finally:
        if backed.file is not None:
            backed.file.close()

    missing = [c for c in CONTROLS_VAR_COLUMNS if c not in var.columns]
    if missing:
        raise ValueError(f"{controls_h5ad}: missing var column(s) {missing}")
    missing = [c for c in CONTROLS_OBS_COLUMNS if c not in obs.columns]
    if missing:
        raise ValueError(f"{controls_h5ad}: missing obs column(s) {missing}")
    if not has_counts:
        raise ValueError(f"{controls_h5ad}: no layers['counts'] — raw counts are required")

    return ChallengeControls(context=context, path=controls_h5ad, obs=obs, var=var)


def load_phase3_genes(genes_csv: Path) -> pd.DataFrame:
    """`phase3/genes.csv` — all challenge genes with their split."""
    frame = pd.read_csv(genes_csv)
    missing = [c for c in GENES_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{genes_csv}: missing column(s) {missing}")
    return frame


def load_targets(targets_csv: Path) -> pd.DataFrame:
    """`phase3/targets.csv` — one row per (context, target) to predict."""
    frame = pd.read_csv(targets_csv)
    missing = [c for c in TARGETS_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{targets_csv}: missing column(s) {missing}")
    return frame


def read_release_context(context_h5ad: Path) -> tuple[pd.DataFrame, list[str]]:
    """`vcc_root/context_<X>.h5ad`: control cells, raw counts.

    Returns (obs, var_names) without reading the matrix. The context label the
    submission must reuse is `obs[context_col]` of this file (CLAUDE.md §8).
    """
    import anndata as ad

    backed = ad.read_h5ad(context_h5ad, backed="r")
    try:
        return backed.obs.copy(), list(backed.var_names)
    finally:
        if backed.file is not None:
            backed.file.close()
