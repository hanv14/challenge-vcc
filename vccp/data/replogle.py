"""Phase 2 — Replogle (`processed/phases/phase2/<context>/`).

A thin adapter over `data_prep.phase_data.ReplogleCells`, which is the loader
the data was prepared with and is reused rather than reimplemented
(CLAUDE.md §2). What this module adds is the part the model needs: which
challenge genes a context measures, where they sit on the full gene axis, the
per-target pseudobulk, and the knockdown efficiency from the bulk files.

**Cells are never loaded whole.** K562 genome-wide is ~2 million cells and
65 GB on the server; `ReplogleCells` reads the rows asked for and nothing
else, and `cells_for_target` / `control_cells` are the only ways in.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

GENES_COLUMNS = ("gene_symbol", "challenge_idx", "source_col", "split", "ctrl_mean", "ctrl_std")
CELLS_COLUMNS = ("row", "target_gene", "is_control", "gem_group", "library_size")
META_KEYS = ("source", "context", "n_cells", "n_targets")
PANEL = "panel"

#: `ctrl_std` floors to this before dividing, so a gene with no variance in a
#: context's controls cannot produce an infinite z-score (CLAUDE.md §3.2).
MIN_CTRL_STD = 1e-3


def _import_replogle_cells():
    """Import `ReplogleCells` from `data_prep/`, wherever the repo sits."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data_prep.phase_data import ReplogleCells  # noqa: PLC0415

    return ReplogleCells


@dataclass(frozen=True)
class ReplogleContext:
    """One Replogle screen: its genes, its cells' metadata, its pseudobulk.

    `K562_gwps` and `rpe1` are present in `mini_data`; the server may also
    have `K562_essential`. Contexts are discovered, never listed.
    """

    name: str
    directory: Path
    meta: dict
    genes: pd.DataFrame
    cells: pd.DataFrame

    # ---- gene geometry ---------------------------------------------------
    @property
    def challenge_idx(self) -> np.ndarray:
        """Challenge index of every gene this context measures, ascending."""
        return self.genes["challenge_idx"].to_numpy(np.int64)

    @property
    def panel_challenge_idx(self) -> np.ndarray:
        """Challenge indices of the panel genes, in the order `ReplogleCells`
        returns them (it splits on `split == 'panel'`, keeping file order)."""
        return self.challenge_idx[self._is_panel]

    @property
    def rest_challenge_idx(self) -> np.ndarray:
        return self.challenge_idx[~self._is_panel]

    @cached_property
    def _is_panel(self) -> np.ndarray:
        return (self.genes["split"] == PANEL).to_numpy()

    @property
    def n_panel(self) -> int:
        return int(self._is_panel.sum())

    @property
    def n_rest(self) -> int:
        return int((~self._is_panel).sum())

    def measured_mask(self, n_genes: int) -> np.ndarray:
        """Mask over the full challenge gene axis: which genes this context
        measures. Phase 2's loss is masked with this (CLAUDE.md §4.5)."""
        mask = np.zeros(n_genes, dtype=bool)
        mask[self.challenge_idx] = True
        return mask

    def control_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """(`ctrl_mean`, `ctrl_std`) per measured gene, std floored.

        These are the control-SD units that bridge the sources (§3.2).
        """
        mean = self.genes["ctrl_mean"].to_numpy(np.float32)
        std = np.maximum(self.genes["ctrl_std"].to_numpy(np.float32), MIN_CTRL_STD)
        return mean, std

    # ---- cells -----------------------------------------------------------
    @property
    def targets(self) -> list[str]:
        """Perturbation targets in this screen, controls excluded."""
        labels = self.cells.loc[~self.cells["is_control"], "target_gene"]
        return sorted(labels.astype(str).unique())

    def n_cells_per_target(self) -> pd.Series:
        return self.cells.groupby("target_gene", observed=True).size()

    @cached_property
    def reader(self):
        """The `data_prep` loader, opened on first use and kept open."""
        return _import_replogle_cells()(self.directory)

    def cells_for_target(self, target: str, n: int | None = None, rng=None):
        """`(panel, rest)` log1p(CP10K) arrays for `n` sampled cells."""
        return self.reader.cells(target, n, rng)

    def control_cells(self, n: int | None = None, rng=None):
        return self.reader.controls(n, rng)

    @cached_property
    def source_path(self) -> Path:
        """The single-cell file, by whichever of `meta.json`'s two paths resolves."""
        source = Path(self.meta["source"])
        if not source.exists() and self.meta.get("source_rel"):
            source = (self.directory / self.meta["source_rel"]).resolve()
        if not source.exists():
            raise FileNotFoundError(
                f"{self.directory / 'meta.json'} points at {source}, which does not exist"
            )
        return source

    def raw_counts(self, rows) -> np.ndarray:
        """Raw integer counts for `rows`, over this screen's measured genes.

        `ReplogleCells` returns log1p(CP10K), which is what the model reads;
        the cell generator works in **count space** (CLAUDE.md §4.7) and the
        official metrics take raw counts on both sides (§6), so the rehearsal
        needs the untransformed values too. Read from the same file and the
        same columns, one block of rows at a time — the K562 genome-wide file
        is ~2 million cells on the server.
        """
        import anndata as ad

        rows = np.atleast_1d(np.asarray(rows))
        order = np.argsort(rows)
        ascending = rows[order]

        backed = ad.read_h5ad(self.source_path, backed="r")
        try:
            block = backed.X[ascending]
        finally:
            if backed.file is not None:
                backed.file.close()

        dense = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        dense = dense[:, self.genes["source_col"].to_numpy()]
        return np.rint(dense[np.argsort(order)]).astype(np.int64)


def load_replogle_context(directory: Path) -> ReplogleContext:
    meta_path = directory / "meta.json"
    genes_path = directory / "genes.csv"
    cells_path = directory / "cells.parquet"
    for path in (meta_path, genes_path, cells_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing")

    meta = json.loads(meta_path.read_text())
    missing_meta = [key for key in META_KEYS if key not in meta]
    if missing_meta:
        raise ValueError(f"{meta_path}: missing key(s) {missing_meta}")

    genes = pd.read_csv(genes_path)
    missing = [c for c in GENES_COLUMNS if c not in genes.columns]
    if missing:
        raise ValueError(f"{genes_path}: missing column(s) {missing}")

    cells = pd.read_parquet(cells_path)
    missing = [c for c in CELLS_COLUMNS if c not in cells.columns]
    if missing:
        raise ValueError(f"{cells_path}: missing column(s) {missing}")

    return ReplogleContext(
        name=str(meta["context"]),
        directory=directory,
        meta=meta,
        genes=genes,
        cells=cells,
    )


def load_pseudobulk(directory: Path, control_label: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """`pseudobulk.h5ad`: one row per target plus the control.

    Returns (obs, X, var). X is mean log1p(CP10K) over the target's cells.
    """
    import anndata as ad

    path = directory / "pseudobulk.h5ad"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing")
    adata = ad.read_h5ad(path)

    if control_label not in adata.obs_names:
        raise ValueError(
            f"{path}: no {control_label!r} row — the pseudobulk must carry the control"
        )
    X = adata.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    return adata.obs.copy(), X, adata.var.copy()


def load_bulk_quality(bulk_h5ad: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Per-target quality signals from a `*_raw_bulk.h5ad`.

    A gene can have several rows — one per promoter — so they are reduced by
    median. Unlike `load_fold_expr`, a `fold_expr` of exactly zero is kept
    here: as a *quality* signal it means the knockdown was complete, which is
    the most reliable case, not a missing one.
    """
    import anndata as ad

    backed = ad.read_h5ad(bulk_h5ad, backed="r")
    try:
        obs = backed.obs.copy()
    finally:
        if backed.file is not None:
            backed.file.close()

    if "target_gene" not in obs.columns:
        raise ValueError(f"{bulk_h5ad}: no 'target_gene' column in obs")

    available = [c for c in columns if c in obs.columns]
    if not available:
        return pd.DataFrame(index=pd.Index([], name="target_gene"))

    frame = pd.DataFrame({"target_gene": obs["target_gene"].astype(str)})
    for column in available:
        values = pd.to_numeric(obs[column], errors="coerce")
        frame[column] = np.where(np.isfinite(values), values, np.nan)
    return frame.groupby("target_gene").median(numeric_only=True)


def load_fold_expr(bulk_h5ad: Path) -> pd.Series:
    """Knockdown efficiency per target from a `*_raw_bulk.h5ad`.

    `obs['fold_expr']` is the target's expression in perturbed cells over
    control (CLAUDE.md §3.3). A gene can have several rows — one per promoter
    — so they are reduced by median. Non-finite and non-positive values are
    dropped rather than propagated.
    """
    import anndata as ad

    backed = ad.read_h5ad(bulk_h5ad, backed="r")
    try:
        obs = backed.obs.copy()
    finally:
        if backed.file is not None:
            backed.file.close()

    if "fold_expr" not in obs.columns:
        raise ValueError(f"{bulk_h5ad}: no 'fold_expr' column in obs")
    if "target_gene" not in obs.columns:
        raise ValueError(f"{bulk_h5ad}: no 'target_gene' column in obs")

    values = pd.to_numeric(obs["fold_expr"], errors="coerce")
    usable = np.isfinite(values) & (values > 0)
    frame = pd.DataFrame(
        {"target_gene": obs.loc[usable, "target_gene"].astype(str), "fold_expr": values[usable]}
    )
    return frame.groupby("target_gene")["fold_expr"].median()
