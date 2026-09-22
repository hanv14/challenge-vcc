"""Phase 1 — LINCS (`processed/phases/phase1_lincs.h5ad`).

One row per (cell_line, target_gene, time_h), CRISPR knockouts only, over the
panel genes (CLAUDE.md §3.3). Four layers:

    ctrl  Level 3 same-plate control wells      log2 normalized expression
    pert  Level 3 treated wells                 log2 normalized expression
    sig   Level 5 mean signature                robust z vs plate controls
    gmt   (#up-sets - #down-sets)/#signatures   in [-1, 1]

LINCS perturbations are CRISPR knockout while the challenge is interference.
Directions and affected genes transfer; magnitudes and the target gene's own
level do not — which is why Phase 1 trains the direction of the response and
Phase 2 re-fits its size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = ("ctrl", "pert", "sig", "gmt")
REQUIRED_OBS = (
    "cell_line",
    "target_gene",
    "time_h",
    "n_pert_wells",
    "n_ctrl_wells",
    "ctrl_source",
    "n_sigs",
    "has_sig",
    "n_gmt_signatures",
    "has_gmt",
    "cc_q75_median",
    "is_challenge_pert",
)
REQUIRED_VAR = ("challenge_idx", "ensembl_id", "lincs_entrez")


@dataclass(frozen=True)
class Phase1Data:
    """LINCS signatures, with the panel genes they are measured on."""

    obs: pd.DataFrame
    var: pd.DataFrame
    layers: dict[str, np.ndarray]

    @property
    def n_rows(self) -> int:
        return len(self.obs)

    @property
    def n_genes(self) -> int:
        return len(self.var)

    @property
    def challenge_idx(self) -> np.ndarray:
        """Challenge index of each column, so the layers can be placed into
        the full gene axis without assuming which genes the panel holds."""
        return self.var["challenge_idx"].to_numpy(np.int64)

    @property
    def cell_lines(self) -> list[str]:
        return sorted(self.obs["cell_line"].astype(str).unique())

    @property
    def targets(self) -> list[str]:
        return sorted(self.obs["target_gene"].astype(str).unique())

    def gmt_mask(self) -> np.ndarray:
        """Rows whose `gmt` layer carries a real value (CLAUDE.md §4.4)."""
        return self.obs["has_gmt"].to_numpy(bool)

    def sig_mask(self) -> np.ndarray:
        return self.obs["has_sig"].to_numpy(bool)


def read_phase1_metadata(phase1_h5ad: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`obs` and `var` only, without reading the layers."""
    import anndata as ad

    backed = ad.read_h5ad(phase1_h5ad, backed="r")
    try:
        return backed.obs.copy(), backed.var.copy()
    finally:
        if backed.file is not None:
            backed.file.close()


def load_phase1(phase1_h5ad: Path) -> Phase1Data:
    """Load every layer.

    This file is one row per (cell line, target, time) rather than per cell,
    so it is small enough to hold whole even on the server.
    """
    import anndata as ad

    adata = ad.read_h5ad(phase1_h5ad)

    missing_obs = [c for c in REQUIRED_OBS if c not in adata.obs.columns]
    if missing_obs:
        raise ValueError(f"{phase1_h5ad}: missing obs column(s) {missing_obs}")
    missing_var = [c for c in REQUIRED_VAR if c not in adata.var.columns]
    if missing_var:
        raise ValueError(f"{phase1_h5ad}: missing var column(s) {missing_var}")
    missing_layers = [name for name in LAYERS if name not in adata.layers]
    if missing_layers:
        raise ValueError(f"{phase1_h5ad}: missing layer(s) {missing_layers}")

    layers = {name: np.asarray(adata.layers[name], dtype=np.float32) for name in LAYERS}
    return Phase1Data(obs=adata.obs.copy(), var=adata.var.copy(), layers=layers)
