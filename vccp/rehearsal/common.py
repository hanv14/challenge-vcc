"""Shared machinery for the three rehearsal variants (CLAUDE.md §4.6).

The rehearsal runs Phase 3's procedure on Replogle data whose perturbed cells
are hidden, then scores against them — so it needs the same conversions and
the same generator the submission will use, and it needs them identical or
the rehearsal measures something the submission will not do.

Two rules from the M0 review (DECISIONS.md D7):

* **the arms share their control resample and their generator seed**, fixed
  per `(variant, target)`, so the method, the upper bound and the floor
  differ only in their predicted fold changes. Arms that also differed in
  which control cells they drew would be compared across sampling noise.
* **variants 1 and 3 are keyed `official_panel_assisted`**, never under
  variant 2's key: they are fed true panel values, so their official metrics
  are not end-to-end performance and must not be able to be read as though
  they were.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..eval import nochange, official
from ..predict.generator import GeneratorSettings, apply_settings, generate_counts, sample_control_cells

METHOD, UPPER_BOUND, FLOOR = "method", "upper_bound", "floor"

#: Key for the official metrics of a variant whose panel values are the
#: truth. Deliberately not `official`.
PANEL_ASSISTED = "official_panel_assisted"
END_TO_END = "official"

#: Floor for the counts-per-10k conversion, so a gene that is zero in the
#: control does not produce an infinite fold change.
CPM_EPSILON = 1e-6


def arm_seed(variant: str, target: str, base_seed: int) -> int:
    """One seed per (variant, target), shared by every arm."""
    material = f"{base_seed}:{variant}:{target}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32)


def sd_units_to_log2fc(
    predicted_sd: np.ndarray, ctrl_mean: np.ndarray, ctrl_std: np.ndarray
) -> np.ndarray:
    """Control-SD units to a log2 fold change the generator can apply.

    The model works in control-SD units, `z = (log1p(CP10K) - mean) / std`
    (§3.2); the generator works in count space (§4.7). The bridge undoes the
    standardization, undoes the log1p, and takes the ratio against the
    control's own level.

    `expm1(mean of log1p)` is not the mean of the counts — Jensen — so this
    is the fold change between two *typical* cells rather than between two
    group means. It is the quantity a per-cell multiplicative change needs,
    and the generator applies it per cell.
    """
    predicted_lognorm = np.asarray(predicted_sd, dtype=np.float64) * ctrl_std + ctrl_mean
    predicted_cpm = np.expm1(np.maximum(predicted_lognorm, 0.0))
    control_cpm = np.expm1(np.maximum(np.asarray(ctrl_mean, dtype=np.float64), 0.0))
    return np.log2((predicted_cpm + CPM_EPSILON) / (control_cpm + CPM_EPSILON)).astype(np.float32)


@dataclass
class TargetPrediction:
    """One arm's prediction for one target, ready for the generator."""

    target: str
    log2_fold_change: np.ndarray
    detail: dict[str, Any] = field(default_factory=dict)


def build_cells(
    predictions: dict[str, TargetPrediction],
    control_counts: np.ndarray,
    cells_per_target: dict[str, int],
    settings: GeneratorSettings,
    variant: str,
    base_seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Generate every target's cells, sharing the draw across arms.

    The control rows are drawn from a generator seeded per (variant, target),
    so every arm scoring this variant draws the *same* cells and the
    comparison between them is about the fold changes alone.
    """
    blocks, labels = [], []
    for target in sorted(predictions):
        n_cells = cells_per_target[target]
        rng = np.random.default_rng(arm_seed(variant, target, base_seed))
        rows = sample_control_cells(
            control_counts.shape[0], n_cells, rng, allow_replacement=settings.allow_replacement
        )
        applied = apply_settings(predictions[target].log2_fold_change, settings)
        blocks.append(generate_counts(control_counts[rows], applied, rng))
        labels.extend([target] * n_cells)
    return np.vstack(blocks), labels


def control_rows_for(
    variant: str, target: str, base_seed: int, n_available: int, n_cells: int,
    settings: GeneratorSettings,
) -> np.ndarray:
    """The rows `build_cells` would draw — used by the test that proves the
    arms share them."""
    rng = np.random.default_rng(arm_seed(variant, target, base_seed))
    return sample_control_cells(
        n_available, n_cells, rng, allow_replacement=settings.allow_replacement
    )


def to_anndata(counts: np.ndarray, labels: list[str], gene_names: list[str], pert_col: str):
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    matrix = sp.csr_matrix(np.asarray(counts, dtype=np.int32))
    matrix.eliminate_zeros()
    return ad.AnnData(
        matrix,
        obs=pd.DataFrame(
            {pert_col: labels}, index=[f"cell{i}" for i in range(len(labels))]
        ),
        var=pd.DataFrame(index=list(gene_names)),
    )


def score_arm(
    cfg,
    predicted_counts: np.ndarray,
    predicted_labels: list[str],
    real_counts: np.ndarray,
    real_labels: list[str],
    gene_names: list[str],
    pert_col: str,
    control_label: str,
    scale_reference=None,
) -> dict[str, Any]:
    """Score one arm with the six official metrics.

    `scale_reference`, when the two ends could be built on this dataset, also
    places the arm on the leaderboard's 0–1 scale (§6).
    """
    from . import common  # noqa: F401 — kept for symmetry with the callers

    prediction = to_anndata(predicted_counts, predicted_labels, gene_names, pert_col)
    real = to_anndata(real_counts, real_labels, gene_names, pert_col)

    result = official.score(
        cfg, prediction, real, pert_col=pert_col, control_label=control_label
    )
    scored = {
        **result.as_dict(),
        "normalized": nochange.summarize(result.scored),
    }
    if scale_reference is not None:
        from ..eval import scale as scale_mod

        scored["leaderboard"] = scale_mod.rescale(result, scale_reference)
    return scored


def per_gene_scores(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Correlation and error over the genes a partial variant predicts.

    Variants 1 and 3 predict only part of the gene set, so §4.6 asks for
    these beside whatever the official metrics can be made to say.
    """
    from ..train.loop import pearson

    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    error = predicted - truth
    return {
        "mse": float((error**2).mean()),
        "mae": float(np.abs(error).mean()),
        "pearson": pearson(predicted, truth),
        "mse_no_change": float((truth**2).mean()),
        "mse_ratio_to_no_change": float(
            (error**2).mean() / max(float((truth**2).mean()), 1e-12)
        ),
        "n_genes": int(predicted.shape[-1]) if predicted.ndim else 0,
    }


def zero_prediction(target: str, n_genes: int) -> TargetPrediction:
    """The floor: this perturbation changed nothing."""
    return TargetPrediction(
        target=target,
        log2_fold_change=np.zeros(n_genes, dtype=np.float32),
        detail={"arm": FLOOR},
    )


def as_tensor(values: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)
