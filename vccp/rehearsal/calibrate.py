"""Fitting the generator's two settings on the rehearsal (CLAUDE.md §4.7).

A confidence threshold below which a predicted change is set to zero, and a
global scale applied to what survives. Both are fitted on variant 2 — the one
that produces whole cells and is scored exactly as the challenge scores them.

**The objective is not the raw mean of the six metrics.** Two of them are
lower-is-better and all six live on different scales, so their raw mean is not
a quantity worth maximizing. Each is mapped so that its documented no-change
value is 0 and perfection is 1, and the mean of *that* is maximized
(DECISIONS.md D5). The objective actually used is written into the policy.

Why these two knobs and not more: the metrics punish a prediction in two
opposite directions at once. The Jaccard divides by the union, so calling too
many genes costs; fidelity is scaled by yield and reach is measured against
everything the reference called, so calling too few costs. The threshold
trades one against the other and the scale sets how far the surviving calls
move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..eval import nochange
from ..logging_utils import get_logger
from ..predict.generator import GeneratorSettings


@dataclass
class CalibrationPoint:
    """One grid point, scored."""

    threshold: float
    scale: float
    objective: float
    metrics: dict[str, float] = field(default_factory=dict)
    n_genes_moved: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence_threshold": self.threshold,
            "effect_scale": self.scale,
            "objective": self.objective,
            "metrics": self.metrics,
            "mean_genes_moved": self.n_genes_moved,
            "error": self.error,
        }


@dataclass
class Calibration:
    """The grid, and what it chose."""

    settings: GeneratorSettings
    best: CalibrationPoint | None
    grid: list[CalibrationPoint] = field(default_factory=list)
    context: str | None = None
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings.as_dict(),
            "objective_definition": nochange.summarize({})["objective_definition"],
            "fitted_on": self.context,
            "best": self.best.as_dict() if self.best else None,
            "grid": [point.as_dict() for point in self.grid],
            "fallback_reason": self.fallback_reason,
            "note": "the threshold is applied to the model's own output before the "
            "scale, so it means what it says about the model rather than about "
            "the output after shrinking",
        }


def calibrate(
    cfg,
    predictions,
    control_counts: np.ndarray,
    real_counts: np.ndarray,
    real_labels: list[str],
    cells_per_target: dict[str, int],
    gene_names: list[str],
    variant: str,
    control_label: str,
) -> Calibration:
    """Search the grid, maximizing the normalized objective."""
    from . import common

    log = get_logger()
    truth_counts = np.vstack([real_counts, control_counts])
    truth_labels = list(real_labels) + [control_label] * control_counts.shape[0]

    grid: list[CalibrationPoint] = []
    for threshold in cfg.rehearsal.calibration_thresholds:
        for scale in cfg.rehearsal.calibration_scales:
            settings = GeneratorSettings(
                confidence_threshold=float(threshold), effect_scale=float(scale)
            )
            moved = float(
                np.mean(
                    [
                        (
                            np.abs(
                                np.asarray(p.log2_fold_change)[
                                    np.abs(np.asarray(p.log2_fold_change)) >= threshold
                                ]
                            ).size
                        )
                        for p in predictions.values()
                    ]
                )
            )
            try:
                counts, labels = common.build_cells(
                    predictions, control_counts, cells_per_target, settings, variant, cfg.seed
                )
                scored = common.score_arm(
                    cfg, counts, labels, truth_counts, truth_labels, gene_names,
                    "target_gene", control_label,
                )
                metrics = scored["scored"]
                grid.append(
                    CalibrationPoint(
                        threshold=float(threshold),
                        scale=float(scale),
                        objective=nochange.objective(metrics),
                        metrics=metrics,
                        n_genes_moved=moved,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — a failed point is a result
                grid.append(
                    CalibrationPoint(
                        threshold=float(threshold), scale=float(scale),
                        objective=float("-inf"), n_genes_moved=moved,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    usable = [p for p in grid if np.isfinite(p.objective)]
    if not usable:
        # No grid point scored. Predicting nothing is the safe default: the
        # expression metric scores no change at 1.0 and the DE metrics cannot
        # punish calls that were never made (§4.7).
        return Calibration(
            settings=GeneratorSettings(confidence_threshold=float("inf"), effect_scale=0.0),
            best=None,
            grid=grid,
            fallback_reason="no grid point could be scored, so the generator is set to "
            "predict no change at all — the point the metrics treat as neutral",
        )

    best = max(usable, key=lambda p: p.objective)
    log.info(
        "  calibration: threshold %.3g, scale %.3g  ->  objective %+.4f "
        "(%d of %d grid points scored)",
        best.threshold, best.scale, best.objective, len(usable), len(grid),
    )
    return Calibration(
        settings=GeneratorSettings(
            confidence_threshold=best.threshold, effect_scale=best.scale
        ),
        best=best,
        grid=grid,
    )


def default_calibration(reason: str) -> Calibration:
    """When variant 2 could not run at all."""
    return Calibration(
        settings=GeneratorSettings(),
        best=None,
        grid=[],
        fallback_reason=reason,
    )
