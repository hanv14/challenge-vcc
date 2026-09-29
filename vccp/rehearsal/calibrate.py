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
    #: The mean of the six on the leaderboard's scale (0 = the organizers'
    #: mean-response baseline, 1 = a split-half replicate), when the
    #: rehearsal could build that scale for this screen. This, not
    #: `objective`, is what the calibration maximizes when it is there (D113).
    leaderboard: float | None = None
    leaderboard_metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence_threshold": self.threshold,
            "effect_scale": self.scale,
            "objective": self.objective,
            "leaderboard": self.leaderboard,
            "leaderboard_metrics": self.leaderboard_metrics,
            "metrics": self.metrics,
            "mean_genes_moved": self.n_genes_moved,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, entry: dict[str, Any]) -> "CalibrationPoint":
        return cls(
            threshold=float(entry["confidence_threshold"]),
            scale=float(entry["effect_scale"]),
            objective=float(entry["objective"]),
            metrics=dict(entry.get("metrics") or {}),
            n_genes_moved=float(entry.get("mean_genes_moved") or 0.0),
            error=entry.get("error"),
            leaderboard=entry.get("leaderboard"),
            leaderboard_metrics=dict(entry.get("leaderboard_metrics") or {}),
        )


@dataclass
class Calibration:
    """The grid, and what it chose."""

    settings: GeneratorSettings
    best: CalibrationPoint | None
    grid: list[CalibrationPoint] = field(default_factory=list)
    context: str | None = None
    fallback_reason: str | None = None
    #: The perturbation *modality* the settings were fitted on, and the one
    #: they are applied to. These usually match — Replogle and the challenge
    #: are both CRISPR interference — which is exactly why the recording must
    #: not stop there: the two are still different labs, protocols and
    #: sequencing depths (PLAN_PERCELL.md §7). `carried_across_assays` says so
    #: unconditionally, and the spread in `per_dataset` is the measurement.
    fitted_assay: str | None = None
    applied_assay: str | None = None
    #: The same fit on every other usable screen, when
    #: `rehearsal.calibrate_per_dataset` asked for it. Screens that agree are
    #: evidence the transfer is safe; screens that disagree are evidence it
    #: is not, and either reading is worth more than the single number.
    per_dataset: dict[str, Any] = field(default_factory=dict)
    #: "leaderboard" when every screen could be placed on the leaderboard's
    #: scale and the choice maximized it, "no_change" otherwise (D113).
    objective_used: str = "no_change"
    #: Per grid point, the value on each screen and their mean.
    combined: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings.as_dict(),
            "objective_used": self.objective_used,
            "objective_definition": (
                "mean over the cross-context screens of the leaderboard score "
                "(0 = the organizers' mean-response baseline, 1 = a split-half "
                "replicate), as cell-eval2 computes it (D113)"
                if self.objective_used == LEADERBOARD
                else nochange.summarize({})["objective_definition"]
            ),
            "combined": self.combined,
            "fitted_on": self.context,
            "fitted_modality": self.fitted_assay,
            "applied_modality": self.applied_assay,
            "modality_differs": bool(
                self.fitted_assay
                and self.applied_assay
                and self.fitted_assay != self.applied_assay
            ),
            # Said explicitly because `modality_differs: false` would
            # otherwise read as "no transfer risk here", and that is wrong.
            # Replogle and the challenge are both CRISPR interference, so the
            # modality matches — but they are different labs, different
            # protocols and ~20,000 UMIs per cell against 11,000-15,000. These
            # settings are still carried from one assay to another, and
            # nothing in the model represents that difference. The spread in
            # `per_dataset` is the only measurement of whether it matters.
            "carried_across_assays": True,
            "assay_note": (
                "fitted on the rehearsal's Replogle screens and applied to the "
                "challenge. A matching modality is not the same assay: the two "
                "differ in lab, protocol and sequencing depth, and that "
                "difference is not represented in the model"
            ),
            "per_dataset": self.per_dataset,
            "best": self.best.as_dict() if self.best else None,
            "grid": [point.as_dict() for point in self.grid],
            "fallback_reason": self.fallback_reason,
            "note": "the threshold is applied to the model's own output before the "
            "scale, so it means what it says about the model rather than about "
            "the output after shrinking",
        }


#: The two objectives a calibration can maximize.
LEADERBOARD, NO_CHANGE = "leaderboard", "no_change"


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
    scale_reference=None,
) -> Calibration:
    """Search the grid on one screen. Each point is scored on the no-change
    objective and, when `scale_reference` is given, on the leaderboard's
    scale; `combine` chooses across screens."""
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
                    "target_gene", control_label, scale_reference=scale_reference,
                )
                metrics = scored["scored"]
                board = scored.get("leaderboard") or {}
                grid.append(
                    CalibrationPoint(
                        threshold=float(threshold),
                        scale=float(scale),
                        objective=nochange.objective(metrics),
                        metrics=metrics,
                        n_genes_moved=moved,
                        leaderboard=(
                            float(board["avg_score"])
                            if board.get("available") and board.get("avg_score") is not None
                            else None
                        ),
                        leaderboard_metrics=dict(board.get("from_baseline") or {}),
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


def combine(grids: dict[str, list[CalibrationPoint]]) -> Calibration:
    """One setting for every screen: the grid point with the best mean.

    The mean is over the leaderboard scale when every screen has it for that
    point, which is what the leaderboard ranks on. It falls back to the
    no-change objective only when no screen could be placed on that scale.
    The two disagree where it matters most: predicting almost nothing scores
    near 0 on the no-change objective and far below 0 on the leaderboard,
    because the leaderboard's 0 is a mean-response baseline that does call
    genes. Calibrating on the no-change objective chose exactly that corner,
    and the upload lost 1.34 on DE direction fidelity (D113).
    """
    log = get_logger()
    screens = sorted(grids)
    keyed: dict[tuple[float, float], dict[str, CalibrationPoint]] = {}
    for screen in screens:
        for point in grids[screen]:
            keyed.setdefault((point.threshold, point.scale), {})[screen] = point

    def finite(point, attr) -> bool:
        value = getattr(point, attr, None) if point is not None else None
        return value is not None and bool(np.isfinite(value))

    def scored_on(attr) -> list[str]:
        """The screens where at least one grid point has this reading. A
        screen whose scorer failed everywhere says nothing about any point,
        and must not veto them all."""
        return [s for s in screens if any(finite(p, attr) for p in grids[s])]

    board_screens = scored_on("leaderboard")
    use_board = bool(board_screens)
    attr = "leaderboard" if use_board else "objective"
    screens = board_screens if use_board else scored_on("objective")

    def complete(points, attr):
        values = [getattr(points.get(s), attr, None) if points.get(s) else None for s in screens]
        return bool(screens) and all(
            v is not None and np.isfinite(v) for v in values
        ), values

    combined, best_key, best_value = [], None, float("-inf")
    for key, points in sorted(keyed.items()):
        ok, values = complete(points, attr)
        mean = float(np.mean(values)) if ok else float("-inf")
        combined.append({
            "confidence_threshold": key[0],
            "effect_scale": key[1],
            "per_screen": {s: v for s, v in zip(screens, values)},
            "mean": mean if ok else None,
        })
        if ok and mean > best_value:
            best_key, best_value = key, mean

    if best_key is None:
        return Calibration(
            settings=GeneratorSettings(confidence_threshold=float("inf"), effect_scale=0.0),
            best=None,
            grid=[p for s in screens for p in grids[s]],
            fallback_reason="no grid point could be scored on every screen, so the "
            "generator is set to predict no change at all",
            combined=combined,
        )

    per_screen_best = {}
    for screen in screens:
        usable = [p for p in grids[screen] if getattr(p, attr) is not None
                  and np.isfinite(getattr(p, attr))]
        if usable:
            top = max(usable, key=lambda p: getattr(p, attr))
            per_screen_best[screen] = {
                "confidence_threshold": top.threshold, "effect_scale": top.scale,
                attr: getattr(top, attr),
            }

    representative = max(keyed[best_key].values(), key=lambda p: getattr(p, attr) or -np.inf)
    log.info(
        "  calibration over %s: threshold %.3g, scale %.3g  ->  mean %s %+.4f",
        ", ".join(screens), best_key[0], best_key[1],
        "leaderboard score" if use_board else "no-change objective", best_value,
    )
    return Calibration(
        settings=GeneratorSettings(confidence_threshold=best_key[0], effect_scale=best_key[1]),
        best=representative,
        grid=[p for s in sorted(grids) for p in grids[s]],
        context="+".join(screens),
        per_dataset={"ran": True, "fitted_on": "+".join(screens), "others": per_screen_best},
        objective_used=LEADERBOARD if use_board else NO_CHANGE,
        combined=combined,
    )
