"""The knockdown prior on the target gene's own expression (CLAUDE.md §4.7).

A CRISPRi knockdown of gene g should lower g. The model is not asked to learn
that: it is measured, it is the most reliable single thing known about any
perturbation, and the sanity checks test for it (§6.1 check 3).

    predicted g = control g x fold_expr

`fold_expr` is Replogle's measured knockdown efficiency — the target's
expression in perturbed cells over control. Per target where the screens have
it, otherwise the pooled median over every screen present.

Two rules from the M2 review (DECISIONS.md D8):

* the **resolved source is recorded per target**, because the challenge
  contexts are not identified with any Replogle cell line and the medians
  differ by screen (0.155 K562, 0.088 RPE1) — the number used is a transfer
  assumption and should be visible as one;
* the prior applies only where the target gene is **detectably expressed** in
  that context's controls. Applying a fold change to a gene that is not
  expressed would manufacture a change out of noise.

None of the six official metrics scores a target's own gene (§6), so this
moves no score. It is here because it is correct biology and part of the
design, and because check 3 would otherwise have nothing to find.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

POOLED = "pooled"
PER_TARGET = "per_target"
NOT_APPLIED = "not_applied"


@dataclass
class KnockdownPrior:
    """Measured knockdown efficiency, per target and pooled."""

    per_target: dict[str, float] = field(default_factory=dict)
    #: target -> which screen the per-target value came from.
    source_screen: dict[str, str] = field(default_factory=dict)
    pooled_median: float = 1.0
    screens: dict[str, float] = field(default_factory=dict)

    def resolve(self, target: str) -> tuple[float, str]:
        """`(fold_expr, source)` for one target."""
        if target in self.per_target:
            return self.per_target[target], f"{PER_TARGET}:{self.source_screen[target]}"
        return self.pooled_median, POOLED

    def as_dict(self) -> dict[str, Any]:
        return {
            "pooled_median_fold_expr": self.pooled_median,
            "median_per_screen": self.screens,
            "n_targets_with_own_value": len(self.per_target),
        }


def load_knockdown_prior(cfg, paths) -> KnockdownPrior:
    """Read `fold_expr` from every Replogle bulk file present."""
    from ..data import replogle as replogle_data

    per_target: dict[str, list[float]] = {}
    source: dict[str, str] = {}
    screens: dict[str, float] = {}
    everything: list[float] = []

    for name, path in sorted(paths.discover_replogle_bulk().items()):
        if not name.endswith("_raw_bulk"):
            continue
        screen = name[: -len("_raw_bulk")]
        series = replogle_data.load_fold_expr(path)
        if series.empty:
            continue

        screens[screen] = float(series.median())
        everything.extend(series.to_list())
        for target, value in series.items():
            per_target.setdefault(str(target), []).append(float(value))
            source.setdefault(str(target), screen)

    if not everything:
        raise RuntimeError(
            "no Replogle bulk file carries fold_expr, so the knockdown prior has "
            "no measured value to use"
        )

    # A target measured in several screens: the median over them, and the
    # first screen alphabetically recorded as where it came from.
    resolved = {target: float(np.median(values)) for target, values in per_target.items()}
    multi = {t for t, v in per_target.items() if len(v) > 1}
    for target in multi:
        source[target] = "+".join(
            sorted(s for s, _ in screens.items() if target in per_target)
        ) or source[target]

    return KnockdownPrior(
        per_target=resolved,
        source_screen=source,
        pooled_median=float(np.median(everything)),
        screens=screens,
    )


def apply_to_log2fc(
    log2_fold_change: np.ndarray,
    target_position: int | None,
    fold_expr: float,
    control_mean_cpm: float,
    min_control_cpm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Overwrite the target gene's own predicted change with the prior.

    Returns the modified fold changes and a record of what was decided —
    which is what `phase3/knockdown_<context>.csv` and the knockdown figure
    are built from.

    `log2_fold_change` is one profile `(n_genes,)` or a per-cell block
    `(n_cells, n_genes)`. The prior overwrites a **column** either way: it is
    a measured property of the perturbation, the same for every cell of the
    target, and `fold_expr` is a population quantity to begin with.
    """
    values = np.asarray(log2_fold_change, dtype=np.float32).copy()

    if target_position is None:
        return values, {
            "applied": False,
            "reason": "the target is not a challenge gene, so it has no column",
            "control_mean_cpm": float(control_mean_cpm),
        }
    if not np.isfinite(control_mean_cpm) or control_mean_cpm < min_control_cpm:
        return values, {
            "applied": False,
            "reason": f"not detectably expressed in this context's controls "
            f"({control_mean_cpm:.3g} < {min_control_cpm:g} CPM)",
            "control_mean_cpm": float(control_mean_cpm),
        }

    fold = max(float(fold_expr), 1e-6)
    before = float(np.mean(values[..., target_position]))
    values[..., target_position] = np.log2(fold)
    return values, {
        "applied": True,
        "fold_expr": fold,
        "log2_fold_change": float(np.log2(fold)),
        # The model's own answer for this gene, averaged over the cells when
        # the prediction was per cell. Kept because the gap between it and
        # the measured value is the only reading we have on whether the
        # perturbation module has learned what a knockdown does.
        "model_log2_fold_change": before,
        "control_mean_cpm": float(control_mean_cpm),
    }


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """The per-target knockdown table written beside each context."""
    return pd.DataFrame(records)
