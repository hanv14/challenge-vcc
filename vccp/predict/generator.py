"""Turning predicted fold changes into cells (CLAUDE.md §4.7).

The official metrics compute DE on the *submitted cells*, with a Wilcoxon
test against the real controls, and reward realistic cell-to-cell variation.
So the generator is part of the method: a group of 400 identical cells
carrying the right mean would score badly on the expression metric and would
have no dispersion for the DE test to work with.

What it does, per (context, target):

* draws a **fresh, independent** sample of that context's control cells. Never
  one block reused across targets — the expression metric's
  across-perturbation budget withholds the whole sampling correction from a
  submission whose predicted variation cancels in the pseudobulk (§2.3, #348);
* applies the predicted per-gene fold changes **in count space, keeping
  integers**: binomial thinning for decreases, scaling with stochastic
  rounding for increases, so each cell's dispersion stays what a count
  process would give;
* holds genes below the confidence threshold at a fold change of **exactly
  one**, so their cells are exchangeable with the controls and the DE test
  has no reason to call them. The DE metrics penalize calling too many genes
  (the Jaccard divides by the union) and too few (fidelity is scaled by
  yield), and the expression metric scores a prediction of no change at 1.0 —
  so a noisy guess is worse than silence.

The two calibration settings — the confidence threshold and the effect-size
scale — are fitted on the rehearsal and live in `phase3_policy.json`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Fold changes are clipped to this before being applied, so a runaway
#: prediction cannot produce a cell of 10^9 counts. Well outside anything
#: `sanity.max_abs_log2fc` would allow through.
MAX_FOLD_CHANGE = 1e4


@dataclass(frozen=True)
class GeneratorSettings:
    """The two knobs the rehearsal calibrates, plus the sampling policy."""

    #: |log2 fold change| below this is set to exactly zero.
    confidence_threshold: float = 0.05
    #: Every surviving log2 fold change is multiplied by this.
    effect_scale: float = 1.0
    #: Resample control cells with replacement when a context has fewer
    #: control cells than the submission needs per target.
    allow_replacement: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "effect_scale": self.effect_scale,
            "allow_replacement": self.allow_replacement,
        }

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> GeneratorSettings:
        return cls(
            confidence_threshold=float(spec.get("confidence_threshold", 0.05)),
            effect_scale=float(spec.get("effect_scale", 1.0)),
            allow_replacement=bool(spec.get("allow_replacement", True)),
        )


def apply_settings(log2_fold_change: np.ndarray, settings: GeneratorSettings) -> np.ndarray:
    """Threshold, then scale. Returns log2 fold changes.

    Thresholding happens *before* scaling, so the threshold means what it says
    about the model's own output rather than about the output after shrinking.
    """
    values = np.asarray(log2_fold_change, dtype=np.float32).copy()
    values[~np.isfinite(values)] = 0.0
    values[np.abs(values) < settings.confidence_threshold] = 0.0
    return values * np.float32(settings.effect_scale)


def sample_control_cells(
    n_available: int, n_needed: int, rng: np.random.Generator, *, allow_replacement: bool = True
) -> np.ndarray:
    """Row indices for one target's fresh draw of control cells."""
    if n_available < 1:
        raise ValueError("the context has no control cells to sample")
    if n_needed <= n_available:
        return rng.choice(n_available, n_needed, replace=False)
    if not allow_replacement:
        raise ValueError(
            f"{n_needed} cells were asked for but the context has {n_available} "
            "control cells; set allow_replacement to draw with replacement"
        )
    return rng.choice(n_available, n_needed, replace=True)


def generate_counts(
    control_counts: np.ndarray,
    log2_fold_change: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply per-gene fold changes to control cells, in count space.

    `control_counts` is `(n_cells, n_genes)` of raw integer counts;
    `log2_fold_change` is `(n_genes,)` and is applied to every cell.

    A gene whose fold change is exactly 1 comes back bit-for-bit unchanged —
    that is what makes its cells exchangeable with the controls.
    """
    counts = np.asarray(control_counts)
    if counts.ndim != 2:
        raise ValueError(f"expected (cells, genes), got shape {counts.shape}")
    if log2_fold_change.shape[0] != counts.shape[1]:
        raise ValueError(
            f"{log2_fold_change.shape[0]} fold changes for {counts.shape[1]} genes"
        )

    fold = np.clip(np.exp2(np.asarray(log2_fold_change, dtype=np.float64)), 0.0, MAX_FOLD_CHANGE)
    out = counts.astype(np.int64, copy=True)

    unchanged = fold == 1.0
    decreasing = (fold < 1.0) & ~unchanged
    increasing = (fold > 1.0) & ~unchanged

    if decreasing.any():
        # Binomial thinning: keep each count with probability `fold`. This is
        # what down-regulation looks like to a counting process, and it keeps
        # the mean-variance relationship a real cell has.
        block = out[:, decreasing]
        out[:, decreasing] = rng.binomial(block, fold[decreasing][None, :])

    if increasing.any():
        # Scaling up has no exact integer analogue, so the fractional part is
        # resolved by a coin flip per cell and gene: unbiased in the mean, and
        # it adds the scatter that rounding would remove.
        scaled = out[:, increasing] * fold[increasing][None, :]
        out[:, increasing] = stochastic_round(scaled, rng)

    return out


def stochastic_round(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Round to integers, keeping the mean: 2.3 goes to 2 or 3 with p = 0.7/0.3."""
    floor = np.floor(values)
    return (floor + (rng.random(values.shape) < (values - floor))).astype(np.int64)


def generate_cells(
    control_counts: np.ndarray,
    log2_fold_change: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
    settings: GeneratorSettings,
) -> np.ndarray:
    """One target's predicted cells: draw controls, then apply the changes."""
    rows = sample_control_cells(
        control_counts.shape[0], n_cells, rng, allow_replacement=settings.allow_replacement
    )
    drawn = control_counts[rows]
    return generate_counts(drawn, apply_settings(log2_fold_change, settings), rng)


def describe(settings: GeneratorSettings, log2_fold_change: np.ndarray) -> dict[str, Any]:
    """What the settings did to one target's prediction, for the reports."""
    raw = np.asarray(log2_fold_change, dtype=np.float32)
    applied = apply_settings(raw, settings)
    moved = applied != 0.0
    return {
        **settings.as_dict(),
        "n_genes": int(raw.size),
        "n_genes_moved": int(moved.sum()),
        "fraction_moved": float(moved.mean()) if raw.size else 0.0,
        "max_abs_log2fc": float(np.abs(applied).max()) if raw.size else 0.0,
        "mean_abs_log2fc_moved": float(np.abs(applied[moved]).mean()) if moved.any() else 0.0,
    }


#: The generator is a swappable component (§4.7); this is the registry the
#: config's `predict.generator` name resolves through.
GENERATORS = {"count_space": generate_cells}


def get_generator(name: str):
    if name not in GENERATORS:
        raise KeyError(f"unknown generator {name!r}; available: {sorted(GENERATORS)}")
    return GENERATORS[name]
