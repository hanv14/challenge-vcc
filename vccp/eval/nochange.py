"""What each official metric scores when a prediction says nothing happened.

From CLAUDE.md §6's table and the metrics brief. These are the points the
sanity checks test against (§6.1 check 6) and the origin the calibration
objective measures from (§4.7, DECISIONS.md D5) — two of the six are
lower-is-better and all six live on different scales, so their raw mean is
not a quantity worth maximizing.

They are the *no-change* points, not the leaderboard's zero. The leaderboard
rescales against the organizers' mean-response baseline, which is a stronger
arm than predicting nothing; `eval/scale.py` builds that where the data allow.
"""

from __future__ import annotations

from dataclasses import dataclass

#: (value when the prediction says nothing happened, higher-is-better)
NO_CHANGE: dict[str, tuple[float, bool]] = {
    "pds_cosine": (0.5, True),
    "expr_mse_unbiased_capped_norm": (1.0, False),
    "de_wilcoxon_lfc_nmae": (1.0, False),
    "de_wilcoxon_direction_fidelity_yield_raw": (0.0, True),
    "de_wilcoxon_direction_reach_raw": (0.0, True),
    "de_wilcoxon_sig_jaccard": (0.0, True),
}


@dataclass(frozen=True)
class Normalized:
    """One metric, mapped so that no-change is 0 and better is positive."""

    metric: str
    raw: float
    no_change: float
    higher_is_better: bool
    value: float


def normalize(metric: str, value: float) -> Normalized | None:
    """Map a metric onto the common scale, or None if it is not one of the six.

    For a higher-is-better metric bounded by 1 the scale is the distance left
    to perfection, so 1.0 means perfect and 0 means no better than saying
    nothing. For the two error metrics, whose no-change point is 1.0 and whose
    perfection is 0, it is the fraction of that gap closed.
    """
    entry = NO_CHANGE.get(metric)
    if entry is None:
        return None
    no_change, higher = entry

    if higher:
        span = 1.0 - no_change
        normalized = (value - no_change) / span if span else value - no_change
    else:
        # Perfection is 0 and no-change is `no_change`, so closing the gap
        # gives 1 and scoring the no-change value gives 0.
        normalized = (no_change - value) / no_change if no_change else -value

    return Normalized(metric, float(value), no_change, higher, float(normalized))


def objective(metrics: dict[str, float]) -> float:
    """The calibration objective: the mean over the six, on the common scale.

    Reported alongside the raw values, never instead of them.
    """
    scores = [n.value for n in (normalize(m, v) for m, v in metrics.items()) if n is not None]
    if not scores:
        return float("nan")
    return float(sum(scores) / len(scores))


def summarize(metrics: dict[str, float]) -> dict:
    """Every member on both scales, plus the objective."""
    normalized = {}
    for metric, value in metrics.items():
        entry = normalize(metric, value)
        if entry is not None:
            normalized[metric] = {
                "raw": entry.raw,
                "no_change": entry.no_change,
                "higher_is_better": entry.higher_is_better,
                "from_no_change": round(entry.value, 6),
            }
    return {
        "members": normalized,
        "objective": objective(metrics),
        "objective_definition": "mean over the six official metrics, each mapped so "
        "that the no-change prediction scores 0 and perfection scores 1 "
        "(DECISIONS.md D5)",
    }


def beats_no_change(metrics: dict[str, float]) -> dict[str, bool]:
    """Per metric, whether it beat saying nothing happened (§6.1 check 6)."""
    out = {}
    for metric, value in metrics.items():
        entry = normalize(metric, value)
        if entry is not None:
            out[metric] = entry.value > 0
    return out
