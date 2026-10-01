"""The leaderboard's number is the `from_replicate` average (D120).

`cell_eval2.score_metrics` returns two scaled columns. Both put 0 at the
organizers' mean-response baseline; `from_baseline` puts 1 at each metric's
constant perfection anchor, `from_replicate` at the measured split-half
replicate. The leaderboard enrols `from_replicate`. These tests need no
scorer: they feed `placement` the frame `score_metrics` returns, keyed by
metric, and check which column becomes the score.
"""

from __future__ import annotations

import pytest

from vccp.eval import scale
from vccp.eval.official import SCORED

# rpe1's method arm in `runs/server_shared` (D121), as cell-eval2 scored it.
REPLICATE = {
    "pds_cosine": -0.019,
    "expr_mse_unbiased_capped_norm": 0.0,
    "de_wilcoxon_lfc_nmae": -0.628,
    "de_wilcoxon_direction_fidelity_yield_raw": 0.281,
    "de_wilcoxon_direction_reach_raw": -0.583,
    "de_wilcoxon_sig_jaccard": -0.019,
}
BASELINE = {
    "pds_cosine": -0.018,
    "expr_mse_unbiased_capped_norm": 0.0,
    "de_wilcoxon_lfc_nmae": -0.120,
    "de_wilcoxon_direction_fidelity_yield_raw": 0.180,
    "de_wilcoxon_direction_reach_raw": -0.530,
    "de_wilcoxon_sig_jaccard": -0.014,
}


def frame(baseline_average: float | None, *, drop: str | None = None):
    rows = {
        m: {"metric": m, "from_baseline": BASELINE[m],
            "from_replicate": None if m == drop else REPLICATE[m]}
        for m in SCORED
    }
    # A diagnostic the profile emits but does not score: no replicate value.
    rows["expr_mse_unbiased"] = {"metric": "expr_mse_unbiased", "from_baseline": 0.3,
                                 "from_replicate": None}
    rows["avg_score"] = {"metric": "avg_score", "from_baseline": baseline_average,
                         "from_replicate": sum(REPLICATE.values()) / len(SCORED)}
    return rows


def test_the_score_is_the_mean_of_the_six_from_replicate_values():
    placed = scale.placement(frame(-0.0837), "rpe1")
    assert placed["available"] is True
    assert placed["column"] == "from_replicate"
    assert placed["avg_score"] == pytest.approx(sum(REPLICATE.values()) / 6)
    assert placed["avg_score"] == pytest.approx(-0.16133, abs=1e-4)
    assert placed["from_replicate"] == REPLICATE


def test_from_baseline_is_kept_as_a_diagnostic_and_never_scores():
    placed = scale.placement(frame(-0.0837), "rpe1")
    assert placed["from_baseline_avg_score"] == pytest.approx(-0.0837)
    assert placed["avg_score"] != pytest.approx(placed["from_baseline_avg_score"])
    assert placed["from_baseline"]["de_wilcoxon_lfc_nmae"] == BASELINE["de_wilcoxon_lfc_nmae"]
    # A diagnostic bundle keeps both averages, so it is not enrolled.
    assert placed["enrolled"] is False


def test_a_competition_bundle_scores_the_same():
    """An enrolled bundle nulls the from_baseline average; the score does not move."""
    placed = scale.placement(frame(None), "rpe1")
    assert placed["enrolled"] is True
    assert placed["from_baseline_avg_score"] is None
    assert placed["avg_score"] == pytest.approx(-0.16133, abs=1e-4)


def test_a_member_missing_from_the_replicate_column_is_not_averaged_away():
    placed = scale.placement(frame(-0.0837, drop="de_wilcoxon_sig_jaccard"), "rpe1")
    assert placed["available"] is False
    assert "de_wilcoxon_sig_jaccard" in placed["reason"]
    assert "avg_score" not in placed


def test_a_grid_point_says_which_column_it_was_read_from():
    """A point read back from a pre-D120 report must not claim the new column."""
    from vccp.rehearsal.calibrate import CalibrationPoint

    old = {"confidence_threshold": 0.05, "effect_scale": 1.0, "shared_scale": 2.0,
           "objective": 0.0, "leaderboard": -0.0395}
    assert CalibrationPoint.from_dict(old).leaderboard_column is None
    fresh = CalibrationPoint(threshold=0.05, scale=1.0, objective=0.0, leaderboard=-0.1256)
    assert fresh.as_dict()["leaderboard_column"] == "from_replicate"
    assert CalibrationPoint.from_dict(fresh.as_dict()).leaderboard_column == "from_replicate"
