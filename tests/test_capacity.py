"""The capacity diagnostic: can the perturbation module fit at all?

Phase 2 reports it at 1.036 on its own training targets. Before redesigning
what it is supervised against, this asks whether it can fit targets it is
allowed to memorize — and these tests check that the diagnostic reports
honestly, including the failure mode it exists to catch.
"""

from __future__ import annotations

import pytest

from vccp.diagnostics import capacity


def test_it_reports_a_curve_and_a_verdict(pipeline_cfg, pipeline_run):
    """Few steps: this checks the wiring, not the answer. It needs a finished
    `priors` stage, which the shared pipeline run provides."""
    report = capacity.run_capacity_check(pipeline_cfg, n_targets=4, steps=6, log_every=3)

    assert report["n_targets"] == 4
    assert report["curve"], "no checkpoint was recorded"
    for point in report["curve"]:
        for key in ("ratio_to_no_change", "pearson", "magnitude_ratio"):
            assert key in point
    assert report["verdict"] in ("can-fit", "cannot-fit", "collapses", "no-data")
    assert report["meaning"]


def test_a_module_that_predicts_nothing_is_called_out():
    """The failure the pipeline has been showing: error near the no-change
    line *and* predicted changes far too small."""
    result = capacity.CapacityResult(screen="s", targets=["A"], steps=10)
    result.curve.append(
        capacity.Checkpoint(step=10, loss=1.0, ratio=1.01, pearson=0.0, magnitude_ratio=0.01)
    )
    verdict, meaning = result.verdict()
    assert verdict == "collapses"
    assert "no change" in meaning


def test_failing_to_fit_while_predicting_plausible_sizes_reads_differently():
    """Same error, very different cause: this one points at conditioning."""
    result = capacity.CapacityResult(screen="s", targets=["A"], steps=10)
    result.curve.append(
        capacity.Checkpoint(step=10, loss=1.0, ratio=1.02, pearson=0.0, magnitude_ratio=0.9)
    )
    verdict, meaning = result.verdict()
    assert verdict == "cannot-fit"
    assert "conditioning" in meaning


def test_fitting_is_reported_as_fitting():
    result = capacity.CapacityResult(screen="s", targets=["A"], steps=10)
    result.curve.append(
        capacity.Checkpoint(step=10, loss=0.01, ratio=0.04, pearson=0.98, magnitude_ratio=1.0)
    )
    assert result.verdict()[0] == "can-fit"


def test_it_refuses_a_screen_without_enough_challenge_gene_targets(
    pipeline_cfg, pipeline_run
):
    with pytest.raises(RuntimeError, match="no screen has"):
        capacity.run_capacity_check(pipeline_cfg, n_targets=10_000, steps=1)
