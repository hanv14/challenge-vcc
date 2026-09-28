"""The panel→rest ceiling: what a linear map reaches, and whether the target
is there to be predicted (PLAN_MAPPING.md §B1)."""

from __future__ import annotations

import numpy as np
import pytest

from vccp.diagnostics import mapping


# --------------------------------------------------------------------------- #
# the streamed ridge agrees with the dense one
# --------------------------------------------------------------------------- #
def test_streamed_normal_equations_match_a_dense_ridge():
    """The point of accumulating `XᵀX` a block at a time is that the 2-million
    cell file is never held; the answer has to be the one a dense solve gives."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6))
    y = x @ rng.normal(size=(6, 4)) + rng.normal(scale=0.1, size=(200, 4))
    design = mapping._design(x)

    streamed = mapping.Normals(design.shape[1], y.shape[1])
    for start in range(0, 200, 37):  # deliberately uneven blocks
        streamed.add(design[start : start + 37], y[start : start + 37])
    assert streamed.n == 200

    for penalty in (0.0, 0.1, 10.0):
        ridge = np.eye(design.shape[1]) * penalty * 200
        ridge[0, 0] = 0.0
        dense = np.linalg.solve(design.T @ design + ridge, design.T @ y)
        assert np.allclose(streamed.solve(penalty), dense, atol=1e-8)


def test_the_intercept_is_never_penalized():
    """Penalizing the intercept would pull the fit toward zero rather than
    toward the mean, and the targets are z-scored, not centred per sample."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 3))
    y = 7.0 + np.zeros((500, 1))  # a pure offset, nothing to learn from x

    normals = mapping.Normals(4, 1)
    normals.add(mapping._design(x), y)
    for penalty in (0.0, 1000.0):
        coefficients = normals.solve(penalty)
        assert coefficients[0, 0] == pytest.approx(7.0, abs=1e-6)


def test_a_singular_system_does_not_raise():
    """Fewer cells than panel genes makes the unpenalized fit underdetermined,
    which is information about the sample size, not a failure."""
    normals = mapping.Normals(5, 2)
    normals.add(np.ones((3, 5)), np.ones((3, 2)))
    assert np.isfinite(normals.solve(0.0)).all()


# --------------------------------------------------------------------------- #
# the target's reliability
# --------------------------------------------------------------------------- #
def test_split_half_reliability_is_high_for_deep_cells_and_low_for_shallow():
    """A cell sequenced deeply enough measures its own rest genes twice and
    agrees with itself; at one count per gene it cannot."""

    rng = np.random.default_rng(2)
    n_cells, n_genes = 64, 200
    is_panel = np.zeros(n_genes, dtype=bool)
    is_panel[:50] = True
    mean = np.zeros(n_genes, dtype=np.float32)
    std = np.ones(n_genes, dtype=np.float32)

    def reliability(scale):
        rate = rng.gamma(2.0, scale, size=(n_cells, n_genes))
        counts = rng.poisson(rate)
        library = counts.sum(axis=1).astype(np.float64)
        return mapping.split_half_reliability(
            counts, library, mean, std, is_panel, np.random.default_rng(3)
        )

    deep = reliability(200.0)
    shallow = reliability(0.3)
    assert deep["half_depth_pearson"] > 0.8
    assert shallow["half_depth_pearson"] < deep["half_depth_pearson"]
    # Spearman-Brown lifts the half-depth figure toward the full-depth one.
    assert deep["reliability"] > deep["half_depth_pearson"]


def test_the_ceiling_is_the_square_root_of_the_reliability():
    """Two fallible measurements of one quantity correlate `r`; a noiseless
    predictor of the underlying signal correlates `sqrt(r)`. Reporting `r` as
    the ceiling understated it so badly that the server's ridge regression
    came out above it — 0.14 against a claimed ceiling of 0.086 (D101)."""
    rng = np.random.default_rng(4)
    n_cells, n_genes = 256, 400
    is_panel = np.zeros(n_genes, dtype=bool)
    is_panel[:100] = True
    rate = rng.gamma(2.0, 3.0, size=(n_cells, n_genes))
    counts = rng.poisson(rate)
    library = counts.sum(axis=1).astype(np.float64)

    result = mapping.split_half_reliability(
        counts, library, np.zeros(n_genes, np.float32), np.ones(n_genes, np.float32),
        is_panel, np.random.default_rng(5),
    )
    r = result["reliability"]
    assert 0.0 < r < 1.0
    assert result["max_achievable_pearson"] == pytest.approx(np.sqrt(r))
    assert result["max_achievable_pearson"] > r, "sqrt lifts a reliability below 1"
    assert result["best_possible_mse_ratio"] == pytest.approx(1.0 - r)


def test_reliability_is_nan_safe():
    """A constant target has no correlation to report, and reporting 0 for it
    would be read as "measured, and nothing there"."""
    n_cells, n_genes = 8, 12
    is_panel = np.zeros(n_genes, dtype=bool)
    result = mapping.split_half_reliability(
        np.zeros((n_cells, n_genes), dtype=np.int64),
        np.ones(n_cells),
        np.zeros(n_genes, dtype=np.float32),
        np.ones(n_genes, dtype=np.float32),
        is_panel,
        np.random.default_rng(0),
    )
    assert np.isnan(result["half_depth_pearson"])
    assert np.isnan(result["reliability"])
    assert np.isnan(result["max_achievable_pearson"])
    assert np.isnan(result["best_possible_mse_ratio"])


# --------------------------------------------------------------------------- #
# which branch of §B the answer chooses
# --------------------------------------------------------------------------- #
def _screen(linear, model_pearson, ceiling, at_edge=False):
    return {
        "measured": True,
        "linear_ceiling": {"pearson": linear},
        "best_at_grid_edge": at_edge,
        "target_reliability": {
            "reliability": ceiling**2,
            "max_achievable_pearson": ceiling,
            "best_possible_mse_ratio": 1 - ceiling**2,
        },
    }, {"map_control_pearson": model_pearson}


def _verdict(linear, model_pearson, ceiling, at_edge=False):
    screen, model = _screen(linear, model_pearson, ceiling, at_edge)
    return mapping.verdict({"s": screen}, {"per_context": {"s": model}})


def test_a_linear_map_that_beats_the_model_points_at_the_model():
    decision, reading = _verdict(linear=0.35, model_pearson=0.06, ceiling=0.7)
    assert decision == "model-underperforms"
    assert "B3" in reading
    # The share of the ceiling each one reaches is the reading that matters.
    assert "9%" in reading and "50%" in reading


def test_an_unreliable_target_points_at_the_reframe():
    """Neither predictor works and the target barely correlates with itself:
    the quantity does not exist at this depth, so §B2 rather than §B3."""
    decision, reading = _verdict(linear=0.04, model_pearson=0.06, ceiling=0.05)
    assert decision == "not-predictable-per-cell"
    assert "B2" in reading
    assert "control resample" in reading


def test_a_reliable_target_nobody_predicts_is_a_capacity_question():
    decision, reading = _verdict(linear=0.07, model_pearson=0.06, ceiling=0.8)
    assert decision == "linear-is-no-better"
    assert "64 latents" in reading


def test_a_ceiling_chosen_at_the_edge_of_the_grid_says_so():
    """The best penalty being the largest offered is a statement about the
    grid, and the reading must not be quoted as the ceiling."""
    _, reading = _verdict(linear=0.07, model_pearson=0.06, ceiling=0.8, at_edge=True)
    assert "lower bound" in reading


def test_without_the_model_the_verdict_refuses_to_choose():
    screen, _ = _screen(0.35, 0.06, 0.7)
    decision, reading = mapping.verdict({"s": screen}, {"per_context": {}})
    assert decision == "no-model-to-compare"
    assert "phase2" in reading


def test_no_screen_with_enough_cells_is_not_a_verdict():
    decision, _ = mapping.verdict({"s": {"measured": False}}, {})
    assert decision == "not-measured"


# --------------------------------------------------------------------------- #
# end to end on mini_data
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_the_ceiling_runs_on_a_mini_screen(session_cfg, tmp_path):
    import dataclasses

    from vccp.data import replogle as replogle_data
    from vccp.paths import DataPaths

    cfg = dataclasses.replace(session_cfg, output_root=tmp_path / "runs")
    discovered = DataPaths(cfg).discover_phase2_contexts()
    assert discovered, "mini_data has no phase2 screens"

    name = sorted(discovered)[0]
    context = replogle_data.load_replogle_context(discovered[name])
    entry = mapping.measure_context(context, cfg, n_fit=128, n_eval=64)

    assert entry["measured"] is True
    assert entry["n_panel"] > 0 and entry["n_rest"] > 0
    assert set(entry["ridge"]) == {f"{p:g}" for p in mapping.PENALTIES}
    for scores in entry["ridge"].values():
        assert scores["mse_no_change"] > 0
        assert np.isfinite(scores["mse_ratio_to_no_change"])
    # The depth-only fit uses one predictor and must still produce a number,
    # because "the panel is a depth correction" is a live hypothesis.
    assert np.isfinite(entry["depth_only"]["pearson"])
    assert np.isfinite(entry["target_reliability"]["half_depth_pearson"])
    assert np.isfinite(entry["target_reliability"]["max_achievable_pearson"])
