"""The cell generator, the knockdown prior and the no-change scale.

The generator is part of the method, not packaging (CLAUDE.md §4.7): the
official metrics run DE on the submitted cells, so what matters here is that
an unchanged gene is *bit-for-bit* the control's, that changes stay integral
and non-negative, and that the two calibration knobs mean what the policy
says they mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from vccp.eval import nochange
from vccp.predict import knockdown
from vccp.predict.generator import (
    GeneratorSettings,
    apply_settings,
    describe,
    generate_cells,
    generate_counts,
    get_generator,
    sample_control_cells,
    stochastic_round,
)


@pytest.fixture
def controls():
    rng = np.random.default_rng(0)
    return rng.poisson(20.0, size=(64, 32)).astype(np.int64)


# --------------------------------------------------------------------------- #
# the two calibration settings
# --------------------------------------------------------------------------- #
def test_threshold_is_applied_before_the_scale():
    """Otherwise the threshold would mean something different at every scale."""
    values = np.array([0.4, -0.4, 0.04], dtype=np.float32)
    settings = GeneratorSettings(confidence_threshold=0.1, effect_scale=0.5)

    applied = apply_settings(values, settings)

    assert applied[2] == 0.0, "below the threshold, so silenced"
    assert applied[0] == pytest.approx(0.2)
    # Had the scale come first, 0.4 * 0.5 = 0.2 would still clear 0.1 — the
    # case that distinguishes the two orders is one that would not.
    shrunk = apply_settings(values, GeneratorSettings(confidence_threshold=0.3, effect_scale=0.5))
    assert shrunk[0] == pytest.approx(0.2), "thresholded on the model's own output"


def test_non_finite_predictions_become_no_change():
    values = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    applied = apply_settings(values, GeneratorSettings(confidence_threshold=0.0))
    assert list(applied[:3]) == [0.0, 0.0, 0.0]
    assert applied[3] == pytest.approx(1.0)


def test_an_unconfident_gene_is_bit_for_bit_the_control(controls):
    """§4.7: a gene with no confident change must be exchangeable with the
    control, or the DE test has a reason to call it."""
    log2fc = np.zeros(controls.shape[1], dtype=np.float32)
    log2fc[0] = -2.0
    settings = GeneratorSettings(confidence_threshold=0.05)

    out = generate_counts(controls, apply_settings(log2fc, settings), np.random.default_rng(1))

    np.testing.assert_array_equal(out[:, 1:], controls[:, 1:])
    assert out[:, 0].sum() < controls[:, 0].sum()


# --------------------------------------------------------------------------- #
# count space
# --------------------------------------------------------------------------- #
def test_counts_stay_non_negative_whole_numbers(controls):
    rng = np.random.default_rng(2)
    log2fc = rng.normal(0.0, 1.5, size=controls.shape[1]).astype(np.float32)

    out = generate_counts(controls, log2fc, rng)

    assert out.dtype.kind in "iu"
    assert (out >= 0).all()
    assert np.isfinite(out).all()


def test_thinning_never_increases_a_decreasing_gene(controls):
    log2fc = np.full(controls.shape[1], -1.0, dtype=np.float32)
    out = generate_counts(controls, log2fc, np.random.default_rng(3))
    assert (out <= controls).all(), "binomial thinning only ever keeps counts"


def test_stochastic_rounding_is_unbiased():
    rng = np.random.default_rng(4)
    values = np.full((20000,), 2.3)
    rounded = stochastic_round(values, rng)
    assert set(np.unique(rounded)) <= {2, 3}
    assert rounded.mean() == pytest.approx(2.3, abs=0.02)


def test_cells_vary_rather_than_being_copies_of_one_average(controls):
    """§6.1 check 5: a group of identical cells has no dispersion for the DE
    test and is punished by the expression metric."""
    log2fc = np.full(controls.shape[1], -1.0, dtype=np.float32)
    out = generate_cells(controls, log2fc, 40, np.random.default_rng(5), GeneratorSettings())
    assert out.shape == (40, controls.shape[1])
    assert len(np.unique(out, axis=0)) > 1
    assert out.var(axis=0).mean() > 0


def test_a_mismatched_gene_axis_is_refused(controls):
    with pytest.raises(ValueError, match="fold changes for"):
        generate_counts(controls, np.zeros(controls.shape[1] + 1, dtype=np.float32),
                        np.random.default_rng(6))


# --------------------------------------------------------------------------- #
# drawing the control cells
# --------------------------------------------------------------------------- #
def test_a_fresh_draw_per_target_is_distinct():
    rows_a = sample_control_cells(100, 20, np.random.default_rng(1))
    rows_b = sample_control_cells(100, 20, np.random.default_rng(2))
    assert not np.array_equal(np.sort(rows_a), np.sort(rows_b))
    assert len(set(rows_a.tolist())) == 20, "without replacement while it can be"


def test_replacement_is_used_only_when_there_are_too_few_cells():
    rows = sample_control_cells(10, 25, np.random.default_rng(1))
    assert rows.size == 25
    with pytest.raises(ValueError, match="allow_replacement"):
        sample_control_cells(10, 25, np.random.default_rng(1), allow_replacement=False)


def test_the_arms_of_a_variant_share_their_control_draw():
    """DECISIONS.md D7: the arms must differ only in their fold changes."""
    from vccp.rehearsal import common

    settings = GeneratorSettings()
    first = common.control_rows_for("cross_context", "ADNP", 0, 100, 20, settings)
    again = common.control_rows_for("cross_context", "ADNP", 0, 100, 20, settings)
    other_target = common.control_rows_for("cross_context", "TDRD7", 0, 100, 20, settings)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(np.sort(first), np.sort(other_target))


def test_the_generator_registry_names_the_configured_one():
    assert get_generator("count_space") is generate_cells
    with pytest.raises(KeyError, match="unknown generator"):
        get_generator("no_such_generator")


def test_describe_reports_what_the_settings_did():
    log2fc = np.array([0.0, 0.02, 1.0, -2.0], dtype=np.float32)
    report = describe(GeneratorSettings(confidence_threshold=0.1, effect_scale=0.5), log2fc)
    assert report["n_genes"] == 4
    assert report["n_genes_moved"] == 2
    assert report["max_abs_log2fc"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# the knockdown prior
# --------------------------------------------------------------------------- #
def test_the_resolved_source_is_recorded_per_target():
    """DECISIONS.md D8: the number used is a transfer assumption and has to be
    visible as one."""
    prior = knockdown.KnockdownPrior(
        per_target={"ADNP": 0.2}, source_screen={"ADNP": "K562_gwps"},
        pooled_median=0.155, screens={"K562_gwps": 0.155},
    )
    value, source = prior.resolve("ADNP")
    assert value == pytest.approx(0.2)
    assert source == "per_target:K562_gwps"

    value, source = prior.resolve("NOT_MEASURED")
    assert value == pytest.approx(0.155)
    assert source == knockdown.POOLED


def test_the_prior_lowers_the_target_gene():
    log2fc = np.zeros(5, dtype=np.float32)
    out, record = knockdown.apply_to_log2fc(log2fc, 2, 0.25, 50.0, 1.0)
    assert record["applied"] is True
    assert out[2] == pytest.approx(-2.0)
    assert list(out[[0, 1, 3, 4]]) == [0.0, 0.0, 0.0, 0.0]


def test_the_prior_is_not_applied_to_an_undetectable_gene():
    """Applying a fold change to a gene the controls do not express would
    manufacture a change out of noise (DECISIONS.md D8)."""
    out, record = knockdown.apply_to_log2fc(np.zeros(3, dtype=np.float32), 1, 0.25, 0.01, 1.0)
    assert record["applied"] is False
    assert "not detectably expressed" in record["reason"]
    assert out[1] == 0.0


def test_the_prior_skips_a_target_that_is_not_a_challenge_gene():
    out, record = knockdown.apply_to_log2fc(np.zeros(3, dtype=np.float32), None, 0.25, 50.0, 1.0)
    assert record["applied"] is False
    assert list(out) == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# the common scale the calibration maximizes
# --------------------------------------------------------------------------- #
def test_no_change_scores_zero_on_every_metric():
    values = {metric: value for metric, (value, _) in nochange.NO_CHANGE.items()}
    assert nochange.objective(values) == pytest.approx(0.0)
    assert not any(nochange.beats_no_change(values).values())


def test_the_scale_points_the_same_way_for_both_directions():
    """Two of the six are lower-is-better; their raw mean is not a quantity
    worth maximizing (DECISIONS.md D5)."""
    better = nochange.normalize("expr_mse_unbiased_capped_norm", 0.5)
    worse = nochange.normalize("expr_mse_unbiased_capped_norm", 1.5)
    assert better.value > 0 > worse.value

    better = nochange.normalize("pds_cosine", 0.75)
    worse = nochange.normalize("pds_cosine", 0.25)
    assert better.value > 0 > worse.value


def test_a_metric_that_is_not_one_of_the_six_is_ignored():
    assert nochange.normalize("de_wilcoxon_nsig_counts_pred", 42.0) is None
    assert np.isnan(nochange.objective({"not_a_metric": 1.0}))


def test_a_measured_knockdown_is_exempt_from_the_two_settings():
    """§4.7's settings are calibrated against the model's own uncertainty; a
    measured `fold_expr` was never in doubt, and shrinking it would undo the
    knockdown sanity check 3 looks for (DECISIONS.md D38)."""
    values = np.array([-0.234, 1.0, 0.02], dtype=np.float32)  # a weak guide, 0.85x
    exempt = np.array([True, False, False])
    settings = GeneratorSettings(confidence_threshold=0.3, effect_scale=0.5)

    applied = apply_settings(values, settings, exempt)

    assert applied[0] == pytest.approx(-0.234, abs=1e-6), "measured, so untouched"
    assert applied[1] == pytest.approx(0.5), "predicted, so scaled"
    assert applied[2] == 0.0, "predicted and below the threshold, so silenced"

    without = apply_settings(values, settings)
    assert without[0] == 0.0, "without the exemption the threshold would silence it"
