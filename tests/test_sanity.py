"""The seven sanity checks, each fed a deliberately broken prediction (§6.1).

The broken predictions are built from a synthetic control matrix and written
as real `.npz` blocks, so each test goes through the same measurement path
the pipeline uses rather than through hand-written statistics.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from vccp.paths import RunPaths
from vccp.sanity import checks as check_mod
from vccp.sanity import stage as sanity_stage
from vccp.submit import validator
from vccp.submit.spec import SubmissionSpec

GENES = tuple(f"G{i}" for i in range(12))
TARGETS = ("G0", "G1", "G2", "G3")
CONTEXT = "A"
CELLS = 12
CONTROL_CELLS = 40


@pytest.fixture
def spec():
    return SubmissionSpec(
        axis=GENES,
        contexts=(CONTEXT,),
        targets=TARGETS,
        cells_per_target={t: CELLS for t in TARGETS},
        pert_col="target_gene",
        context_col="context",
        control_label="non-targeting",
    )


@pytest.fixture
def controls():
    """Controls with a wide range of expression, so a knockdown is visible."""
    rng = np.random.default_rng(0)
    rates = rng.uniform(20, 200, size=len(GENES))
    return rng.poisson(rates, size=(CONTROL_CELLS, len(GENES))).astype(np.int64)


def write_prediction(run_paths, spec, controls, make_cells, make_changes, rng_seed=1):
    """Write one context's blocks and fold changes, as `predict` would."""
    rng = np.random.default_rng(rng_seed)
    changes = []
    for i, target in enumerate(spec.targets):
        cells = make_cells(target, i, controls, rng)
        path = run_paths.prediction_block(CONTEXT, target)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = sp.csr_matrix(np.asarray(cells, dtype=np.int32))
        matrix.eliminate_zeros()
        sp.save_npz(path, matrix)
        changes.append(np.asarray(make_changes(target, i), dtype=np.float32))
    np.savez_compressed(
        run_paths.fold_changes(CONTEXT),
        targets=np.array(list(spec.targets), dtype=object),
        log2_fold_change=np.vstack(changes),
    )


def measure(cfg, spec, run_paths, controls):
    """Run the stage's own measurement over the written blocks."""
    context = SimpleNamespace(
        name=CONTEXT, controls=SimpleNamespace(counts=lambda rows=None: controls)
    )
    audit = validator.MatrixAudit()
    pairs: Counter = Counter()
    stats, records = sanity_stage.measure_context(
        cfg, spec, run_paths, context,
        {symbol: i for i, symbol in enumerate(spec.axis)}, audit, pairs,
    )
    return stats.as_dict(), records, audit, pairs


@pytest.fixture
def run_paths(mini_cfg, tmp_path):
    cfg = dataclasses.replace(mini_cfg, output_root=tmp_path / "runs")
    paths = RunPaths(cfg)
    paths.ensure()
    return paths


def knockdown_cells(target, index, controls, rng):
    """A believable prediction: the target's own gene down, a little noise."""
    rows = rng.choice(controls.shape[0], CELLS, replace=False)
    cells = controls[rows].copy()
    position = GENES.index(target)
    cells[:, position] = rng.binomial(cells[:, position], 0.2)
    moved = (position + 1 + index) % len(GENES)
    cells[:, moved] = rng.binomial(cells[:, moved], 0.5)
    return cells


def _one_moved_gene():
    change = np.zeros(len(GENES), dtype=np.float32)
    change[len(TARGETS) + 1] = -1.0
    return change


def knockdown_changes(target, index):
    change = np.zeros(len(GENES), dtype=np.float32)
    change[GENES.index(target)] = -2.3
    change[(GENES.index(target) + 1 + index) % len(GENES)] = -1.0
    return change


# --------------------------------------------------------------------------- #
# the baseline: a reasonable prediction passes
# --------------------------------------------------------------------------- #
def test_a_reasonable_prediction_passes_checks_1_to_5(mini_cfg, spec, controls, run_paths):
    write_prediction(run_paths, spec, controls, knockdown_cells, knockdown_changes)
    stats, records, audit, pairs = measure(mini_cfg, spec, run_paths, controls)

    result = validator.result_from_audit(mini_cfg, spec, audit, pairs, source="t")
    assert result.ok, result.summary()
    assert check_mod.check_magnitudes(mini_cfg, {CONTEXT: stats}).status == check_mod.PASS
    assert check_mod.check_knockdown(mini_cfg, records).status == check_mod.PASS
    assert check_mod.check_targets_differ(mini_cfg, {CONTEXT: stats}).status == check_mod.PASS
    assert check_mod.check_cells_vary(mini_cfg, {CONTEXT: stats}).status == check_mod.PASS


# --------------------------------------------------------------------------- #
# check 1 — complete and valid
# --------------------------------------------------------------------------- #
def test_check_1_catches_a_wrong_cell_count(mini_cfg, spec, controls, run_paths):
    def short(target, index, controls, rng):
        return knockdown_cells(target, index, controls, rng)[: CELLS - 1]

    write_prediction(run_paths, spec, controls, short, knockdown_changes)
    _, _, audit, pairs = measure(mini_cfg, spec, run_paths, controls)
    result = validator.result_from_audit(mini_cfg, spec, audit, pairs, source="t")
    assert not result.ok
    assert "cells_per_perturbation" in {r.rule for r in result.failures}


def test_check_1_is_the_only_hard_failure(mini_cfg):
    assert check_mod.verdict(False, hard=True) == check_mod.FAIL
    assert check_mod.verdict(False, hard=False) == check_mod.WARN


# --------------------------------------------------------------------------- #
# check 2 — plausible magnitudes
# --------------------------------------------------------------------------- #
def test_check_2_catches_an_implausible_library_size(mini_cfg, spec, controls, run_paths):
    def inflated(target, index, controls, rng):
        return knockdown_cells(target, index, controls, rng) * 5

    write_prediction(run_paths, spec, controls, inflated, knockdown_changes)
    stats, _, _, _ = measure(mini_cfg, spec, run_paths, controls)
    result = check_mod.check_magnitudes(mini_cfg, {CONTEXT: stats})
    assert result.status == check_mod.WARN
    assert stats["median_library_ratio"] > mini_cfg.sanity.library_ratio_high


def test_check_2_bounds_the_genes_the_model_did_move(mini_cfg, spec, controls, run_paths):
    """The user's addition at the M2 review: the moved genes are bounded too."""
    tight = dataclasses.replace(
        mini_cfg, sanity=dataclasses.replace(mini_cfg.sanity, max_abs_log2fc=0.5)
    )
    write_prediction(run_paths, spec, controls, knockdown_cells, knockdown_changes)
    stats, _, _, _ = measure(tight, spec, run_paths, controls)
    result = check_mod.check_magnitudes(tight, {CONTEXT: stats})
    assert result.status == check_mod.WARN
    assert stats["moved_over_max_abs_log2fc"] > 0
    assert stats["worst_moved"]


def test_check_2_catches_a_moved_gene_the_model_never_declared(
    mini_cfg, spec, controls, run_paths
):
    """A gene the generator was told to leave alone must come back unchanged;
    if it moved anyway, the first half of check 2 sees it."""
    def meddled(target, index, controls, rng):
        cells = knockdown_cells(target, index, controls, rng)
        cells[:, -1] = rng.binomial(cells[:, -1], 0.1)  # not in the fold changes
        return cells

    write_prediction(run_paths, spec, controls, meddled, knockdown_changes)
    stats, _, _, _ = measure(mini_cfg, spec, run_paths, controls)
    assert stats["unmoved_outside_control_range"] > 0
    assert check_mod.check_magnitudes(mini_cfg, {CONTEXT: stats}).status == check_mod.WARN


# --------------------------------------------------------------------------- #
# check 3 — the knockdown is visible
# --------------------------------------------------------------------------- #
def test_check_3_catches_a_target_gene_that_was_not_reduced(
    mini_cfg, spec, controls, run_paths
):
    def untouched(target, index, controls, rng):
        rows = rng.choice(controls.shape[0], CELLS, replace=False)
        return controls[rows].copy()

    write_prediction(run_paths, spec, controls, untouched, knockdown_changes)
    _, records, _, _ = measure(mini_cfg, spec, run_paths, controls)
    result = check_mod.check_knockdown(mini_cfg, records)
    assert result.status == check_mod.WARN
    assert result.value < mini_cfg.sanity.knockdown_min_fraction
    assert result.detail["failing"]


def test_check_3_skips_a_gene_the_controls_do_not_express(mini_cfg):
    records = [
        {"context": "A", "target_gene": "X", "control_mean_cpm": 0.01,
         "predicted_mean_cpm": 0.01, "ratio": 1.0, "eligible": False}
    ]
    result = check_mod.check_knockdown(mini_cfg, records)
    assert result.status == check_mod.WARN
    assert result.detail["n_eligible"] == 0


# --------------------------------------------------------------------------- #
# check 4 — targets differ
# --------------------------------------------------------------------------- #
def test_check_4_catches_identical_predictions(mini_cfg, spec, controls, run_paths):
    def same_for_every_target(target, index, controls, rng):
        # One block of cells reused for every target — exactly what §4.7
        # forbids, and what check 4 must see.
        fixed = np.random.default_rng(7).choice(controls.shape[0], CELLS, replace=False)
        cells = controls[fixed].copy()
        cells[:, len(TARGETS) + 1] = cells[:, len(TARGETS) + 1] // 2
        return cells

    write_prediction(
        run_paths, spec, controls, same_for_every_target,
        lambda target, index: _one_moved_gene(),
    )
    stats, _, _, _ = measure(mini_cfg, spec, run_paths, controls)
    result = check_mod.check_targets_differ(mini_cfg, {CONTEXT: stats})
    assert result.status == check_mod.WARN
    assert stats["identical_target_pairs"]


def test_check_4_catches_targets_that_are_merely_too_alike(mini_cfg):
    rng = np.random.default_rng(0)
    shared = rng.normal(size=200)
    matrix = np.stack([shared + 0.01 * rng.normal(size=200) for _ in range(5)])
    assert check_mod.median_pairwise_correlation(matrix) > 0.95


# --------------------------------------------------------------------------- #
# check 5 — cells vary
# --------------------------------------------------------------------------- #
def test_check_5_catches_identical_cells(mini_cfg, spec, controls, run_paths):
    def one_cell_repeated(target, index, controls, rng):
        row = controls[index % controls.shape[0]]
        return np.tile(row, (CELLS, 1))

    write_prediction(run_paths, spec, controls, one_cell_repeated, knockdown_changes)
    stats, _, _, _ = measure(mini_cfg, spec, run_paths, controls)
    result = check_mod.check_cells_vary(mini_cfg, {CONTEXT: stats})
    assert result.status == check_mod.WARN
    assert min(stats["variance_ratios"].values()) < mini_cfg.sanity.variance_ratio_low


# --------------------------------------------------------------------------- #
# checks 6 and 7 — the rehearsal
# --------------------------------------------------------------------------- #
def rehearsal_report(scored, nsig=None):
    from vccp.rehearsal import common

    arm = {"scored": scored, "normalized": {"objective": 0.0}}
    if nsig is not None:
        arm["nsig_counts"] = nsig
    return {
        "variants": [
            {
                "variant": "cross_context",
                "context": "rpe1",
                "n_targets": 4,
                "arms": {common.METHOD: {common.END_TO_END: arm}},
            }
        ]
    }


def test_check_6_catches_a_prediction_worse_than_no_change(mini_cfg):
    worse = rehearsal_report(
        {"pds_cosine": 0.4, "expr_mse_unbiased_capped_norm": 1.4,
         "de_wilcoxon_lfc_nmae": 1.2}
    )
    assert check_mod.check_beats_no_change(mini_cfg, worse).status == check_mod.WARN

    better = rehearsal_report(
        {"pds_cosine": 0.7, "expr_mse_unbiased_capped_norm": 0.6,
         "de_wilcoxon_lfc_nmae": 0.8}
    )
    assert check_mod.check_beats_no_change(mini_cfg, better).status == check_mod.PASS


def test_check_6_never_reads_a_panel_assisted_arm(mini_cfg):
    """Variants 1 and 3 are fed the true panel; they answer a different
    question and must not stand in for the end-to-end one."""
    from vccp.rehearsal import common

    report = {
        "variants": [
            {
                "variant": "controls_only", "context": "rpe1", "n_targets": 4,
                "arms": {common.METHOD: {common.PANEL_ASSISTED: {
                    "scored": {"pds_cosine": 1.0,
                               "expr_mse_unbiased_capped_norm": 0.1,
                               "de_wilcoxon_lfc_nmae": 0.1}}}},
            }
        ]
    }
    result = check_mod.check_beats_no_change(mini_cfg, report)
    assert result.status == check_mod.WARN
    assert result.value is None


def test_check_7_catches_a_prediction_that_calls_nothing(mini_cfg):
    report = rehearsal_report(
        {"pds_cosine": 0.6, "expr_mse_unbiased_capped_norm": 0.9,
         "de_wilcoxon_lfc_nmae": 0.9},
        nsig={t: {"real": 100.0, "pred": 0.0} for t in TARGETS},
    )
    result = check_mod.check_de_call_count(mini_cfg, report)
    assert result.status == check_mod.WARN
    assert result.value == 0.0


def test_check_7_catches_a_prediction_that_floods_the_de_test(mini_cfg):
    report = rehearsal_report(
        {"pds_cosine": 0.6, "expr_mse_unbiased_capped_norm": 0.9,
         "de_wilcoxon_lfc_nmae": 0.9},
        nsig={t: {"real": 10.0, "pred": 900.0} for t in TARGETS},
    )
    result = check_mod.check_de_call_count(mini_cfg, report)
    assert result.status == check_mod.WARN
    assert result.value > mini_cfg.sanity.nsig_ratio_high


def test_check_7_passes_a_sensible_number_of_calls(mini_cfg):
    report = rehearsal_report(
        {"pds_cosine": 0.6, "expr_mse_unbiased_capped_norm": 0.9,
         "de_wilcoxon_lfc_nmae": 0.9},
        nsig={t: {"real": 100.0, "pred": 80.0} for t in TARGETS},
    )
    assert check_mod.check_de_call_count(mini_cfg, report).status == check_mod.PASS


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def test_every_threshold_is_reported_with_the_value(mini_cfg):
    results = [
        check_mod.CheckResult(1, "a", check_mod.PASS, "ok", value=1, threshold=0),
        check_mod.CheckResult(2, "b", check_mod.WARN, "hm", value=2, threshold=1),
    ]
    report = check_mod.report_dict(mini_cfg, results, on_mini=True)
    assert report["n_pass"] == 1 and report["n_warn"] == 1 and report["ok"] is True
    assert report["thresholds"]["max_abs_log2fc"] == mini_cfg.sanity.max_abs_log2fc
    assert "not indicative" in check_mod.summarize(results, on_mini=True)
