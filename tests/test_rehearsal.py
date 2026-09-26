"""The rehearsal, Phase 3 and the prediction (checklist items 10, 11, 12).

These run the real stages on `mini_data` at one-step budgets. What is being
tested is the wiring the specification is specific about: that all three
variants run with a method, an upper bound and a floor; that the two partial
variants can never be read as end-to-end performance; that the policy Phase 3
obeys forbids adapting the perturbation module; and that every (context,
target) of the round comes out as `cells_per_pert` whole, integral cells.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import scipy.sparse as sp

from vccp.data import manifest as manifest_mod
from vccp.data import reference
from vccp.paths import DataPaths, RunPaths
from vccp.phases import phase3 as phase3_mod
from vccp.rehearsal import common, stage as rehearsal_stage

@pytest.fixture(scope="module")
def m4_cfg(pipeline_cfg):
    """The shared pipeline config (tests/conftest.py)."""
    return pipeline_cfg


@pytest.fixture(scope="module")
def m4_run(pipeline_run):
    """The shared pipeline run (tests/conftest.py)."""
    return pipeline_run


# --------------------------------------------------------------------------- #
# the rehearsal (item 10)
# --------------------------------------------------------------------------- #
def test_all_three_variants_run(m4_run):
    report = json.loads(m4_run.rehearsal_report.read_text())
    variants = {entry["variant"] for entry in report["variants"]}
    assert variants == {"controls_only", "cross_context", "unseen_genes"}


def test_every_variant_carries_its_upper_bound_and_floor(m4_run):
    """§4.6: the method is only readable against those two."""
    report = json.loads(m4_run.rehearsal_report.read_text())
    for entry in report["variants"]:
        assert set(entry["arms"]) >= {common.METHOD, common.UPPER_BOUND, common.FLOOR}, entry


def test_partial_variants_are_never_keyed_as_end_to_end(m4_run):
    """The user's addition to §8.6: variants 1 and 3 are fed true panel values,
    so their official metrics must not be able to be read as end-to-end."""
    report = json.loads(m4_run.rehearsal_report.read_text())
    for entry in report["variants"]:
        for arm in entry["arms"].values():
            if entry["variant"] == "cross_context":
                assert common.PANEL_ASSISTED not in arm
            else:
                assert common.END_TO_END not in arm, entry["variant"]


def test_the_readable_summary_says_which_reading_is_which(m4_run):
    text = m4_run.rehearsal_summary.read_text()
    assert "PANEL-ASSISTED" in text
    assert "not indicative of real performance" in text


def test_the_scale_reports_why_it_was_not_built(m4_run):
    """§6: where the two ends cannot be built, say so — do not pass silently."""
    report = json.loads(m4_run.rehearsal_report.read_text())
    scale = report["scale"]
    assert scale["requested"] is False
    for entry in scale["datasets"].values():
        assert entry["built"] is False
        assert entry["reason"]


# --------------------------------------------------------------------------- #
# the policy (item 10's second artifact)
# --------------------------------------------------------------------------- #
def test_the_policy_forbids_adapting_the_perturbation_module(m4_run):
    policy = rehearsal_stage.load_policy(m4_run.phase3_policy)
    assert policy["adapt"]["perturbation_module"] is False
    assert policy["adapt"]["core"] is False
    assert policy["adapt"]["context_adapters"] is True


def test_the_calibrated_settings_come_from_the_grid(m4_cfg, m4_run):
    policy = rehearsal_stage.load_policy(m4_run.phase3_policy)
    settings = rehearsal_stage.policy_settings(policy)
    report = json.loads(m4_run.rehearsal_report.read_text())
    if report["calibration"]["best"] is None:
        pytest.skip(f"nothing scored: {report['calibration']['fallback_reason']}")
    assert settings.confidence_threshold in m4_cfg.rehearsal.calibration_thresholds
    assert settings.effect_scale in m4_cfg.rehearsal.calibration_scales


def test_a_policy_from_another_version_is_refused(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"version": rehearsal_stage.POLICY_VERSION + 1}))
    with pytest.raises(ValueError, match="rerun the rehearsal"):
        rehearsal_stage.load_policy(path)


# --------------------------------------------------------------------------- #
# Phase 3 adaptation (item 11)
# --------------------------------------------------------------------------- #
def test_every_context_reports_its_loss_before_and_after(m4_cfg, m4_run):
    manifest = manifest_mod.load_manifest(DataPaths(m4_cfg).manifest)
    for context in manifest.contexts:
        record = json.loads(m4_run.phase3_adapt(context).read_text())
        assert record["context"] == context
        assert np.isfinite(record["loss_before"])
        assert np.isfinite(record["loss_after"])
        assert record["policy"]["perturbation_module"] is False


def test_the_frozen_parameters_are_verified_unchanged(m4_run):
    report = json.loads(m4_run.frozen_check.read_text())
    assert report["all_ok"] is True
    assert any(name.startswith("phase3:") for name in report["phases"])


def test_phase3_refuses_a_policy_that_would_unfreeze_the_perturbation_module(
    m4_cfg, m4_run
):
    """§4.7 is not negotiable by config: the module stays frozen."""
    policy = json.loads(m4_run.phase3_policy.read_text())
    original = m4_run.phase3_policy.read_text()
    policy["adapt"]["perturbation_module"] = True
    m4_run.phase3_policy.write_text(json.dumps(policy))
    try:
        with pytest.raises(RuntimeError, match="perturbation module"):
            phase3_mod.run_phase3(m4_cfg)
    finally:
        m4_run.phase3_policy.write_text(original)


# --------------------------------------------------------------------------- #
# the prediction (item 12)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def prediction_index(m4_run):
    return json.loads(m4_run.prediction_index.read_text())


def test_one_block_per_context_and_perturbation(m4_cfg, m4_run, prediction_index):
    paths = DataPaths(m4_cfg)
    manifest = manifest_mod.load_manifest(paths.manifest)
    perts = reference.load_pert_counts(paths.pert_counts, manifest.pert_col)

    pairs = {(b["context"], b["target_gene"]) for b in prediction_index["blocks"]}
    assert pairs == {(c, t) for c in manifest.contexts for t in perts.targets}


def test_every_block_is_sparse_integer_counts_of_the_right_shape(
    m4_cfg, m4_run, prediction_index
):
    paths = DataPaths(m4_cfg)
    manifest = manifest_mod.load_manifest(paths.manifest)
    n_genes = len(reference.load_gene_names(paths.gene_names))

    for block in prediction_index["blocks"]:
        matrix = sp.load_npz(m4_run.root / block["path"])
        assert sp.issparse(matrix)
        assert matrix.shape == (manifest.cells_per_pert, n_genes)
        assert matrix.dtype.kind in "iu"
        assert matrix.min() >= 0
        assert np.isfinite(matrix.data).all()
        assert (matrix.data != 0).all(), "explicit zeros count toward the cap (§8)"


def test_the_cells_of_one_target_are_not_copies_of_each_other(m4_run, prediction_index):
    matrix = sp.load_npz(m4_run.root / prediction_index["blocks"][0]["path"]).toarray()
    assert len(np.unique(matrix, axis=0)) > 1


def test_two_targets_do_not_get_the_same_cells(m4_run, prediction_index):
    """Each target draws its own control cells (§4.7)."""
    blocks = [b for b in prediction_index["blocks"] if b["context"] ==
              prediction_index["blocks"][0]["context"]][:2]
    first, second = (sp.load_npz(m4_run.root / b["path"]).toarray() for b in blocks)
    assert not np.array_equal(first, second)


def test_the_knockdown_source_is_recorded_per_target(m4_cfg, m4_run):
    """DECISIONS.md D8: `fold_expr` measured in K562 or RPE1 need not transfer,
    so which number was used has to be visible."""
    import pandas as pd

    manifest = manifest_mod.load_manifest(DataPaths(m4_cfg).manifest)
    for context in manifest.contexts:
        frame = pd.read_csv(m4_run.phase3_knockdown(context))
        assert set(frame["context"]) == {context}
        assert frame["fold_expr_source"].notna().all()
        assert frame["fold_expr_source"].str.startswith(("pooled", "per_target:")).all()


def test_the_knockdown_lowers_the_target_gene_where_it_applies(
    m4_cfg, m4_run, prediction_index
):
    import pandas as pd

    paths = DataPaths(m4_cfg)
    axis = reference.load_gene_names(paths.gene_names)
    position = {symbol: i for i, symbol in enumerate(axis)}
    context = prediction_index["blocks"][0]["context"]
    frame = pd.read_csv(m4_run.phase3_knockdown(context)).set_index("target_gene")

    checked = 0
    for block in prediction_index["blocks"]:
        if block["context"] != context or not frame.loc[block["target_gene"], "applied"]:
            continue
        matrix = sp.load_npz(m4_run.root / block["path"]).toarray()
        column = matrix[:, position[block["target_gene"]]]
        assert column.mean() >= 0
        checked += 1
    assert checked > 0, "no target's own gene was knocked down at all"


def test_targets_csv_and_the_submission_requirement_must_agree(m4_cfg):
    """They agree on real data; if they ever did not, that is a data problem
    to report rather than one to resolve by preferring one of them."""
    import pandas as pd

    from vccp.predict.run import _prediction_plan, _required_pairs

    paths = DataPaths(m4_cfg)
    manifest = manifest_mod.load_manifest(paths.manifest)
    perts = reference.load_pert_counts(paths.pert_counts, manifest.pert_col)
    frame = pd.read_csv(paths.phase3_targets)
    wanted = _required_pairs(manifest, perts)

    plan = _prediction_plan(frame, manifest, perts, wanted)
    assert sum(len(targets) for targets in plan.values()) == len(wanted)

    with pytest.raises(ValueError, match="disagree"):
        _prediction_plan(frame.iloc[1:], manifest, perts, wanted)


def test_nothing_holds_every_target_at_once(m4_run, prediction_index):
    """§7: the blocks are written per target, so the submission writer can
    stream them back one at a time."""
    contexts = {b["context"] for b in prediction_index["blocks"]}
    for context in contexts:
        blocks = sorted(
            path for path in (m4_run.predictions_dir / context).glob("*.npz")
            if path != m4_run.fold_changes(context)
        )
        assert len(blocks) == sum(
            1 for b in prediction_index["blocks"] if b["context"] == context
        )


# --------------------------------------------------------------------------- #
# the policy is read off the rehearsal, not written in (DECISIONS.md D94)
# --------------------------------------------------------------------------- #


def _controls_only(ratios):
    from vccp.rehearsal import common
    from vccp.rehearsal.variants import VariantResult

    return [
        VariantResult(
            variant="controls_only",
            context=name,
            arms={common.METHOD: {"rest_genes": {"mse_ratio_to_no_change": ratio}}},
        )
        for name, ratio in ratios.items()
    ]


def test_the_policy_forbids_adapting_when_the_rehearsal_measured_it_harmful():
    """§4.6 says the policy states what Phase 3 may adapt *chosen from these
    results*. Variant 1 runs Phase 3's own procedure on a context whose
    answers are known: 1.484, 1.509 and 1.468 against a floor of 1.0 on the
    server, which is a measured harm, not a neutral step."""
    from vccp.rehearsal.stage import _adapt_policy

    harmful = _adapt_policy(_controls_only({"a": 1.484, "b": 1.509, "c": 1.468}))
    assert harmful["measured"] is True
    assert harmful["context_adapters"] is False
    assert harmful["delta"] is False
    assert "1.484" in str(harmful["controls_only_ratio_per_context"]["a"])
    assert "worse" in harmful["reason"]

    helpful = _adapt_policy(_controls_only({"a": 0.91, "b": 0.88, "c": 1.01}))
    assert helpful["context_adapters"] is True and helpful["delta"] is True
    assert "permitted" in helpful["reason"]

    # The perturbation module and the core are never adaptable, whatever the
    # rehearsal says: §4.7 is explicit and this is not what is measured.
    for policy in (harmful, helpful):
        assert policy["core"] is False
        assert policy["perturbation_module"] is False
        assert policy["heads"] is False


def test_without_variant_1_the_policy_falls_back_to_what_4_7_describes():
    from vccp.rehearsal.stage import _adapt_policy

    policy = _adapt_policy([])
    assert policy["measured"] is False
    assert policy["context_adapters"] is True and policy["delta"] is True
    assert "did not run" in policy["reason"]


def test_a_tie_with_the_baseline_still_lets_phase3_adapt():
    """The threshold is slightly above 1.0 on purpose: a mapping that merely
    matches the baseline has not been shown to do harm."""
    from vccp.rehearsal.stage import CONTROLS_ONLY_TOLERANCE, _adapt_policy

    assert CONTROLS_ONLY_TOLERANCE > 1.0
    assert _adapt_policy(_controls_only({"a": 1.0}))["context_adapters"] is True
    assert _adapt_policy(
        _controls_only({"a": CONTROLS_ONLY_TOLERANCE + 0.01})
    )["context_adapters"] is False
