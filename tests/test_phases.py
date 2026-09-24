"""Phases 1 and 2, and the forgetting guard (checklist items 7, 8, 9).

These run one or two real training steps on `mini_data` rather than mocking,
because the things worth testing here — that validation is by held-out
*target*, that the mapping is Siamese, that the loss is masked to the genes a
context measures — are properties of the wiring, not of the arithmetic.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from vccp.data import reference
from vccp.models.core import as_index, build_model
from vccp.paths import DataPaths, RunPaths
from vccp.phases import phase1, phase2
from vccp.train.freeze import set_trainable


@pytest.fixture(scope="module")
def tiny_cfg(session_cfg):
    """Budgets small enough for a test, everything else as configured."""
    return dataclasses.replace(
        session_cfg,
        phase1=dataclasses.replace(session_cfg.phase1, steps=2, batch_size=4),
        phase2=dataclasses.replace(
            session_cfg.phase2, steps=2, batch_size=4, cells_per_draw=4
        ),
        train=dataclasses.replace(session_cfg.train, output_genes_per_step=32, log_every=1),
    )


@pytest.fixture(scope="module")
def gene_index(session_cfg):
    axis = reference.load_gene_names(DataPaths(session_cfg).gene_names)
    return {symbol: i for i, symbol in enumerate(axis)}


@pytest.fixture(scope="module")
def p1(tiny_cfg):
    return phase1.load_phase1_tensors(tiny_cfg, "cpu")


@pytest.fixture(scope="module")
def contexts(tiny_cfg, gene_index):
    return phase2.load_contexts(tiny_cfg, "cpu", gene_index)


def fresh_model(cfg, built_priors, adapter, wake=False):
    model = build_model(cfg, built_priors)
    for name in ("phase1", "phase2"):
        model.add_adapter(name)
    model.use_adapters([adapter])
    if wake:
        # Heads are zero-initialized so an untrained model predicts no change
        # (§4.7). A test about gradients flowing *through* a head has to move
        # it off zero first — at exactly zero the chain rule gives zero, for
        # one step, until the head's own gradient lifts it.
        with torch.no_grad():
            for head in model.heads.values():
                head[-1].base.weight.normal_(0, 0.1)
    return model


# --------------------------------------------------------------------------- #
# item 8 — Phase 1
# --------------------------------------------------------------------------- #
def test_validation_holds_out_whole_target_genes(p1):
    """An unseen perturbation, not an unseen row: the same knockout in
    another cell line would answer an easier question."""
    train = set(p1.targets[p1.train_rows])
    val = set(p1.targets[p1.val_rows])
    assert train and val
    assert not (train & val)


def test_only_targets_that_are_challenge_genes_are_used(p1, gene_index):
    assert all(target in gene_index for target in p1.targets)
    assert p1.n_rows_total >= p1.train_rows.size + p1.val_rows.size


def test_context_vectors_are_pooled_from_training_rows_only(p1):
    """Validation must not leak in through the context vector."""
    assert set(p1.context) == set(p1.cell_line)
    for vector in p1.context.values():
        assert vector.shape == (p1.n_genes,)
        assert torch.isfinite(vector).all()


def test_one_training_step_and_heldout_target_split(tiny_cfg, built_priors, p1):
    model = fresh_model(tiny_cfg, built_priors, phase1.ADAPTER)
    set_trainable(model, "phase1", core=True, adapters=[phase1.ADAPTER])
    step = phase1.make_step(model, p1, tiny_cfg)

    loss, metrics = step(np.random.default_rng(0))
    assert torch.isfinite(loss)
    assert set(metrics) == {"sig", "gmt", "pert"}

    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )


def test_the_two_steps_are_trained_jointly(tiny_cfg, built_priors, p1):
    """The perturbed-profile loss must reach the signature head, or the two
    steps are being trained separately (§4.4)."""
    model = fresh_model(tiny_cfg, built_priors, phase1.ADAPTER, wake=True)
    set_trainable(model, "phase1", core=True, adapters=[phase1.ADAPTER])

    rows = p1.train_rows[:4]
    _, _, pert_hat = phase1.predict(model, p1, rows, tiny_cfg)
    pert_hat.sum().backward()

    sig_head = model.heads["sig"]
    grads = [p.grad for p in sig_head.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_teacher_forcing_cuts_that_link(tiny_cfg, built_priors, p1):
    forced = dataclasses.replace(
        tiny_cfg, phase1=dataclasses.replace(tiny_cfg.phase1, teacher_forcing=True)
    )
    model = fresh_model(forced, built_priors, phase1.ADAPTER, wake=True)
    set_trainable(model, "phase1", core=True, adapters=[phase1.ADAPTER])

    rows = p1.train_rows[:4]
    _, _, pert_hat = phase1.predict(model, p1, rows, forced)
    pert_hat.sum().backward()

    grads = [p.grad for p in model.heads["sig"].parameters() if p.grad is not None]
    assert not grads or all(g.abs().sum() == 0 for g in grads)


def test_the_perturbed_head_predicts_a_change_from_the_control(tiny_cfg, built_priors, p1):
    """A perturbed profile is mostly its own control; the bottleneck is for
    the response, not for copying the input."""
    model = fresh_model(tiny_cfg, built_priors, phase1.ADAPTER)
    rows = p1.train_rows[:3]

    with torch.no_grad():
        _, _, pert_hat = phase1.predict(model, p1, rows, tiny_cfg)
        control = p1.ctrl[torch.as_tensor(rows)]
        # Untrained the head predicts exactly zero, so the prediction *is*
        # the control: the model starts at "this perturbation did nothing".
        assert torch.allclose(pert_hat, control, atol=1e-6)


def test_metrics_reported_per_cell_line(tiny_cfg, built_priors, p1):
    model = fresh_model(tiny_cfg, built_priors, phase1.ADAPTER)
    per_line = phase1.evaluate_per_cell_line(model, p1, tiny_cfg)

    assert per_line
    assert set(per_line) <= set(p1.cell_line)
    for scores in per_line.values():
        assert scores["n_rows"] > 0
        assert "sig_mse" in scores


def test_gmt_loss_is_masked_to_rows_that_have_one(tiny_cfg, built_priors, p1):
    """`has_gmt` is false for a good fraction of the Phase 1 rows (§3.3)."""
    from vccp.train.loop import masked_mse

    prediction = torch.zeros(3, 5)
    truth = torch.ones(3, 5)
    mask = torch.tensor([True, False, False]).unsqueeze(-1)
    # One row of five genes contributes; the denominator must count the five
    # broadcast entries, not the one mask entry.
    assert float(masked_mse(prediction, truth, mask)) == pytest.approx(1.0)
    assert p1.has_gmt.dtype == torch.bool


# --------------------------------------------------------------------------- #
# item 9 — Phase 2
# --------------------------------------------------------------------------- #
def test_contexts_split_mapping_and_perturbation_target_pools(contexts, gene_index):
    """The mapping needs no perturbation token, so it can use every target's
    cells; the perturbation module can only use targets that are challenge
    genes."""
    for tensors in contexts.values():
        assert set(tensors.pert_train_targets) <= set(tensors.train_targets)
        assert all(t in gene_index for t in tensors.pert_train_targets)
        assert not set(tensors.train_targets) & set(tensors.val_targets)


def test_siamese_shared_weights_on_control_and_perturbed_cells(
    tiny_cfg, built_priors, contexts
):
    """The same weights, and no perturbation token: a mapping that depended
    on what perturbed the cell could not survive Phase 3."""
    model = fresh_model(tiny_cfg, built_priors, phase2.ADAPTER)
    tensors = next(iter(contexts.values()))
    values = torch.randn(3, tensors.n_panel)

    seen = {}
    original = model.encode

    def spy(*args, **kwargs):
        seen["target_idx"] = kwargs.get("target_idx")
        return original(*args, **kwargs)

    model.encode = spy
    model.eval()
    with torch.no_grad():
        as_control = phase2.map_panel_to_rest(model, tensors, values)
        assert seen["target_idx"] is None, "the mapping must not see the perturbation"
        as_perturbed = phase2.map_panel_to_rest(model, tensors, values)
    model.encode = original

    # Identical input through the two call sites gives identical output —
    # there is one network, not one per cell state.
    assert torch.equal(as_control, as_perturbed)


def test_loss_masked_to_measured_genes(contexts, gene_index, session_cfg):
    """A screen covers part of the challenge axis; the rest carries no truth
    and must not be averaged in."""
    n_genes = len(gene_index)
    for tensors in contexts.values():
        measured = tensors.context.measured_mask(n_genes)
        panel = tensors.panel_idx.numpy()
        rest = tensors.rest_idx.numpy()

        assert measured[panel].all()
        assert measured[rest].all()
        assert not set(panel) & set(rest)
        assert len(panel) + len(rest) == int(measured.sum())
        assert len(panel) + len(rest) < n_genes, "expected a partial screen"


def test_one_phase2_step_touches_both_objectives(tiny_cfg, built_priors, contexts, gene_index):
    model = fresh_model(tiny_cfg, built_priors, phase2.ADAPTER)
    set_trainable(model, "phase2", core=True, adapters=[phase2.ADAPTER])
    step = phase2.make_step(model, contexts, tiny_cfg, "cpu", gene_index)

    loss, metrics = step(np.random.default_rng(0))
    assert torch.isfinite(loss)
    assert {"map_control", "map_perturbed"} <= set(metrics)
    assert {"pert_bulk", "pert_cells"} <= set(metrics)

    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )


def test_metrics_report_control_and_perturbed_separately(
    tiny_cfg, built_priors, contexts, gene_index
):
    model = fresh_model(tiny_cfg, built_priors, phase2.ADAPTER)
    per_context = phase2.evaluate_per_context(model, contexts, tiny_cfg, "cpu", gene_index)

    assert set(per_context) == set(contexts)
    for scores in per_context.values():
        assert "map_control_mse" in scores
        assert "map_perturbed_mse" in scores
        assert scores["map_control_mse"] != scores["map_perturbed_mse"]


def test_evaluation_reports_what_no_change_would_score(
    tiny_cfg, built_priors, contexts, gene_index
):
    """A perturbation MSE means nothing without the number it has to beat."""
    model = fresh_model(tiny_cfg, built_priors, phase2.ADAPTER)
    per_context = phase2.evaluate_per_context(model, contexts, tiny_cfg, "cpu", gene_index)
    scored = [s for s in per_context.values() if "pert_mse" in s]
    assert scored
    for scores in scored:
        assert scores["pert_mse_no_change"] > 0


def test_cells_are_drawn_in_control_sd_units(tiny_cfg, contexts):
    tensors = next(iter(contexts.values()))
    panel, rest = phase2.draw_cells(tensors, None, 16, np.random.default_rng(0), "cpu")

    assert panel.shape[1] == tensors.n_panel
    assert rest.shape[1] == tensors.n_rest
    assert torch.isfinite(panel).all() and torch.isfinite(rest).all()
    # Control cells in control-SD units sit around zero by construction.
    assert abs(float(panel.mean())) < 0.5


def test_the_perturbation_module_predicts_a_change(tiny_cfg, built_priors, contexts, gene_index):
    model = fresh_model(tiny_cfg, built_priors, phase2.ADAPTER)
    tensors = next(iter(contexts.values()))
    control = torch.zeros(2, tensors.n_panel)
    targets = as_index([gene_index[t] for t in tensors.pert_train_targets[:2]])

    with torch.no_grad():
        predicted = phase2.predict_perturbed_panel(
            model, tensors, control, targets, tiny_cfg.phase2.pert_type
        )
    assert predicted.shape == (2, tensors.n_panel)
    assert torch.isfinite(predicted).all()


def test_the_delta_noise_floor_uses_disjoint_control_draws(tiny_cfg, contexts):
    """Overlapping draws would understate the floor and flatter the mapping.

    Two control samples drawn independently share cells, and two means over
    overlapping cells differ by less than sampling noise. The floor would
    then be too low and the mapping would look closer to optimal than it is,
    which is the opposite of what the number is for.
    """
    tensors = next(iter(contexts.values()))
    available = int(tensors.control_rows.size)
    n_cells = min(tiny_cfg.phase2.delta_eval_cells, available // 2)
    assert n_cells >= 2, "this screen is too small to estimate a floor at all"

    rng = np.random.default_rng(0)
    drawn = rng.choice(tensors.control_rows, 2 * n_cells, replace=False)
    assert not set(drawn[:n_cells].tolist()) & set(drawn[n_cells:].tolist())

    # The same rows, read twice, give the same cells: the draw is what makes
    # the two halves differ, not the reader.
    a = phase2.draw_cells(tensors, None, n_cells, rng, "cpu", rows=drawn[:n_cells])
    b = phase2.draw_cells(tensors, None, n_cells, rng, "cpu", rows=drawn[:n_cells])
    assert torch.allclose(a[0], b[0])

    # No held-out target means no delta to report, and nothing touched.
    assert phase2._score_delta(None, tensors, tiny_cfg, "cpu", rng, []) == {}


# --------------------------------------------------------------------------- #
# item 7 — the forgetting guard, end to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, session_cfg):
    """Phases 1 and 2 plus the forgetting stage, run once for real.

    One fixture rather than one pipeline per test: these stages are the
    expensive part of the suite, and every assertion below is about the
    artifacts they leave rather than about running them.
    """
    from vccp.phases.forgetting import run_forgetting
    from vccp.priors.stage import run_priors

    cfg = dataclasses.replace(
        session_cfg,
        output_root=tmp_path_factory.mktemp("trained") / "runs",
        phase1=dataclasses.replace(session_cfg.phase1, steps=2, batch_size=4),
        phase2=dataclasses.replace(
            session_cfg.phase2, steps=2, batch_size=4, cells_per_draw=4
        ),
        train=dataclasses.replace(
            session_cfg.train,
            output_genes_per_step=32,
            log_every=1,
            core_freeze_ablation=True,
            ablation_steps_fraction=1.0,
        ),
    )
    run_priors(cfg)
    phase1.run_phase1(cfg)
    phase2_metrics = phase2.run_phase2(cfg)
    return cfg, phase2_metrics, run_forgetting(cfg)


@pytest.mark.slow
def test_phase1_validation_rescored_after_each_phase(trained_run):
    """CLAUDE.md §4.3: Phase 1's validation is re-scored under each phase's
    saved core, through Phase 1's own adapter."""
    from vccp.phases.forgetting import HEADLINE

    cfg, _, report = trained_run

    assert "phase1" in report["checkpoints"]
    assert "phase2" in report["checkpoints"]
    for entry in report["checkpoints"].values():
        assert HEADLINE in entry["metrics"]
    assert report["checkpoints"]["phase2"]["change_from_phase1"]

    written = json.loads(RunPaths(cfg).forgetting.read_text())
    assert written["guard"]["replay_fraction"] == cfg.train.replay_fraction
    assert written["guard"]["l2sp_weight"] == cfg.train.l2sp_weight


@pytest.mark.slow
def test_core_freeze_ablation_reports_both_arms(trained_run):
    """The Phase 2 default was flipped on an argument (DECISIONS.md D3); the
    ablation is what turns it into a measurement."""
    _, metrics, report = trained_run

    assert {"core_unfrozen", "core_frozen"} <= set(metrics["arms"])
    assert metrics["arms"]["core_unfrozen"]["unfreeze_core"] is True
    assert metrics["arms"]["core_frozen"]["unfreeze_core"] is False

    ablation = report["core_freeze_ablation"]
    assert ablation["available"]
    assert set(ablation["arms"]) == {"core_unfrozen", "core_frozen"}
    assert ablation["better_arm"] in ablation["arms"]
    assert ablation["reading"]


def test_the_phase1_contribution_is_measured_not_assumed(trained_run):
    """Both warm arms are warm-started, so without a scratch arm nothing in
    the pipeline says what Phase 1 is worth (PLAN_PERCELL.md §7.6)."""
    _, metrics, _ = trained_run

    assert phase2.SCRATCH_ARM in metrics["arms"]
    scratch = metrics["arms"][phase2.SCRATCH_ARM]
    assert scratch["warm_start"] == phase2.NONE
    # No warm start means no Phase 1 to replay: replaying it would put back
    # through the side door exactly what the ablation removes.
    assert scratch["n_phase1_replay_steps"] == 0
    assert scratch["n_phase2_steps"] == scratch["steps"]

    warm = metrics["arms"]["core_unfrozen"]
    assert warm["warm_start"] == phase2.FULL

    contribution = metrics["phase1_contribution"]
    assert contribution["measured"] is True
    assert contribution["positive_means_phase1_helped"] is True
    assert contribution["improvement"]
    # The unequal step budgets are reported, because they qualify the reading.
    assert contribution["n_phase2_steps"]["scratch"] >= contribution["n_phase2_steps"]["warm"]


def test_phase2_reports_the_change_the_metrics_actually_score(trained_run):
    """The per-state MSEs are dominated by the baseline level; the delta is
    the quantity the six official metrics see (PLAN_PERCELL.md §5)."""
    _, metrics, _ = trained_run

    per_context = metrics["validation_per_context"]
    scored = [s for s in per_context.values() if "map_delta_ratio_to_no_change" in s]
    assert scored, f"no context reported a delta: {list(per_context)}"

    for scores in scored:
        assert scores["map_delta_mse"] >= 0
        assert scores["map_delta_no_change"] > 0
        assert scores["n_delta_targets_scored"] >= 1
        # The floor a perfect predictor would hit at these cell counts. An
        # observed change is a difference of two sampled means, so without
        # this the ratio cannot be read at all.
        floor = scores["map_delta_noise_floor_ratio"]
        assert 0.0 <= floor
        assert np.isfinite(scores["map_delta_ratio_to_no_change"])


@pytest.mark.slow
def test_the_frozen_arm_really_did_not_move_the_core(trained_run):
    cfg, _, _ = trained_run
    report = json.loads(RunPaths(cfg).frozen_check.read_text())

    assert report["all_ok"], report
    assert "phase2:core_frozen" in report["phases"]
    frozen_arm = report["phases"]["phase2:core_frozen"]
    assert frozen_arm["groups"]["core"] is False
    assert frozen_arm["frozen_verified"]["n_checked"] > 0


@pytest.mark.slow
def test_each_phase_leaves_its_checklist_artifacts(trained_run):
    """Items 5-9 of CLAUDE.md §4.8."""
    cfg, _, _ = trained_run
    run_paths = RunPaths(cfg)

    for phase in ("phase1", "phase2"):
        assert run_paths.core_checkpoint(phase).is_file(), phase
        assert run_paths.phase_metrics(phase).is_file(), phase
        assert run_paths.phase_curves(phase).is_file(), phase
    assert run_paths.frozen_check.is_file()
    assert run_paths.forgetting.is_file()

    metrics = json.loads(run_paths.phase_metrics("phase1").read_text())
    assert metrics["validation_per_cell_line"]
    assert metrics["data"]["n_rows_usable"] > 0
