"""The training machinery: what trains, the frozen check, replay, L2-SP."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vccp.models.core import as_index, build_model
from vccp.train.freeze import (
    FrozenCheck,
    describe,
    parameter_group,
    set_trainable,
    snapshot,
    verify,
)
from vccp.train.l2sp import L2SP
from vccp.train.loop import masked_mse, pearson, run_training, sample_output_genes
from vccp.train.replay import ReplayMixer, ReplayTask


@pytest.fixture
def model(session_cfg, built_priors):
    # Seeded, so the weights are the same whatever ran before in this
    # session. Without it the initialization depends on how far earlier
    # tests advanced the global stream — the same trap DECISIONS.md D84
    # records for the ablation arms.
    torch.manual_seed(session_cfg.seed)
    built = build_model(session_cfg, built_priors)
    for name in ("phase1", "phase2"):
        built.add_adapter(name)
    return built


def toy_step(model, gene_in=60, gene_out=60, seed=0):
    idx_in, idx_out = as_index(np.arange(gene_in)), as_index(np.arange(gene_out))
    # Same reason as the fixture: a toy regression problem drawn from
    # wherever the global stream happens to be is a different problem in
    # every session, and "the loss goes down in 30 steps" is not true of
    # every one of them.
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(4, gene_in, 2, generator=generator)
    target = torch.randn(4, gene_out, generator=generator)

    def step(rng):
        prediction = model(values, idx_in, idx_out)
        return ((prediction - target) ** 2).mean(), {}

    return step


# --------------------------------------------------------------------------- #
# what trains
# --------------------------------------------------------------------------- #
def test_parameter_groups_are_recognised():
    assert parameter_group("blocks.0.cross.to_q.base.weight") == "core"
    assert parameter_group("blocks.0.cross.to_q.up.phase2") == "adapters"
    assert parameter_group("blocks.0.cross.to_q.down.phase2") == "adapters"
    assert parameter_group("vocabulary.deltas.feature") == "delta"
    assert parameter_group("vocabulary.projections.feature.weight") == "projections"
    assert parameter_group("heads.sig.1.base.weight") == "heads"


def test_a_frozen_core_leaves_only_adapters_and_delta_trainable(model):
    plan = set_trainable(model, "phase2", core=False, adapters=["phase2"], delta=True)
    groups = {parameter_group(name) for name in plan.trainable}
    assert groups <= {"adapters", "delta", "heads"}
    assert "core" in {parameter_group(name) for name in plan.frozen}


def test_only_the_named_adapter_trains(model):
    """An earlier phase's adapter must not drift while a later phase runs."""
    plan = set_trainable(model, "phase2", core=False, adapters=["phase2"])
    trainable = [n for n in plan.trainable if parameter_group(n) == "adapters"]
    assert trainable
    assert all(n.endswith("phase2") for n in trainable)
    assert any(n.endswith("phase1") for n in plan.frozen)


# --------------------------------------------------------------------------- #
# item 6 — the frozen check
# --------------------------------------------------------------------------- #
def test_declared_frozen_parameters_are_bitwise_unchanged_after_a_training_step(model):
    plan = set_trainable(model, "phase2", core=False, adapters=["phase2"], heads=False)
    check = FrozenCheck()
    before = check.begin(model, plan)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    step = toy_step(model)
    for _ in range(3):
        optimizer.zero_grad()
        loss, _ = step(None)
        loss.backward()
        optimizer.step()

    result = check.end(model, plan, before)
    assert result["frozen_verified"]["ok"], result["frozen_verified"]["changed"]
    assert result["frozen_verified"]["n_checked"] > 0
    assert check.report()["all_ok"]


def test_the_frozen_check_notices_a_parameter_that_did_move(model):
    """The check is only worth having if it fires."""
    plan = set_trainable(model, "phase2", core=False, adapters=["phase2"])
    before = snapshot(model, plan.frozen)

    with torch.no_grad():
        name = plan.frozen[0]
        dict(model.named_parameters())[name].add_(1.0)

    result = verify(model, before)
    assert not result["ok"]
    assert name in result["changed"]


def test_the_check_reports_parameter_counts_by_group(model):
    plan = set_trainable(model, "phase1", core=True, adapters=["phase1"], projections=True)
    report = describe(model, plan)
    assert report["n_trainable_parameters"] > 0
    assert set(report["parameters_by_group"]) <= {
        "core",
        "adapters",
        "delta",
        "projections",
        "heads",
    }


# --------------------------------------------------------------------------- #
# item 7 — the forgetting guard
# --------------------------------------------------------------------------- #
def test_l2sp_is_zero_at_the_start_and_grows_with_drift(model):
    set_trainable(model, "phase2", core=True, adapters=["phase2"])
    l2sp = L2SP(model, weight=1.0)

    assert l2sp.enabled
    assert l2sp.penalty(model).detach().item() == pytest.approx(0.0, abs=1e-6)
    assert l2sp.drift(model) == pytest.approx(0.0, abs=1e-6)

    with torch.no_grad():
        next(p for p in model.parameters() if p.requires_grad).add_(1.0)
    assert l2sp.penalty(model).detach().item() > 0
    assert l2sp.drift(model) > 0


def test_l2sp_is_off_when_its_weight_is_zero(model):
    set_trainable(model, "phase2", core=True, adapters=["phase2"])
    l2sp = L2SP(model, weight=0.0)
    assert not l2sp.enabled
    assert l2sp.penalty(model).detach().item() == 0.0


def test_l2sp_and_replay_are_on_by_default_in_phase2(session_cfg):
    """CLAUDE.md §4.3: both are on by default in the later phases."""
    assert session_cfg.train.l2sp_weight > 0
    assert session_cfg.train.replay_fraction > 0


def test_replay_runs_the_earlier_objective_not_the_later_one():
    """Replaying Phase 1's batches through Phase 2's loss would measure
    nothing, so a task carries its own loss."""
    calls = []

    def phase1_step(rng):
        calls.append("phase1")
        return torch.zeros(()), {}

    mixer = ReplayMixer([ReplayTask("phase1", phase1_step)], fraction=1.0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        task = mixer.pick(rng)
        assert task is not None
        task.step(rng)

    assert calls == ["phase1"] * 5
    assert mixer.report()["steps_replayed"] == {"phase1": 5}


def test_replay_is_off_when_the_fraction_is_zero():
    mixer = ReplayMixer([ReplayTask("phase1", lambda rng: (torch.zeros(()), {}))], fraction=0.0)
    assert not mixer.enabled
    rng = np.random.default_rng(0)
    assert all(mixer.pick(rng) is None for _ in range(10))


def test_replay_fires_at_about_the_configured_rate():
    mixer = ReplayMixer([ReplayTask("a", lambda rng: (torch.zeros(()), {}))], fraction=0.3)
    rng = np.random.default_rng(0)
    fired = sum(mixer.pick(rng) is not None for _ in range(2000))
    assert 0.25 < fired / 2000 < 0.35


# --------------------------------------------------------------------------- #
# the loop and its helpers
# --------------------------------------------------------------------------- #
def test_training_reduces_the_loss_and_records_a_curve(model, session_cfg):
    set_trainable(model, "phase1", core=True, adapters=["phase1"])
    step = toy_step(model)

    result = run_training(
        model,
        step,
        steps=30,
        cfg=session_cfg,
        phase="phase1",
        rng=np.random.default_rng(0),
        device="cpu",
    )
    assert result.steps == 30
    assert result.curve
    assert result.curve[0]["loss"] > result.curve[-1]["loss"]


def test_training_refuses_when_nothing_is_trainable(model, session_cfg):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="nothing is trainable"):
        run_training(
            model,
            toy_step(model),
            steps=1,
            cfg=session_cfg,
            phase="phase1",
            rng=np.random.default_rng(0),
            device="cpu",
        )


def test_masked_mse_ignores_what_the_mask_drops():
    prediction = torch.tensor([[1.0, 100.0]])
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])

    assert float(masked_mse(prediction, target, mask)) == pytest.approx(0.0)
    assert float(masked_mse(prediction, target)) > 0
    # An all-false mask must not divide by zero.
    assert float(masked_mse(prediction, target, torch.zeros_like(mask))) == 0.0


def test_masked_mse_counts_a_broadcast_mask_after_broadcasting():
    """A per-row mask is (B, 1) against a (B, G) error. Counting the mask's
    own entries would divide a sum over B x G terms by B."""
    prediction = torch.zeros(3, 5)
    target = torch.ones(3, 5)
    row_mask = torch.tensor([True, False, False]).unsqueeze(-1)

    assert float(masked_mse(prediction, target, row_mask)) == pytest.approx(1.0)
    # Same answer as spelling the mask out in full.
    full = row_mask.expand(3, 5)
    assert float(masked_mse(prediction, target, full)) == pytest.approx(1.0)


def test_sampled_output_genes_are_a_subset_in_order():
    genes = as_index(np.arange(100))
    rng = np.random.default_rng(0)

    subset, positions = sample_output_genes(genes, 10, rng)
    assert subset.shape[0] == 10
    assert list(positions) == sorted(positions)
    assert torch.equal(subset, genes[torch.as_tensor(positions)])

    whole, positions = sample_output_genes(genes, 0, rng)
    assert torch.equal(whole, genes)
    assert len(positions) == 100


def test_pearson_is_nan_safe():
    assert np.isnan(pearson(np.zeros(5), np.arange(5)))
    assert pearson(np.arange(5), np.arange(5)) == pytest.approx(1.0)
    assert np.isnan(pearson(np.array([1.0]), np.array([1.0])))


# --------------------------------------------------------------------------- #
# what the phase keeps (DECISIONS.md D85)
# --------------------------------------------------------------------------- #


def scripted_eval(values, metric="ratio"):
    """An evaluation that reads off a script, one value per call."""
    remaining = list(values)

    def evaluate():
        return {metric: remaining.pop(0)}

    return evaluate


def test_the_weights_kept_are_the_best_ones_not_the_last(model, session_cfg):
    """A phase's checkpoint is what the next phase inherits. A server arm
    reached 0.817 at step 3000 and ended at 1.001 — keeping the last step
    means shipping a model we measured and knew to be worse."""
    set_trainable(model, "phase1", core=True, adapters=["phase1"])

    result = run_training(
        model,
        toy_step(model),
        steps=4,
        cfg=session_cfg,
        phase="phase1",
        rng=np.random.default_rng(0),
        device="cpu",
        evaluate=scripted_eval([0.9, 0.5, 0.8, 1.0]),
        eval_every=1,
        select_on="ratio",
        select_anchor=1.0,
    )

    assert result.best_step == 2
    assert result.restored == "best"
    assert result.final_metrics["ratio"] == 1.0
    assert result.best_metrics["ratio"] == 0.5
    # It learned and then lost all of it, which `final / best` calls 2.0x but
    # which is really "nothing left": 1.0 is the no-change baseline.
    assert result.signal_retained == pytest.approx(0.0)
    assert result.instability == pytest.approx(2.0)
    # The tensors are not carried around after the restore.
    assert result.best_state is None


def test_keeping_the_final_weights_stays_available(model, session_cfg):
    """`final` remains selectable, because "what the last step produced" is
    the honest answer when no validation was selected on."""
    set_trainable(model, "phase1", core=True, adapters=["phase1"])

    result = run_training(
        model,
        toy_step(model),
        steps=2,
        cfg=session_cfg,
        phase="phase1",
        rng=np.random.default_rng(0),
        device="cpu",
        evaluate=scripted_eval([0.5, 1.0]),
        eval_every=1,
        select_on="ratio",
        keep_best=False,
    )

    assert result.best_step == 1
    assert result.restored == "final"


def test_signal_retained_reads_an_anchored_metric_the_way_it_means(session_cfg):
    """`final / best` is the wrong scale for a ratio against no change. The
    three measurements that mattered on the server, by hand."""
    from vccp.train.loop import TrainResult

    def retained(best, final):
        return TrainResult(
            steps=1,
            selected_on="r",
            anchor=1.0,
            best_metrics={"r": best},
            final_metrics={"r": final},
        ).signal_retained

    # The server arm that read as "not diverged" at 1.22x, just under the
    # 1.25 threshold, having kept none of what it learned.
    assert retained(0.8171, 1.0013) == pytest.approx(-0.007, abs=0.01)
    assert retained(0.9171, 1.0642) == pytest.approx(-0.77, abs=0.01)
    # Ending at the best keeps all of it; ending halfway keeps half.
    assert retained(0.8, 0.8) == pytest.approx(1.0)
    assert retained(0.8, 0.9) == pytest.approx(0.5)
    # No anchor, no reading — the caller has to say what "nothing learned"
    # scores before the share of it can mean anything.
    assert TrainResult(
        steps=1, selected_on="r", best_metrics={"r": 0.8}, final_metrics={"r": 0.9}
    ).signal_retained is None


def test_an_arm_that_never_beat_the_baseline_has_no_signal_to_have_kept():
    """`(anchor - final) / (anchor - best)` inverts when the best is on the
    wrong side of the anchor: a mini arm whose best was 1.0844 against a
    baseline of 1.0 reported that it kept 116% of what it learned."""
    from vccp.train.loop import TrainResult

    never = TrainResult(
        steps=1, selected_on="r", anchor=1.0,
        best_metrics={"r": 1.0844}, final_metrics={"r": 1.0982},
    )
    assert never.signal_retained is None

    # Higher-is-better metrics read the same way, mirrored.
    rising = TrainResult(
        steps=1, selected_on="r", anchor=0.5, lower_is_better=False,
        best_metrics={"r": 0.9}, final_metrics={"r": 0.7},
    )
    assert rising.signal_retained == pytest.approx(0.5)
    assert TrainResult(
        steps=1, selected_on="r", anchor=0.5, lower_is_better=False,
        best_metrics={"r": 0.4}, final_metrics={"r": 0.3},
    ).signal_retained is None


def test_a_share_of_almost_nothing_is_not_reported_as_a_share():
    """An arm whose best beat the baseline by 0.0016 and ended 0.155 above it
    reported keeping -9922% of what it learned (D91). The ratio is right; as
    a percentage it is arithmetic, not information."""
    from vccp.train.loop import TrainResult

    barely = TrainResult(
        steps=1, selected_on="r", anchor=1.0,
        best_metrics={"r": 0.9984}, final_metrics={"r": 1.1554},
    )
    assert barely.signal_headroom == pytest.approx(0.0016)
    assert barely.signal_retained == pytest.approx(-97.1, abs=0.5)

    # A run with real headroom keeps a readable share.
    real = TrainResult(
        steps=1, selected_on="r", anchor=1.0,
        best_metrics={"r": 0.7585}, final_metrics={"r": 0.8},
    )
    assert real.signal_headroom == pytest.approx(0.2415)
    assert real.signal_retained == pytest.approx(0.828, abs=0.01)

    # Higher-is-better metrics measure the headroom in their own direction.
    rising = TrainResult(
        steps=1, selected_on="r", anchor=0.5, lower_is_better=False,
        best_metrics={"r": 0.9}, final_metrics={"r": 0.7},
    )
    assert rising.signal_headroom == pytest.approx(0.4)
