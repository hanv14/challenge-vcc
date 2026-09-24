"""Cross-assay transfer and the delta consistency term (PLAN_PERCELL §5, §7).

Three pieces land together in P2 and they answer three separate questions:

* the **type vocabulary** gives LINCS knockout and challenge knockdown
  distinct rows around a shared base, so Phase 1's evidence informs Phase 2
  without asserting the magnitudes are the same;
* the **delta consistency term** makes the mapping answer for the difference
  between its two arms, which is the only quantity the official metrics
  score;
* the **`core_scratch` arm** measures what Phase 1 is worth instead of
  assuming it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vccp.config import ConfigError, Phase2, Train
from vccp.models.core import as_index, build_model, perturbation_types
from vccp.train import ot
from vccp.phases import phase2 as phase2_mod


@pytest.fixture(scope="module")
def typed_model(session_cfg, built_priors):
    return build_model(session_cfg, built_priors)


# --------------------------------------------------------------------------- #
# the perturbation-type vocabulary (§7.4)
# --------------------------------------------------------------------------- #


def test_the_types_come_from_the_config_not_from_code(session_cfg):
    """A new assay is a config change, never an edit to a list in a module."""
    assert perturbation_types(session_cfg) == ("crispr_ko", "crispri")
    assert session_cfg.phase1.pert_type != session_cfg.phase2.pert_type
    # Replogle and the challenge are the same modality, so they share a row —
    # that sharing is the transfer the vocabulary exists to allow.
    assert session_cfg.phase2.pert_type == session_cfg.phase3.pert_type


def test_each_assay_has_its_own_offset_around_a_shared_base(typed_model):
    model = typed_model
    assert model.pert_types == ("crispr_ko", "crispri")
    assert model.pert_type_delta.shape == (2, model.dim)

    # Every offset starts at zero, so before any training the two assays are
    # the same vector: transfer is the default and divergence has to be paid
    # for by the penalty.
    assert torch.allclose(model.pert_type_delta, torch.zeros_like(model.pert_type_delta))
    for name in model.pert_types:
        assert torch.allclose(model.perturbation_type(name), model.pert_type_base)

    with torch.no_grad():
        model.pert_type_delta[0] += 1.0
    knockout = model.perturbation_type("crispr_ko")
    knockdown = model.perturbation_type("crispri")
    assert not torch.allclose(knockout, knockdown)
    assert torch.allclose(knockout - knockdown, torch.ones_like(knockout))
    with torch.no_grad():
        model.pert_type_delta.zero_()


def test_the_offsets_are_penalized_toward_the_shared_base(typed_model):
    model = typed_model
    assert float(model.pert_type_penalty().detach()) == 0.0
    with torch.no_grad():
        model.pert_type_delta[1] += 2.0
    assert float(model.pert_type_penalty().detach()) > 0.0
    # The combined penalty a phase's loss adds includes it.
    assert float(model.regularization().detach()) >= float(
        model.pert_type_penalty().detach()
    )
    with torch.no_grad():
        model.pert_type_delta.zero_()


def test_the_type_changes_what_the_model_predicts(typed_model):
    """Otherwise the row is decoration rather than conditioning.

    The offset has to vary across the embedding's dimensions. A *constant*
    added to every dimension of a token is removed by the block's
    `cross_norm_tokens` LayerNorm before the latents ever see it, so a type
    row can only ever say something through its direction, never through a
    uniform shift.
    """
    model = typed_model
    generator = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for head in model.heads.values():
            head[-1].base.weight.normal_(0, 0.1)
        model.pert_type_delta[0] += torch.randn(model.dim, generator=generator)

    gene_idx = as_index(list(range(12)), "cpu")
    values = torch.randn(2, 12, 1, generator=torch.Generator().manual_seed(0))
    out = {
        name: model(values, gene_idx, gene_idx, target_idx=as_index([0, 1], "cpu"), pert_type=name)
        for name in model.pert_types
    }
    assert not torch.allclose(out["crispr_ko"], out["crispri"])
    with torch.no_grad():
        model.pert_type_delta.zero_()


def test_a_uniform_offset_is_invisible_and_that_is_expected(typed_model):
    """The LayerNorm above, stated as a property rather than a surprise."""
    model = typed_model
    with torch.no_grad():
        model.pert_type_delta[0] += 0.5

    gene_idx = as_index(list(range(12)), "cpu")
    values = torch.randn(2, 12, 1, generator=torch.Generator().manual_seed(0))
    target_idx = as_index([0, 1], "cpu")
    tokens = [
        model.tokenize(values, gene_idx, target_idx, name) for name in model.pert_types
    ]
    latents = [
        model.encode(values, gene_idx, target_idx=target_idx, pert_type=name)
        for name in model.pert_types
    ]
    # The tokens differ by exactly the offset...
    assert float((tokens[0] - tokens[1]).detach().abs().max()) == pytest.approx(
        0.5, abs=1e-5
    )
    # ...and the latents do not, because the normalization removed it.
    assert float((latents[0] - latents[1]).detach().abs().max()) < 1e-4
    with torch.no_grad():
        model.pert_type_delta.zero_()


def test_a_perturbation_token_without_a_declared_assay_is_an_error(typed_model):
    """Silently treating a knockout as a knockdown is the bug this prevents."""
    gene_idx = as_index(list(range(6)), "cpu")
    values = torch.zeros(1, 6, 1)
    with pytest.raises(ValueError, match="without one"):
        typed_model.tokenize(values, gene_idx, target_idx=as_index([0], "cpu"))
    with pytest.raises(KeyError, match="unknown perturbation type"):
        typed_model.perturbation_type("shRNA")

    # No perturbation token, no type needed: the mapping is a statement about
    # a transcriptome, not about what was done to it.
    typed_model.tokenize(values, gene_idx)


# --------------------------------------------------------------------------- #
# the delta consistency term (§5)
# --------------------------------------------------------------------------- #


def test_the_delta_loss_reads_as_a_fraction_of_the_change():
    observed = torch.tensor([0.4, -0.2, 0.1])
    assert float(phase2_mod._delta_loss(torch.zeros(3), observed)) == pytest.approx(1.0)
    assert float(phase2_mod._delta_loss(observed, observed)) == pytest.approx(0.0)
    assert float(phase2_mod._delta_loss(observed * 0.5, observed)) == pytest.approx(0.25)


def test_the_delta_loss_survives_a_batch_with_no_change_at_all():
    """A degenerate batch must not divide by zero and poison the gradient."""
    value = phase2_mod._delta_loss(torch.zeros(4), torch.zeros(4))
    assert torch.isfinite(value)


def test_the_delta_term_is_what_the_metrics_score_and_the_arms_are_not(typed_model):
    """Two arms can both be accurate in level and wrong about the change.

    This is the failure the term exists to catch: a constant offset added to
    the perturbed arm leaves each arm's own error small while making the
    predicted change wrong by exactly that offset.
    """
    control_truth = torch.randn(16, 20, generator=torch.Generator().manual_seed(1))
    change = torch.full((20,), 0.3)
    perturbed_truth = control_truth + change

    # A "mapping" that reproduces the control arm and the perturbed arm to
    # within a shared constant: per-arm error is the constant squared...
    offset = 0.3
    control_predicted = control_truth + offset
    perturbed_predicted = perturbed_truth + offset
    assert float(((control_predicted - control_truth) ** 2).mean()) == pytest.approx(
        offset**2, rel=1e-5
    )
    # ...while the change is predicted perfectly, because the offset cancels.
    assert float(
        phase2_mod._delta_loss(
            perturbed_predicted.mean(0) - control_predicted.mean(0),
            perturbed_truth.mean(0) - control_truth.mean(0),
        )
    ) == pytest.approx(0.0, abs=1e-6)

    # And the converse: arms accurate on their own but with the perturbed one
    # shifted score well per arm and fail the delta entirely.
    shifted = perturbed_truth - change
    assert float(
        phase2_mod._delta_loss(
            shifted.mean(0) - control_truth.mean(0),
            perturbed_truth.mean(0) - control_truth.mean(0),
        )
    ) == pytest.approx(1.0, rel=1e-5)


def test_the_config_rejects_impossible_delta_settings():
    with pytest.raises(ConfigError, match="loss_delta"):
        Phase2(loss_delta=-1.0).validate()
    with pytest.raises(ConfigError, match="delta_eval_cells"):
        Phase2(delta_eval_cells=1).validate()
    with pytest.raises(ConfigError, match="delta_eval_targets"):
        Phase2(delta_eval_targets=0).validate()
    with pytest.raises(ConfigError, match="type_delta_l2"):
        Train(type_delta_l2=-1.0).validate()


# --------------------------------------------------------------------------- #
# the Phase 1 contribution ablation (§7.6)
# --------------------------------------------------------------------------- #


def test_the_contribution_is_reported_with_the_step_budgets_that_qualify_it():
    arms = {
        "core_unfrozen": {
            "validation": {"pert_mse_ratio_to_no_change": 0.8, "map_delta_ratio_to_no_change": 0.5},
            "n_phase2_steps": 450,
        },
        phase2_mod.SCRATCH_ARM: {
            "validation": {"pert_mse_ratio_to_no_change": 0.9, "map_delta_ratio_to_no_change": 0.4},
            "n_phase2_steps": 600,
        },
    }
    report = phase2_mod._phase1_contribution(arms)

    assert report["measured"] is True
    assert report["warm_arm"] == "core_unfrozen"
    # Lower is better, so scratch - warm > 0 means Phase 1 helped.
    assert report["improvement"]["pert_mse_ratio_to_no_change"] == pytest.approx(0.1)
    # ...and negative means it did not, on that metric.
    assert report["improvement"]["map_delta_ratio_to_no_change"] == pytest.approx(-0.1)
    # The budgets are reported because they are unequal, and which arm the
    # inequality helps is measured rather than assumed: a warm arm gives
    # steps to replay, but the scratch arm's whole budget is
    # `ablation_steps_fraction` of the main arm's, so either can come out
    # ahead.
    assert report["n_phase2_steps"] == {"warm": 450, "scratch": 600}
    assert report["budget_favours"] == phase2_mod.SCRATCH_ARM


def test_a_run_that_diverges_is_judged_at_its_best_not_its_last_step():
    """The first server Phase 2 ended 2.86x worse than its own best.

    Reading an ablation off the final step then measures the divergence
    rather than the thing the ablation is about.
    """
    from vccp.train.loop import TrainResult

    result = TrainResult(steps=600)
    result.selected_on = phase2_mod.SELECTION_METRIC
    result.best_metrics = {phase2_mod.SELECTION_METRIC: 0.9991}
    result.best_step = 300
    result.final_metrics = {phase2_mod.SELECTION_METRIC: 2.8528}

    assert result.divergence == pytest.approx(2.8528 / 0.9991, rel=1e-4)
    assert result.divergence > phase2_mod.DIVERGENCE_RATIO
    assert result.as_dict()["best_step"] == 300

    # A steady run is not flagged.
    steady = TrainResult(steps=450)
    steady.selected_on = phase2_mod.SELECTION_METRIC
    steady.best_metrics = {phase2_mod.SELECTION_METRIC: 0.9797}
    steady.final_metrics = {phase2_mod.SELECTION_METRIC: 1.0727}
    assert steady.divergence < phase2_mod.DIVERGENCE_RATIO

    # And a run with nothing selected reports nothing rather than guessing.
    assert TrainResult(steps=10).divergence is None


def test_a_run_that_swings_mid_training_is_unstable_even_if_it_lands_well():
    """`final / best` cannot see a swing that recovers.

    The 6000-step server arm hit 2.4615 at step 4500 and ended at 1.2458 —
    1.15x its best, under the divergence threshold — so it read as steady
    while being anything but.
    """
    from vccp.train.loop import TrainResult

    result = TrainResult(steps=6000)
    result.selected_on = phase2_mod.SELECTION_METRIC
    result.best_metrics = {phase2_mod.SELECTION_METRIC: 1.0804}
    result.best_step = 3000
    result.worst_value, result.worst_step = 2.4615, 4500
    result.final_metrics = {phase2_mod.SELECTION_METRIC: 1.2458}

    assert result.divergence == pytest.approx(1.153, abs=1e-3)
    assert result.divergence < phase2_mod.DIVERGENCE_RATIO, "the ending looks fine"
    assert result.instability == pytest.approx(2.278, abs=1e-3)
    assert result.instability > phase2_mod.DIVERGENCE_RATIO, "the swing does not"

    # A genuinely steady run trips neither.
    steady = TrainResult(steps=4500)
    steady.selected_on = phase2_mod.SELECTION_METRIC
    steady.best_metrics = {phase2_mod.SELECTION_METRIC: 0.8068}
    steady.worst_value = 0.9205
    steady.final_metrics = {phase2_mod.SELECTION_METRIC: 0.8068}
    assert steady.divergence == pytest.approx(1.0)
    assert steady.instability < phase2_mod.DIVERGENCE_RATIO


def test_a_cold_main_arm_makes_the_scratch_ablation_redundant():
    """Turning the warm start off would otherwise train the same arm twice."""
    from vccp.config import Phase2

    assert Phase2().warm_start == phase2_mod.FULL, "three phases is the default"
    Phase2(warm_start=phase2_mod.NONE).validate()


def test_the_contribution_names_the_arms_that_diverged():
    """The comparison uses the best, but the checkpoint on disk is the final
    one — so the number and the model are not the same thing."""
    report = phase2_mod._phase1_contribution({
        "core_unfrozen": {
            "validation": {"pert_mse_ratio_to_no_change": 0.9991},
            "n_phase2_steps": 450,
            "diverged": True,
        },
        phase2_mod.SCRATCH_ARM: {
            "validation": {"pert_mse_ratio_to_no_change": 0.9797},
            "n_phase2_steps": 450,
            "diverged": False,
        },
    })
    assert report["arms_that_diverged"] == ["core_unfrozen"]
    assert "best validation" in report["read_at"]
    assert report["budgets_matched"] is True
    # Best against best, the warm arm is only 0.0194 behind — not 0.202.
    assert report["improvement"]["pert_mse_ratio_to_no_change"] == pytest.approx(
        -0.0194, abs=1e-4
    )


def test_the_scratch_arm_is_given_the_warm_arms_phase2_steps():
    """Equal totals are unequal training: the warm arm spends some of its
    steps on Phase 1 and the scratch arm spends none.

    The first server run measured the contribution at −0.202 with budgets of
    442 against 600 — 36% more training for the arm that won — which is a
    number that cannot be read.
    """
    import dataclasses

    from vccp.config import load_config

    cfg = load_config("configs/server.yaml")
    warm_phase2 = cfg.phase2.steps * (1.0 - cfg.train.replay_fraction)
    assert phase2_mod.scratch_arm_steps(cfg) == pytest.approx(warm_phase2, abs=1)

    off = dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, match_phase2_steps=False)
    )
    assert phase2_mod.scratch_arm_steps(off) == cfg.phase2.steps
    assert phase2_mod.scratch_arm_steps(off) > phase2_mod.scratch_arm_steps(cfg)

    # `ablation_steps_fraction` still applies on top, and the floor is 1.
    half = dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, ablation_steps_fraction=0.5)
    )
    assert phase2_mod.scratch_arm_steps(half) == pytest.approx(warm_phase2 / 2, abs=1)
    tiny = dataclasses.replace(
        cfg,
        phase2=dataclasses.replace(cfg.phase2, steps=1),
        train=dataclasses.replace(cfg.train, ablation_steps_fraction=0.01),
    )
    assert phase2_mod.scratch_arm_steps(tiny) == 1


def test_budgets_that_match_are_reported_as_matched():
    """Replay is sampled per step, so two arms meant to train equally land a
    few steps apart. Reporting that as a bias reports noise."""
    def report(warm_steps, scratch_steps):
        return phase2_mod._phase1_contribution({
            "core_unfrozen": {
                "validation": {"pert_mse_ratio_to_no_change": 0.8},
                "n_phase2_steps": warm_steps,
            },
            phase2_mod.SCRATCH_ARM: {
                "validation": {"pert_mse_ratio_to_no_change": 0.9},
                "n_phase2_steps": scratch_steps,
            },
        })

    matched = report(450, 447)
    assert matched["budgets_matched"] is True
    assert matched["budget_favours"] is None

    skewed = report(442, 600)
    assert skewed["budgets_matched"] is False
    assert skewed["budget_favours"] == phase2_mod.SCRATCH_ARM


def test_the_budget_bias_can_run_the_other_way():
    """On `mini.yaml` it does: ablation_steps_fraction 0.5 leaves the scratch
    arm with fewer Phase 2 steps than the warm one, not more."""
    def report(warm_steps, scratch_steps):
        return phase2_mod._phase1_contribution({
            "core_unfrozen": {
                "validation": {"pert_mse_ratio_to_no_change": 0.8},
                "n_phase2_steps": warm_steps,
            },
            phase2_mod.SCRATCH_ARM: {
                "validation": {"pert_mse_ratio_to_no_change": 0.9},
                "n_phase2_steps": scratch_steps,
            },
        })

    assert report(113, 75)["budget_favours"] == "warm"
    assert report(75, 113)["budget_favours"] == phase2_mod.SCRATCH_ARM
    assert report(100, 100)["budget_favours"] is None


def test_the_contribution_says_so_when_it_cannot_be_measured():
    report = phase2_mod._phase1_contribution({"core_unfrozen": {"validation": {}}})
    assert report["measured"] is False
    assert "core_scratch" in report["reason"]


# --------------------------------------------------------------------------- #
# per-cell mode (§3) and the coupling premise (§11)
# --------------------------------------------------------------------------- #


def test_the_modes_are_a_setting_not_a_fork():
    """`pooled` stays reachable, because it is the control arm the per-cell
    mode has to beat before it becomes the default."""
    from vccp.config import Phase2

    assert Phase2().perturbation_mode == phase2_mod.POOLED
    assert {phase2_mod.POOLED, phase2_mod.PERCELL} == {"pooled", "percell"}
    with pytest.raises(ConfigError, match="perturbation_mode"):
        Phase2(perturbation_mode="per-cell").validate()
    with pytest.raises(ConfigError, match="ot_epsilon"):
        Phase2(ot_epsilon=0.0).validate()
    with pytest.raises(ConfigError, match="ot_batch_cells"):
        Phase2(ot_batch_cells=1).validate()
    with pytest.raises(ConfigError, match="ot_targets_per_step"):
        Phase2(ot_targets_per_step=0).validate()
    with pytest.raises(ConfigError, match="ot_iterations"):
        Phase2(ot_iterations=0).validate()


def test_the_no_change_baseline_is_the_control_cell_itself():
    """In per-cell mode "no change" hands back the cell, not a zero vector."""
    control = torch.randn(8, 5, generator=torch.Generator().manual_seed(2))
    truth = control + 0.4

    assert float(phase2_mod._no_change_ratio(control, truth, control)) == pytest.approx(1.0)
    assert float(phase2_mod._no_change_ratio(truth, truth, control)) == pytest.approx(0.0)
    halfway = control + 0.2
    assert float(phase2_mod._no_change_ratio(halfway, truth, control)) == pytest.approx(0.25)


def test_the_coupling_null_spread_is_the_value_that_means_nothing():
    """The reference the measured cost spread is read against."""
    from vccp.diagnostics import coupling

    # Two independent standard-normal cells differ by N(0, 2) per gene, so
    # the per-gene squared difference has mean 2 and variance 8; averaging
    # over G genes divides that variance by G.
    generator = torch.Generator().manual_seed(7)
    for n_genes in (64, 955):
        a = torch.randn(96, n_genes, generator=generator)
        b = torch.randn(96, n_genes, generator=generator)
        cost = ot.squared_cost(a, b)
        measured = float(cost.std()) / float(cost.mean())
        assert measured == pytest.approx(coupling.null_cost_spread(n_genes), rel=0.25)

    # More genes concentrate the distances further, which is the risk.
    assert coupling.null_cost_spread(955) < coupling.null_cost_spread(64)


def test_target_spread_is_zero_when_every_cell_gets_the_same_target():
    """The decisive number: zero means the per-cell loss is the pooled loss."""
    from vccp.diagnostics import coupling

    values = torch.randn(10, 6, generator=torch.Generator().manual_seed(3))
    uniform = ot.sinkhorn_log(torch.zeros(10, 10), 1.0)
    assert coupling.target_spread(uniform, values) == pytest.approx(0.0, abs=1e-6)

    # A hard one-to-one coupling hands each source cell a distinct real cell,
    # so the targets vary exactly as much as the cells do.
    identity = torch.log(torch.eye(10).clamp_min(1e-30))
    assert coupling.target_spread(identity, values) == pytest.approx(1.0, rel=1e-4)


def test_the_verdict_calls_a_degenerate_coupling_what_it_is():
    from vccp.diagnostics import coupling

    def arms(spread, library, residual=0.9):
        return {
            coupling.PERTURBED_ARM: {
                "per_epsilon": {
                    "0.005": {
                        "target_spread": spread,
                        "library_size_pearson": library,
                        "residual_reduction": residual,
                        "effective_partners": 3.2,
                    },
                    "inf": {
                        "target_spread": 0.0,
                        "library_size_pearson": float("nan"),
                        "residual_reduction": 1.0,
                        "effective_partners": 256.0,
                    },
                }
            }
        }

    assert coupling.verdict(arms(0.001, 0.8))[0] == "degenerate"
    assert coupling.verdict(arms(0.5, 0.01))[0] == "pairs-on-noise"
    assert coupling.verdict(arms(0.5, 0.8))[0] == "informative"
    assert coupling.verdict({})[0] == "not-measured"
    # A NaN correlation must not read as "informative" by slipping past a
    # comparison, which is what every NaN comparison does.
    assert coupling.verdict(arms(0.5, float("nan")))[0] == "pairs-on-noise"

    # Pairing that adds more noise than it removes stays "informative" — the
    # loss's minimum is unmoved by noise in its target — but the reading has
    # to say so rather than let the number pass unremarked.
    _, quiet = coupling.verdict(arms(0.5, 0.8, residual=0.9))
    verdict, loud = coupling.verdict(arms(0.5, 0.8, residual=1.35))
    assert verdict == "informative"
    assert "1.35" in loud and "does not reduce the residual" in loud
    assert "does not reduce the residual" not in quiet


# --------------------------------------------------------------------------- #
# the mode sweep's table (PLAN_PERCELL.md §12)
# --------------------------------------------------------------------------- #


def _sweep(rows, scale_built=True):
    scorable = [r for r in rows if r.get("overall") is not None]
    best = max(scorable, key=lambda r: r["overall"])["arm"] if scorable else None
    return {
        "ran": True,
        "epsilons": [None, 0.005],
        "datasets": {
            "rpe1": {
                "n_targets": 14,
                "learned_from": ["K562_gwps"],
                "rows": rows,
                "best": best,
                "scale_built": scale_built,
            }
        },
    }


def test_one_shared_scorer_failure_is_reported_once():
    """Four rows failing identically is a fact about the data, and repeating
    a 500-character scorer message four times buries it."""
    from vccp.rehearsal import stage

    message = (
        "ValueError: the reference panel carries no measurable aggregate effect at "
        "this cell depth, which is a property of the reference rather than of the "
        "submission, so the ratio of sums is undefined and cannot be reported here."
    )
    rows = [{"arm": name, "error": message} for name in ("pooled", "percell:0.005", "floor")]
    text = "\n".join(stage._summarize_mode_sweep(_sweep(rows, scale_built=False)))

    # Said once, not once per row.
    assert text.count("no measurable aggregate effect") == 1, text
    assert "all 3 configurations failed to score" in text
    assert "property of this dataset, not of the configurations" in text
    # And it is folded to a width a terminal can show.
    assert max(len(line) for line in text.splitlines()) < 110


def test_rows_that_fail_differently_are_each_reported():
    """A single arm failing is about that arm, and must not be folded away."""
    from vccp.rehearsal import stage

    rows = [
        {"arm": "pooled", "overall": -0.1, "scaled": {"pds_cosine": 0.0}},
        {"arm": "percell:0.005", "error": "ValueError: something specific to this arm"},
    ]
    text = "\n".join(stage._summarize_mode_sweep(_sweep(rows)))
    assert "something specific to this arm" in text
    assert "failed to score, identically" not in text
    assert "pooled" in text


def test_the_table_names_the_best_arm_and_keeps_the_floor_in_view():
    from vccp.rehearsal import stage

    rows = [
        {"arm": "floor", "overall": 0.0, "scaled": {}},
        {"arm": "pooled", "overall": -0.13, "scaled": {"pds_cosine": 0.0}},
        {"arm": "percell:0.02", "overall": 0.08, "scaled": {"pds_cosine": 0.2}},
    ]
    text = "\n".join(stage._summarize_mode_sweep(_sweep(rows)))
    assert "best: percell:0.02" in text
    # The floor is what "best" has to be read against: an arm can be the best
    # of a bad field.
    assert "floor" in text and "has not earned a submission" in text


def test_an_off_sweep_prints_nothing():
    from vccp.rehearsal import stage

    assert stage._summarize_mode_sweep({"ran": False, "reason": "off"}) == []
    assert stage._summarize_mode_sweep({}) == []


# --------------------------------------------------------------------------- #
# the cross-assay variant (PLAN_PERCELL.md §7.7)
# --------------------------------------------------------------------------- #


def test_the_cross_assay_scope_hides_every_screens_responses():
    """Not just the scored one: the arm has to learn from LINCS alone.

    A prior block built on any single-cell screen's responses would be the
    answer arriving by another route, so hiding only the held-out screen
    would leak the very thing the variant is measuring.
    """
    from vccp.priors.scope import cross_assay_scope, cross_context_scope

    screens = ["K562_gwps", "rpe1"]
    scope = cross_assay_scope(screens, ["ADNP", "BYSL"])

    assert scope.exclude_response_contexts == frozenset(screens)
    assert scope.exclude_targets == frozenset({"ADNP", "BYSL"})
    # Control cells stay visible — that asymmetry is what "adapt using only
    # its controls" means.
    assert scope.exclude_contexts == frozenset()

    # And it is a different scope from variant 2's, which hides one screen.
    narrow = cross_context_scope("rpe1", ["ADNP", "BYSL"])
    assert scope.exclude_response_contexts > narrow.exclude_response_contexts
    assert scope.describe()["cache_key"] != narrow.describe()["cache_key"]


def test_the_cross_assay_scope_is_stable_under_screen_order():
    """Two orderings are the same restriction and must not rebuild priors."""
    from vccp.priors.scope import cross_assay_scope

    a = cross_assay_scope(["rpe1", "K562_gwps"], ["ADNP"])
    b = cross_assay_scope(["K562_gwps", "rpe1"], ["ADNP"])
    assert a.label == b.label
    assert a.describe()["cache_key"] == b.describe()["cache_key"]


def test_the_variant_names_its_second_method_arm():
    """The same weights asked as the modality they were trained on."""
    from vccp.rehearsal import variants

    assert variants.SOURCE_MODALITY == "method_source_modality"
    assert variants.SOURCE_MODALITY != common_module().METHOD


def common_module():
    from vccp.rehearsal import common

    return common


def test_the_cross_assay_config_is_off_and_validated():
    from vccp.config import ConfigError, Rehearsal

    assert Rehearsal().cross_assay is False, "evidence, not score: never a deadline cost"
    assert Rehearsal().cross_assay_contexts == 1
    with pytest.raises(ConfigError, match="cross_assay_contexts"):
        Rehearsal(cross_assay_contexts=0).validate()


def test_the_summary_prints_every_arm_a_variant_produced():
    """An arm that is measured and not printed is measured for nothing."""
    from vccp.rehearsal import common, stage, variants

    entry = {
        "variant": "cross_assay",
        "context": "rpe1",
        "n_targets": 9,
        "modalities": {common.METHOD: "crispri", variants.SOURCE_MODALITY: "crispr_ko"},
        "arms": {
            arm: {common.END_TO_END: {"scored": {"pds_cosine": value}}}
            for arm, value in (
                (common.METHOD, 0.51),
                (common.UPPER_BOUND, 0.62),
                (common.FLOOR, 0.50),
                (variants.SOURCE_MODALITY, 0.55),
            )
        },
    }
    text = stage.summarize({
        "scorer": {"available": True},
        "variants": [entry],
        "calibration": {"settings": {"confidence_threshold": 0.0, "effect_scale": 1.0}},
        "scale": {"built_on": [], "attempted_on": [], "datasets": {}},
        "note": "",
    })
    assert variants.SOURCE_MODALITY in text
    assert "0.5500" in text
    # And the reading that makes the two method arms worth comparing.
    assert "knockout-specific type offset" in text


# --------------------------------------------------------------------------- #
# what Phase 2 inherits from Phase 1 (DECISIONS.md D80)
# --------------------------------------------------------------------------- #
def test_warm_start_has_three_modes_and_rejects_anything_else():
    from vccp.config import Phase2

    assert Phase2().warm_start == phase2_mod.FULL, "three phases is what §4 asks for"
    for mode in (phase2_mod.FULL, phase2_mod.WITHOUT_DELTA, phase2_mod.NONE):
        Phase2(warm_start=mode).validate()
    # The key was a bool for one day; a stale config must fail loudly.
    with pytest.raises(ConfigError, match="warm_start must be"):
        Phase2(warm_start=True).validate()
    with pytest.raises(ConfigError, match="warm_start must be"):
        Phase2(warm_start="partial").validate()


def test_resetting_delta_touches_delta_and_nothing_else(typed_model):
    """`delta` is 4.74M of 5.65M trainable parameters, and Phase 1 trains it
    on the 955 panel genes alone — so a full warm start hands Phase 2 an
    embedding table whose two halves sit on different footings."""
    model = typed_model
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)

    def deltas():
        return {
            name: float(p.detach().abs().sum())
            for name, p in model.named_parameters()
            if name.startswith("vocabulary.") and "delta" in name
        }

    def others():
        return {
            name: float(p.detach().abs().sum())
            for name, p in model.named_parameters()
            if not (name.startswith("vocabulary.") and "delta" in name)
        }

    before_delta, before_other = deltas(), others()
    assert before_delta and all(v > 0 for v in before_delta.values())

    n_reset = phase2_mod.reset_delta(model)

    assert n_reset == len(before_delta)
    assert all(v == 0.0 for v in deltas().values())
    assert others() == before_other, "only the per-gene vectors are reset"

    # And it is what §4.1 says delta starts at, so the reset is a return to
    # the specified initial condition rather than an invention.
    assert all(v == 0.0 for v in deltas().values())
