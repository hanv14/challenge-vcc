"""The rehearsal's leakage rule, verified rather than asserted.

`priors/checks.json` states which blocks each rehearsal variant rebuilds,
drops or reuses. That statement is only worth having if it matches what a
real rebuild under the same scope actually does, so these tests rebuild the
priors under each variant's scope and compare.

CLAUDE.md §4.1: a prior built from a held-out target's own response, reaching
the rehearsal that is supposed to predict that target, would make the
rehearsal measure memory instead of generalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from vccp.data import reference
from vccp.paths import DataPaths
from vccp.priors.blocks import BlockInputs
from vccp.priors.build import build_priors
from vccp.priors.leakage import (
    DROPPED,
    REBUILT,
    REUSED,
    classify_block,
    plan_for_scope,
    rehearsal_rebuild_plan,
    rehearsal_scopes,
)
from vccp.priors.scope import (
    CHALLENGE_CONTROL_SOURCE,
    FULL_SCOPE,
    PriorScope,
    controls_only_scope,
    cross_context_scope,
    unseen_genes_scope,
)


@pytest.fixture
def replogle_contexts(session_cfg):
    return sorted(DataPaths(session_cfg).discover_phase2_contexts())


def inputs_for(cfg, scope):
    paths = DataPaths(cfg)
    return BlockInputs(
        cfg=cfg,
        paths=paths,
        axis=reference.load_gene_names(paths.gene_names),
        scope=scope,
        rng=np.random.default_rng(cfg.seed),
    )


def covered_symbols(result, axis):
    return {axis[i] for i in np.flatnonzero(result.covered)}


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def test_the_plan_covers_every_variant_and_context(built_priors, replogle_contexts):
    plan = rehearsal_rebuild_plan(built_priors, replogle_contexts)

    expected = {
        f"{variant}:{context}"
        for variant in ("controls_only", "cross_context", "unseen_genes")
        for context in replogle_contexts
    }
    assert set(plan["variants"]) == expected

    built = set(built_priors.coverage.loc[built_priors.coverage["built"], "block"])
    for entry in plan["variants"].values():
        assert set(entry["blocks"]) == built


def test_controls_only_rebuilds_every_response_block(built_priors, replogle_contexts):
    """Variant 1 holds out targets, so every block that read a response has
    to be rebuilt; the control-derived blocks are untouched."""
    scope = controls_only_scope(["SOME_TARGET"], replogle_contexts[0])
    verdicts = plan_for_scope(built_priors, scope)["blocks"]

    coverage = built_priors.coverage.set_index("block")
    for name, verdict in verdicts.items():
        expected = REBUILT if coverage.loc[name, "reads_responses"] else REUSED
        assert verdict == expected, name


def test_cross_context_hides_responses_but_keeps_controls(built_priors, replogle_contexts):
    """Variant 2's whole design: learn in one screen, adapt to the other
    **using only its controls**. Its responses must go and its control
    co-expression must stay."""
    held_out = replogle_contexts[0]
    verdicts = plan_for_scope(built_priors, cross_context_scope(held_out, ["T"]))["blocks"]

    assert verdicts[f"pert_phenotype_replogle:{held_out}"] == DROPPED
    assert verdicts[f"replogle_coexpr:{held_out}"] == REUSED

    for other in replogle_contexts[1:]:
        assert verdicts[f"pert_phenotype_replogle:{other}"] == REBUILT
        assert verdicts[f"replogle_coexpr:{other}"] == REUSED


def test_unseen_genes_leaves_only_the_permitted_source(built_priors, replogle_contexts):
    verdicts = plan_for_scope(
        built_priors, unseen_genes_scope(["G"], replogle_contexts[0])
    )["blocks"]

    assert verdicts[CHALLENGE_CONTROL_SOURCE] == REUSED
    for name, verdict in verdicts.items():
        if name != CHALLENGE_CONTROL_SOURCE:
            assert verdict == REBUILT, name


def test_the_full_scope_rebuilds_nothing(built_priors):
    verdicts = plan_for_scope(built_priors, FULL_SCOPE)["blocks"]
    assert set(verdicts.values()) == {REUSED}


def test_classification_does_not_depend_on_which_targets_are_held_out():
    """Why the plan can be written before the rehearsal chooses its sets."""
    a = classify_block("b", "K", True, PriorScope(exclude_targets=frozenset({"X"})))
    b = classify_block("b", "K", True, PriorScope(exclude_targets=frozenset({"Y", "Z"})))
    assert a == b == REBUILT


# --------------------------------------------------------------------------- #
# the plan against real rebuilds
# --------------------------------------------------------------------------- #
def test_a_real_rebuild_matches_the_plan(session_cfg, built_priors, replogle_contexts):
    """The plan is checked against what building under the scope really does.

    Held-out targets are taken from the blocks' own coverage, so the scope
    hides something that was genuinely there.
    """
    axis = reference.load_gene_names(DataPaths(session_cfg).gene_names)
    target_role_covered = [
        axis[i]
        for i in np.flatnonzero(
            np.asarray(
                [
                    built_priors.priors["target"][:, spec.flag_column] == 0.0
                    for spec in built_priors.layouts["target"].blocks
                    if spec.name.startswith("pert_phenotype_replogle")
                ]
            ).any(axis=0)
        )
    ]
    held_out = set(target_role_covered[:5])
    assert held_out, "expected some targets to be covered by a phenotype block"

    scope = controls_only_scope(held_out, replogle_contexts[0])
    predicted = plan_for_scope(built_priors, scope)["blocks"]
    rebuilt = build_priors(session_cfg, scope)

    built_now = set(rebuilt.coverage.loc[rebuilt.coverage["built"], "block"])
    for name, verdict in predicted.items():
        if verdict == DROPPED:
            assert name not in built_now, f"{name} should have been dropped"
        else:
            assert name in built_now, f"{name} should still be built"

    # And the held-out targets really are gone from the response blocks.
    for spec in rebuilt.layouts["target"].blocks:
        if not spec.name.startswith("pert_phenotype"):
            continue
        covered = rebuilt.priors["target"][:, spec.flag_column] == 0.0
        still_there = {axis[i] for i in np.flatnonzero(covered)} & held_out
        assert not still_there, f"{spec.name} still covers {still_there}"


def test_a_cross_context_rebuild_drops_only_that_screens_responses(
    session_cfg, replogle_contexts
):
    held_out = replogle_contexts[0]
    rebuilt = build_priors(session_cfg, cross_context_scope(held_out, ["ANY"]))
    built_now = set(rebuilt.coverage.loc[rebuilt.coverage["built"], "block"])

    assert f"pert_phenotype_replogle:{held_out}" not in built_now
    assert f"replogle_coexpr:{held_out}" in built_now


def test_an_unseen_genes_rebuild_hides_them_everywhere_but_one_source(
    session_cfg, replogle_contexts
):
    axis = reference.load_gene_names(DataPaths(session_cfg).gene_names)
    hidden = set(axis[:30])

    scope = unseen_genes_scope(hidden, replogle_contexts[0])
    rebuilt = build_priors(session_cfg, scope)

    for role, layout in rebuilt.layouts.items():
        for spec in layout.blocks:
            covered = rebuilt.priors[role][:, spec.flag_column] == 0.0
            overlap = {axis[i] for i in np.flatnonzero(covered)} & hidden
            if spec.name == CHALLENGE_CONTROL_SOURCE:
                assert overlap == hidden, "the permitted source must still see them"
            else:
                assert not overlap, f"{spec.name} leaks {len(overlap)} hidden gene(s)"


def test_rebuilt_priors_record_their_scope(session_cfg, replogle_contexts):
    scope = cross_context_scope(replogle_contexts[0], ["T"])
    rebuilt = build_priors(session_cfg, scope)

    assert rebuilt.scope.label == scope.label
    assert rebuilt.scope.cache_key() == scope.cache_key()
    assert rebuilt.build_meta()["scope"]["excluded_response_contexts"] == [
        replogle_contexts[0]
    ]


def test_scopes_are_distinct_per_variant_and_context(replogle_contexts):
    scopes = rehearsal_scopes(replogle_contexts)
    keys = {scope.cache_key() for scope in scopes.values()}
    assert len(keys) == len(scopes)
