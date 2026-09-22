"""The §4.1 prior checks: do the embeddings know any biology?

These are diagnostics rather than gates, so the tests assert that the checks
*run, report, and would notice* — not that mini_data's numbers clear a bar.
The one substantive assertion is that a deliberately structureless prior
scores worse than the real one, which is what makes the check worth having.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vccp.data import reference
from vccp.paths import DataPaths
from vccp.priors.build import FEATURE, TARGET
from vccp.priors.checks import (
    MIN_FAMILY_SIZE,
    matching_groups,
    neighbour_check,
    response_correlation_check,
    run_checks,
    summarize,
)


def test_configured_families_are_found_in_hgnc(session_cfg):
    axis = reference.load_gene_names(DataPaths(session_cfg).gene_names)
    groups = matching_groups(session_cfg, axis)

    assert groups, "no gene group matched the configured patterns"
    for name, (pattern, positions) in groups.items():
        assert pattern.lower() in name.lower()
        assert positions == sorted(set(positions))


def test_groups_are_checked_individually_not_pooled(session_cfg):
    """Complex I and complex IV match the same pattern but are different
    machines; pooling them would ask a question with no right answer."""
    axis = reference.load_gene_names(DataPaths(session_cfg).gene_names)
    groups = matching_groups(session_cfg, axis)

    by_pattern: dict[str, int] = {}
    for _, (pattern, _) in groups.items():
        by_pattern[pattern] = by_pattern.get(pattern, 0) + 1
    assert max(by_pattern.values()) > 1, "expected at least one pattern to match several groups"


def test_neighbour_check_reports_every_family_with_a_status(session_cfg, built_priors):
    result = neighbour_check(session_cfg, built_priors)

    assert result["role"] == FEATURE
    assert result["families"]
    for entry in result["families"]:
        assert entry["status"] in ("ok", "skipped")
        if entry["status"] == "skipped":
            assert entry["reason"]
            assert entry["n_members"] < MIN_FAMILY_SIZE
        else:
            assert 0.0 <= entry["observed_same_family_fraction"] <= 1.0
            assert entry["expected_by_chance"] > 0
            assert entry["enrichment"] is not None


def test_complex_members_are_each_others_neighbours(session_cfg, built_priors):
    """The substantive check of CLAUDE.md §4.1. Scored groups should be
    enriched over chance; mini_data's complexes are small, so this asserts
    the majority rather than all of them."""
    scored = [
        entry
        for entry in neighbour_check(session_cfg, built_priors)["families"]
        if entry["status"] == "ok"
    ]
    assert scored, "no family was large enough to score"
    assert sum(entry["enriched"] for entry in scored) > len(scored) / 2


def test_the_neighbour_check_would_notice_a_structureless_prior(session_cfg, built_priors):
    """A prior with no biology in it must not pass the check that the real
    one passes, or the check is measuring nothing."""
    rng = np.random.default_rng(0)
    noise = dataclasses.replace(
        built_priors,
        priors={
            FEATURE: rng.standard_normal(built_priors.priors[FEATURE].shape).astype(np.float32),
            TARGET: built_priors.priors[TARGET],
        },
    )

    def mean_enrichment(features):
        scored = [
            e
            for e in neighbour_check(session_cfg, features)["families"]
            if e["status"] == "ok"
        ]
        return float(np.mean([e["enrichment"] for e in scored]))

    assert mean_enrichment(noise) < mean_enrichment(built_priors) / 10


def test_response_correlation_is_reported_per_context(session_cfg, built_priors):
    result = response_correlation_check(session_cfg, built_priors)

    assert result["role"] == TARGET
    contexts = {entry["context"] for entry in result["contexts"]}
    assert contexts == set(DataPaths(session_cfg).discover_phase2_contexts())

    for entry in result["contexts"]:
        assert entry["status"] in ("ok", "skipped")
        if entry["status"] == "ok":
            assert -1.0 <= entry["spearman"] <= 1.0
            assert entry["n_pairs"] > 0


def test_similar_target_embeddings_mean_similar_responses(session_cfg, built_priors):
    """Positive, because the target-role prior is partly built from those
    responses — this verifies the wiring, not a held-out claim."""
    scored = [
        entry
        for entry in response_correlation_check(session_cfg, built_priors)["contexts"]
        if entry["status"] == "ok"
    ]
    assert scored
    assert all(entry["positive"] for entry in scored)


def test_response_check_caps_the_number_of_targets(session_cfg, built_priors):
    """The check is over pairs, so it must not go quadratic on the server."""
    capped = dataclasses.replace(
        session_cfg, priors=dataclasses.replace(session_cfg.priors, check_max_targets=5)
    )
    for entry in response_correlation_check(capped, built_priors)["contexts"]:
        if entry["status"] == "ok":
            assert entry["n_targets"] <= 5


def test_run_checks_returns_both_and_summarizes(session_cfg, built_priors):
    checks = run_checks(session_cfg, built_priors)
    assert set(checks) == {"neighbour_check", "response_correlation_check"}

    table = summarize(checks)
    assert not table.empty
    assert set(table["check"]) <= {"neighbours", "response_correlation"}
