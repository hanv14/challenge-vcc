"""Which prior blocks each rehearsal variant has to rebuild.

CLAUDE.md §4.1's leakage rule: any prior built from perturbation responses
must be rebuilt without held-out targets and held-out contexts when used in
the rehearsal, and the "unseen genes" variant may inform those genes only
from the rehearsal context's controls.

This module states that plan explicitly, per variant, so it lands in
`priors/checks.json` and can be read before the rehearsal runs rather than
inferred from it afterwards. The classification depends only on *what kind*
of thing a scope hides, not on which particular targets or genes it picks —
so the plan is exact even though the rehearsal chooses its held-out sets
later.

`tests/test_priors.py` rebuilds under each scope and asserts the result
matches what this predicts, which is what keeps the plan honest.
"""

from __future__ import annotations

from typing import Any

from .build import PriorFeatures
from .scope import (
    CHALLENGE_CONTROL_SOURCE,
    PriorScope,
    controls_only_scope,
    cross_context_scope,
    unseen_genes_scope,
)

DROPPED = "dropped"
REBUILT = "rebuilt"
REUSED = "reused"

#: Stand-ins for the sets the rehearsal will choose. Only their emptiness
#: matters to the classification, which is why the plan can be stated now.
_SOME_TARGETS = ("<held-out targets>",)
_SOME_GENES = ("<hidden genes>",)


def classify_block(
    name: str, source_context: str | None, reads_responses: bool, scope: PriorScope
) -> str:
    """What happens to one block under `scope`."""
    if source_context is not None and not scope.allows_context(source_context):
        return DROPPED
    if reads_responses:
        if source_context is not None and not scope.allows_responses_from(source_context):
            return DROPPED
        if scope.exclude_targets:
            return REBUILT
    if scope.restricted_genes and name != scope.restricted_genes_source:
        return REBUILT
    return REUSED


def plan_for_scope(features: PriorFeatures, scope: PriorScope) -> dict[str, Any]:
    """Per-block verdicts for one scope, from the full-scope build."""
    built = features.coverage[features.coverage["built"]]
    verdicts = {}
    for _, row in built.iterrows():
        verdicts[str(row["block"])] = classify_block(
            str(row["block"]),
            row["source_context"] if row["source_context"] is not None else None,
            bool(row["reads_responses"]),
            scope,
        )

    return {
        "scope": scope.describe(),
        "blocks": verdicts,
        "n_rebuilt": sum(v == REBUILT for v in verdicts.values()),
        "n_dropped": sum(v == DROPPED for v in verdicts.values()),
        "n_reused": sum(v == REUSED for v in verdicts.values()),
    }


def rehearsal_scopes(replogle_contexts: list[str]) -> dict[str, PriorScope]:
    """The scope each rehearsal variant runs under, per Replogle context.

    Variant 1 holds out targets and fits on the context's controls.
    Variant 2 additionally hides that context's *responses* while keeping its
    controls, which is what "adapt using only its controls" means.
    Variant 3 hides a gene set from everything but the challenge controls.
    """
    scopes: dict[str, PriorScope] = {}
    for context in sorted(replogle_contexts):
        scopes[f"controls_only:{context}"] = controls_only_scope(_SOME_TARGETS, context)
        scopes[f"cross_context:{context}"] = cross_context_scope(context, _SOME_TARGETS)
        scopes[f"unseen_genes:{context}"] = unseen_genes_scope(_SOME_GENES, context)
    return scopes


def rehearsal_rebuild_plan(features: PriorFeatures, replogle_contexts: list[str]) -> dict[str, Any]:
    """The whole plan, for `priors/checks.json`."""
    scopes = rehearsal_scopes(replogle_contexts)
    return {
        "description": "which prior blocks each rehearsal variant rebuilds, drops or "
        "reuses, so that no prior built from a held-out response reaches the "
        "rehearsal (CLAUDE.md §4.1)",
        "note": "the verdicts depend on what kind of thing a scope hides, not on which "
        "targets or genes it picks, so they hold for whatever the rehearsal chooses",
        "unseen_genes_permitted_source": CHALLENGE_CONTROL_SOURCE,
        "variants": {name: plan_for_scope(features, scope) for name, scope in scopes.items()},
    }
