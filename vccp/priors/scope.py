"""What a set of priors is allowed to have seen.

The rehearsal (CLAUDE.md §4.6) hides targets, contexts or genes and then
asks the method to predict them. Any prior built from perturbation responses
has to be rebuilt without what is hidden, or the rehearsal measures the
pipeline's memory instead of its generalization (§4.1, leakage rule).

`PriorScope` is that restriction, passed to every block. Two kinds of
context exclusion are needed, because the rehearsal's own design depends on
the difference: the cross-context variant learns perturbation behaviour in
one screen and adapts to the other **using only its controls**, so the
held-out screen's responses must be hidden while its control cells stay
visible. `exclude_contexts` hides a source entirely; `exclude_response_contexts`
hides only what its perturbations did.

The three constructors at the bottom are the scopes the rehearsal's three
variants run under, defined here so that what the priors promise and what the
rehearsal asks for cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PriorScope:
    """A restriction on what the priors may be built from.

    `exclude_targets`
        Perturbation targets held out. Blocks reading responses must drop
        these rows entirely.
    `exclude_contexts`
        Sources hidden completely, controls included.
    `exclude_response_contexts`
        Sources whose **perturbation responses** are hidden while their
        control cells stay visible.
    `restricted_genes`
        Genes that are being pretended unmeasured (§4.6 variant 3). Only
        `restricted_genes_source` may inform them; every other block marks
        them uncovered, so they fall back to `delta` alone.
    `restricted_genes_source`
        The one block allowed to inform `restricted_genes` — in the
        rehearsal, the rehearsal context's controls.
    """

    exclude_targets: frozenset[str] = field(default_factory=frozenset)
    exclude_contexts: frozenset[str] = field(default_factory=frozenset)
    exclude_response_contexts: frozenset[str] = field(default_factory=frozenset)
    restricted_genes: frozenset[str] = field(default_factory=frozenset)
    restricted_genes_source: str | None = None
    #: Free-form label so two scopes that differ only in intent get
    #: different cache keys (e.g. which rehearsal fold this is).
    label: str = "full"

    @property
    def is_unrestricted(self) -> bool:
        return not (
            self.exclude_targets
            or self.exclude_contexts
            or self.exclude_response_contexts
            or self.restricted_genes
        )

    def allows_context(self, context: str) -> bool:
        """Whether this source may be read at all."""
        return context not in self.exclude_contexts

    def allows_responses_from(self, context: str) -> bool:
        """Whether this source's perturbation responses may be read."""
        return context not in self.exclude_contexts and (
            context not in self.exclude_response_contexts
        )

    def allows_target(self, target: str) -> bool:
        return target not in self.exclude_targets

    def visible_targets(self, targets: Iterable[str]) -> list[str]:
        return [t for t in targets if t not in self.exclude_targets]

    def gene_is_visible_to(self, gene: str, source: str) -> bool:
        """Whether `source` may inform `gene`.

        Only the restricted genes are ever hidden, and only from sources
        other than the one allowed to see them.
        """
        if gene not in self.restricted_genes:
            return True
        return source == self.restricted_genes_source

    def cache_key(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:16]

    def as_dict(self) -> dict:
        return {
            "exclude_targets": sorted(self.exclude_targets),
            "exclude_contexts": sorted(self.exclude_contexts),
            "exclude_response_contexts": sorted(self.exclude_response_contexts),
            "restricted_genes": sorted(self.restricted_genes),
            "restricted_genes_source": self.restricted_genes_source,
            "label": self.label,
        }

    def describe(self) -> dict:
        return {
            "label": self.label,
            "n_excluded_targets": len(self.exclude_targets),
            "excluded_contexts": sorted(self.exclude_contexts),
            "excluded_response_contexts": sorted(self.exclude_response_contexts),
            "n_restricted_genes": len(self.restricted_genes),
            "restricted_genes_source": self.restricted_genes_source,
            "cache_key": self.cache_key(),
        }


#: What the submission is built from: nothing hidden.
FULL_SCOPE = PriorScope()

#: The block allowed to see the rehearsal's "unseen" genes. They are
#: pretended unmeasured everywhere except the challenge controls, which is
#: the position the ~10,000 genes measured nowhere else are really in.
CHALLENGE_CONTROL_SOURCE = "challenge_coexpr"


def controls_only_scope(held_out_targets: Iterable[str], context: str) -> PriorScope:
    """Rehearsal variant 1 — fit the mapping on one screen's controls only.

    The held-out targets' own responses are hidden everywhere; the screen's
    controls stay visible, because fitting on them is the point.
    """
    return PriorScope(
        exclude_targets=frozenset(held_out_targets),
        label=f"rehearsal:controls_only:{context}",
    )


def cross_context_scope(held_out_context: str, held_out_targets: Iterable[str]) -> PriorScope:
    """Rehearsal variant 2 — learn in one screen, adapt to the other on its
    controls alone.

    The held-out screen's *responses* are hidden while its control cells stay
    visible; that asymmetry is the variant.
    """
    return PriorScope(
        exclude_targets=frozenset(held_out_targets),
        exclude_response_contexts=frozenset({held_out_context}),
        label=f"rehearsal:cross_context:{held_out_context}",
    )


def cross_assay_scope(
    screens: Iterable[str], held_out_targets: Iterable[str]
) -> PriorScope:
    """The cross-assay variant — learn from LINCS, adapt to a Replogle screen.

    Every Replogle screen's *responses* are hidden, not just the scored one:
    the arm has to learn what a perturbation does from LINCS alone, so a
    prior built on any single-cell screen's responses would be the answer
    arriving by another route. Their control cells stay visible, which is
    what "adapt using only its controls" means, and LINCS's own blocks stay —
    it is the training assay here.
    """
    names = sorted({str(name) for name in screens})
    return PriorScope(
        exclude_targets=frozenset(held_out_targets),
        exclude_response_contexts=frozenset(names),
        label=f"rehearsal:cross_assay:{'+'.join(names)}",
    )


def unseen_genes_scope(
    hidden_genes: Iterable[str],
    context: str,
    source: str = CHALLENGE_CONTROL_SOURCE,
) -> PriorScope:
    """Rehearsal variant 3 — hide genes from everything but one source."""
    return PriorScope(
        restricted_genes=frozenset(hidden_genes),
        restricted_genes_source=source,
        label=f"rehearsal:unseen_genes:{context}",
    )
