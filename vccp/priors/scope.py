"""What a set of priors is allowed to have seen.

The rehearsal (CLAUDE.md §4.6) hides targets, contexts or genes and then
asks the method to predict them. Any prior built from perturbation responses
has to be rebuilt without what is hidden, or the rehearsal measures the
pipeline's memory instead of its generalization (§4.1, leakage rule).

`PriorScope` is that restriction, passed to every block. A block declares
whether it reads perturbation responses (`reads_responses`) and which source
it reads, and the scope decides what it may use. The default scope hides
nothing and is what the submission is built from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PriorScope:
    """A restriction on what the priors may be built from.

    `exclude_targets`
        Perturbation targets held out. Blocks reading responses must drop
        these rows entirely.
    `exclude_contexts`
        Sources held out by name (a Replogle context, a challenge context).
        Blocks reading them are skipped.
    `restricted_genes`
        Genes that are being pretended unmeasured (§4.6 variant 3). Only
        `restricted_genes_source` may inform them; every other block marks
        them uncovered, so they fall back to `delta` alone.
    `restricted_genes_source`
        The one source allowed to inform `restricted_genes` — in the
        rehearsal, the rehearsal context's controls.
    """

    exclude_targets: frozenset[str] = field(default_factory=frozenset)
    exclude_contexts: frozenset[str] = field(default_factory=frozenset)
    restricted_genes: frozenset[str] = field(default_factory=frozenset)
    restricted_genes_source: str | None = None
    #: Free-form label so two scopes that differ only in intent get
    #: different cache keys (e.g. which rehearsal fold this is).
    label: str = "full"

    @property
    def is_unrestricted(self) -> bool:
        return not (self.exclude_targets or self.exclude_contexts or self.restricted_genes)

    def allows_context(self, context: str) -> bool:
        return context not in self.exclude_contexts

    def allows_target(self, target: str) -> bool:
        return target not in self.exclude_targets

    def visible_targets(self, targets) -> list[str]:
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
        payload = json.dumps(
            {
                "exclude_targets": sorted(self.exclude_targets),
                "exclude_contexts": sorted(self.exclude_contexts),
                "restricted_genes": sorted(self.restricted_genes),
                "restricted_genes_source": self.restricted_genes_source,
                "label": self.label,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def describe(self) -> dict:
        return {
            "label": self.label,
            "n_excluded_targets": len(self.exclude_targets),
            "excluded_contexts": sorted(self.exclude_contexts),
            "n_restricted_genes": len(self.restricted_genes),
            "restricted_genes_source": self.restricted_genes_source,
            "cache_key": self.cache_key(),
        }


#: What the submission is built from: nothing hidden.
FULL_SCOPE = PriorScope()
