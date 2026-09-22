"""The gene vocabulary as a module (CLAUDE.md §4.1, checklist items 1 and 3).

    embedding(g) = W . prior(g) + delta(g)

`prior(g)` is fixed — it is what `vccp/priors/` built. `W` is one learned
projection shared by every gene, so a gene the model has never trained on
still gets a sensible embedding from its prior alone. `delta(g)` is a free
per-gene vector, initialized to zero and L2-regularized, so a gene the model
*has* seen can move away from what its prior said.

There is one `W` and one `delta` per role. The two roles are separate
vocabularies over the same genes: what a gene does as a measured feature and
what its knockdown does are different questions, and the priors that answer
them are different widths.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

FEATURE = "feature"
TARGET = "target"


class GeneVocabulary(nn.Module):
    """Embeddings for every challenge gene, in both roles."""

    def __init__(
        self,
        priors: dict[str, np.ndarray],
        embedding_dim: int,
        *,
        delta_l2: float = 0.0,
    ) -> None:
        super().__init__()
        if not priors:
            raise ValueError("at least one role's prior is required")

        widths = {role: int(matrix.shape[1]) for role, matrix in priors.items()}
        n_genes = {int(matrix.shape[0]) for matrix in priors.values()}
        if len(n_genes) != 1:
            raise ValueError(f"the role priors disagree on the gene count: {n_genes}")

        self.roles = tuple(sorted(priors))
        self.n_genes = n_genes.pop()
        self.embedding_dim = int(embedding_dim)
        self.prior_widths = widths
        self.delta_l2 = float(delta_l2)

        self.projections = nn.ModuleDict()
        self.deltas = nn.ParameterDict()
        for role, matrix in priors.items():
            self.register_buffer(
                f"prior_{role}", torch.as_tensor(np.asarray(matrix), dtype=torch.float32)
            )
            self.projections[role] = nn.Linear(widths[role], self.embedding_dim, bias=False)
            # Starts at exactly zero: before training, a gene is what its
            # prior says it is and nothing more.
            self.deltas[role] = nn.Parameter(torch.zeros(self.n_genes, self.embedding_dim))

    def prior(self, role: str) -> torch.Tensor:
        self._check_role(role)
        return getattr(self, f"prior_{role}")

    def delta(self, role: str) -> torch.Tensor:
        self._check_role(role)
        return self.deltas[role]

    def forward(self, gene_idx: torch.Tensor | None = None, role: str = FEATURE) -> torch.Tensor:
        """Embeddings for `gene_idx` (all genes when None), in `role`."""
        self._check_role(role)
        prior = self.prior(role)
        delta = self.deltas[role]
        if gene_idx is not None:
            index = torch.as_tensor(gene_idx, dtype=torch.long, device=prior.device)
            prior = prior.index_select(0, index)
            delta = delta.index_select(0, index)
        return self.projections[role](prior) + delta

    def delta_penalty(self) -> torch.Tensor:
        """The L2 pull on `delta`, ready to add to a loss."""
        device = self.prior(self.roles[0]).device
        if self.delta_l2 == 0.0:
            return torch.zeros((), device=device)
        total = sum((delta**2).sum() for delta in self.deltas.values())
        return self.delta_l2 * total

    def _check_role(self, role: str) -> None:
        if role not in self.roles:
            raise KeyError(f"unknown role {role!r}; this vocabulary has {list(self.roles)}")

    def extra_repr(self) -> str:
        widths = ", ".join(f"{role}:{width}" for role, width in sorted(self.prior_widths.items()))
        return f"n_genes={self.n_genes}, embedding_dim={self.embedding_dim}, priors=({widths})"


def build_vocabulary(prior_features, embedding_dim: int, *, delta_l2: float = 0.0) -> GeneVocabulary:
    """Build the vocabulary from what `vccp.priors.build` produced."""
    return GeneVocabulary(prior_features.priors, embedding_dim, delta_l2=delta_l2)
