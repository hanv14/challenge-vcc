"""The shared gene-token model (CLAUDE.md §4.2, checklist item 5).

One core, used by all three phases. A cell or a bulk profile is a **set of
tokens**, one per input gene, each the gene's embedding plus an encoding of
its value. That is what lets the three phases use different gene sets without
different networks: LINCS measures the panel, a Replogle screen measures a few
thousand genes, the challenge measures all of them, and to the core they are
all just sets of tokens over a shared vocabulary.

    tokens ──► Perceiver latents (fixed count) ──► query any output gene

The latent bottleneck is why cost does not grow with the number of output
genes, and the gene-query decoder is why any gene can be asked for — including
one the model has never been trained to output, since its query is built from
its prior.

Conditioning:

* **Perturbation** — one extra token, the target's *target-role* embedding
  plus a learned knockdown-type embedding. Absent for an unperturbed profile.
* **Context** — a vector computed from that context's control profile, applied
  to the latents by FiLM.

Adapters wrap every linear layer inside the core (§4.3), so a phase or a
context can specialize it without the core's own weights moving.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import torch
from torch import nn

from .adapters import add_adapter, adapter_parameters, use_adapters, wrap_linears
from .blocks import FiLM, PerceiverBlock
from .embedding import FEATURE, TARGET, GeneVocabulary

#: Value channels a token carries. Channel 0 is the profile the phase feeds
#: in; channel 1 is an auxiliary signal — Phase 1's two-step uses it to hand
#: the predicted signature back in alongside the control (§4.4).
N_VALUE_CHANNELS = 2


class ValueEncoder(nn.Module):
    """A small MLP turning a gene's value(s) into something to add to its
    embedding (CLAUDE.md §4.2)."""

    def __init__(self, dim: int, n_channels: int = N_VALUE_CHANNELS) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.net = nn.Sequential(
            nn.Linear(n_channels, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """`values` is (B, G, C) or (B, G) for a single channel."""
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        if values.shape[-1] < self.n_channels:
            pad = self.n_channels - values.shape[-1]
            values = torch.cat(
                [values, values.new_zeros(*values.shape[:-1], pad)], dim=-1
            )
        return self.net(values)


class ContextEncoder(nn.Module):
    """A context's identity, from its control profile.

    The challenge's contexts are unnamed cell lines whose only description is
    the control cells they come with, so this is the only handle the model
    has on "which context is this" (CLAUDE.md §4.2, §4.7).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, pooled_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_tokens)


class GeneQueryDecoder(nn.Module):
    """Ask the latents about a gene, one chunk of genes at a time."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        from .blocks import Attention, FeedForward

        self.query_norm = nn.LayerNorm(dim)
        self.latent_norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, n_heads, dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, 2, dropout)

    def forward(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        out = queries + self.attention(self.query_norm(queries), self.latent_norm(latents))
        return out + self.ff(self.ff_norm(out))


def _output_head(dim: int) -> nn.Module:
    """An output head that starts by predicting exactly zero.

    Every head predicts a *change* — a signature, a GMT score, a deviation
    from the control — so zero is the honest starting point: the model begins
    at "this perturbation did nothing" and has to earn every departure from
    it. That is also what the challenge's own metrics reward a prediction for
    not knowing (§4.7: a gene with no confident change keeps a fold change of
    exactly 1), and it means an untrained head cannot start out worse than
    the no-change baseline.
    """
    final = nn.Linear(dim, 1)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    return nn.Sequential(nn.LayerNorm(dim), final)


class GeneTokenModel(nn.Module):
    """The core, plus the heads each phase reads it through."""

    def __init__(
        self,
        vocabulary: GeneVocabulary,
        *,
        n_latents: int = 64,
        n_blocks: int = 2,
        n_heads: int = 4,
        gene_chunk: int = 4096,
        dropout: float = 0.0,
        adapter_rank: int = 8,
        heads: Iterable[str] = ("sig", "gmt", "value"),
        pert_types: Iterable[str] = (),
        type_delta_l2: float = 1e-3,
    ) -> None:
        super().__init__()
        dim = vocabulary.embedding_dim
        self.vocabulary = vocabulary
        self.dim = dim
        self.gene_chunk = int(gene_chunk)
        self.head_names = tuple(heads)

        self.value_encoder = ValueEncoder(dim)
        self.context_encoder = ContextEncoder(dim)

        # The perturbation type, in the gene vocabulary's shape (§4.1):
        #
        #     type(assay) = base + delta[assay]
        #
        # `base` is trained by every phase and carries what all
        # knockdown-like perturbations share — that is what Phase 1's LINCS
        # data contributes to Phase 2, as an explicit named quantity rather
        # than an assumption buried in a warm start. `delta` absorbs what is
        # specific to one assay, starts at zero and is penalized toward it,
        # so the rows cannot quietly drift apart until nothing transfers.
        #
        # This matters because LINCS is CRISPR **knockout** and the challenge
        # is CRISPR **interference**: CLAUDE.md §3.3 says directions and
        # affected genes carry over between them but magnitudes and the
        # target's own level do not, and before this there was nowhere in the
        # model to put that distinction. A new assay is one new `delta` row.
        self.pert_types = tuple(sorted({str(name) for name in pert_types}))
        self.type_delta_l2 = float(type_delta_l2)
        self._pert_type_index = {name: i for i, name in enumerate(self.pert_types)}
        self.pert_type_base = nn.Parameter(torch.zeros(dim))
        self.pert_type_delta = nn.Parameter(torch.zeros(max(len(self.pert_types), 1), dim))

        self.latents = nn.Parameter(torch.randn(n_latents, dim) * 0.02)
        self.film = FiLM(dim, dim)
        self.blocks = nn.ModuleList(
            PerceiverBlock(dim, n_heads, dropout=dropout) for _ in range(n_blocks)
        )
        self.decoder = GeneQueryDecoder(dim, n_heads, dropout)
        self.heads = nn.ModuleDict(
            {name: _output_head(dim) for name in self.head_names}
        )

        # Every linear in the core becomes adapter-capable. The vocabulary's
        # projections are deliberately left alone: `delta` is already the
        # per-gene degree of freedom the specification gives (§4.1).
        self.n_adapted_linears = wrap_linears(self._core_modules(), adapter_rank)

    # ---- structure -------------------------------------------------------
    def _core_modules(self) -> nn.Module:
        """The parts an adapter may modify."""
        return nn.ModuleList(
            [
                self.value_encoder,
                self.context_encoder,
                self.film,
                self.blocks,
                self.decoder,
                self.heads,
            ]
        )

    def core_parameter_names(self) -> list[str]:
        """Parameters belonging to the core itself, adapters excluded."""
        skip = ("vocabulary.",)
        return [
            name
            for name, _ in self.named_parameters()
            if not name.startswith(skip) and ".up." not in name and ".down." not in name
        ]

    def add_adapter(self, name: str) -> None:
        add_adapter(self._core_modules(), name)

    def use_adapters(self, names: Iterable[str]) -> None:
        use_adapters(self._core_modules(), names)

    def adapter_parameters(self, names: Iterable[str]) -> list[nn.Parameter]:
        return adapter_parameters(self._core_modules(), names)

    # ---- the model -------------------------------------------------------
    def perturbation_type(self, pert_type: str | None) -> torch.Tensor:
        """`base + delta[pert_type]`, the "what kind of perturbation" vector."""
        if not self.pert_types:
            return self.pert_type_base
        if pert_type is None:
            raise ValueError(
                "this model knows the perturbation types "
                f"{list(self.pert_types)} but was given a perturbation token "
                "without one; pass pert_type= from the phase's config key "
                "(phase1.pert_type, phase2.pert_type, phase3.pert_type) so the "
                "assay is declared rather than assumed"
            )
        if pert_type not in self._pert_type_index:
            raise KeyError(
                f"unknown perturbation type {pert_type!r}; this model was built "
                f"for {list(self.pert_types)}. A new assay needs its type in the "
                "config before the model is built, so it gets a row of its own."
            )
        return self.pert_type_base + self.pert_type_delta[self._pert_type_index[pert_type]]

    def pert_type_penalty(self) -> torch.Tensor:
        """L2 on the per-assay offsets, which start at zero.

        The counterpart of the vocabulary's `delta_penalty`: it is what keeps
        one assay's type row near the shared base, and so keeps Phase 1 and
        Phase 2 sharing a perturbation representation instead of each
        training its own in isolation.
        """
        if not self.pert_types:
            return self.pert_type_delta.sum() * 0.0
        return self.type_delta_l2 * (self.pert_type_delta**2).sum()

    def regularization(self) -> torch.Tensor:
        """Every penalty the model asks of a phase's loss."""
        return self.vocabulary.delta_penalty() + self.pert_type_penalty()

    def tokenize(
        self,
        values: torch.Tensor,
        gene_idx: torch.Tensor,
        target_idx: torch.Tensor | None = None,
        pert_type: str | None = None,
    ) -> torch.Tensor:
        """`(B, G, C)` values over `gene_idx` -> `(B, G[+1], D)` tokens."""
        gene_embedding = self.vocabulary(gene_idx, role=FEATURE)
        tokens = gene_embedding.unsqueeze(0) + self.value_encoder(values)

        if target_idx is not None:
            # The perturbation token: what is being perturbed, and by what
            # kind of perturbation.
            pert = self.vocabulary(target_idx, role=TARGET) + self.perturbation_type(
                pert_type
            )
            tokens = torch.cat([tokens, pert.unsqueeze(1)], dim=1)
        return tokens

    def encode(
        self,
        values: torch.Tensor,
        gene_idx: torch.Tensor,
        *,
        target_idx: torch.Tensor | None = None,
        context_profile: torch.Tensor | None = None,
        pert_type: str | None = None,
    ) -> torch.Tensor:
        """Compress a profile into the latent bottleneck."""
        tokens = self.tokenize(values, gene_idx, target_idx, pert_type)
        latents = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)

        if context_profile is not None:
            pooled = self.value_encoder(context_profile).mean(dim=-2)
            latents = self.film(latents, self.context_encoder(pooled))

        for block in self.blocks:
            latents = block(latents, tokens)
        return latents

    def decode(
        self, latents: torch.Tensor, gene_idx: torch.Tensor, head: str = "value"
    ) -> torch.Tensor:
        """Predict `gene_idx` from the latents, in chunks.

        Chunking is what keeps the decoder's memory flat in the number of
        output genes — 18,533 of them on the server (CLAUDE.md §4.2).
        """
        if head not in self.heads:
            raise KeyError(f"unknown head {head!r}; this model has {list(self.head_names)}")

        outputs = []
        for chunk in _chunks(gene_idx, self.gene_chunk):
            queries = self.vocabulary(chunk, role=FEATURE)
            queries = queries.unsqueeze(0).expand(latents.shape[0], -1, -1)
            decoded = self.decoder(queries, latents)
            outputs.append(self.heads[head](decoded).squeeze(-1))
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        values: torch.Tensor,
        input_gene_idx: torch.Tensor,
        output_gene_idx: torch.Tensor,
        *,
        target_idx: torch.Tensor | None = None,
        context_profile: torch.Tensor | None = None,
        pert_type: str | None = None,
        head: str = "value",
    ) -> torch.Tensor:
        latents = self.encode(
            values,
            input_gene_idx,
            target_idx=target_idx,
            context_profile=context_profile,
            pert_type=pert_type,
        )
        return self.decode(latents, output_gene_idx, head)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, latents={self.latents.shape[0]}, blocks={len(self.blocks)}, "
            f"heads={list(self.head_names)}, pert_types={list(self.pert_types)}, "
            f"adapted_linears={self.n_adapted_linears}"
        )


def _chunks(index: torch.Tensor, size: int) -> Iterator[torch.Tensor]:
    for start in range(0, index.shape[0], size):
        yield index[start : start + size]


def build_model(cfg, prior_features, heads: Iterable[str] = ("sig", "gmt", "value")):
    """Build the core from the config and the built priors."""
    from .embedding import build_vocabulary

    vocabulary = build_vocabulary(
        prior_features, cfg.model.embedding_dim, delta_l2=cfg.model.delta_l2
    )
    return GeneTokenModel(
        vocabulary,
        n_latents=cfg.model.n_latents,
        n_blocks=cfg.model.n_blocks,
        n_heads=cfg.model.n_heads,
        gene_chunk=cfg.model.gene_chunk,
        dropout=cfg.model.dropout,
        adapter_rank=cfg.model.adapter_rank,
        heads=heads,
        pert_types=perturbation_types(cfg),
        type_delta_l2=cfg.train.type_delta_l2,
    )


def perturbation_types(cfg) -> tuple[str, ...]:
    """Every assay this config's phases will show the model.

    Read from the config rather than listed here, so that a new assay — a
    different screen, a drug perturbation, a future CRISPR flavour — is a
    config change and one new `delta` row, not a code change.
    """
    return tuple(
        sorted({cfg.phase1.pert_type, cfg.phase2.pert_type, cfg.phase3.pert_type})
    )


def zscore_across_genes(profile: np.ndarray) -> np.ndarray:
    """The *shape* of a profile: which genes are relatively high in it.

    The context vector is built from this rather than from the raw control
    profile, so that one encoder serves all three phases. A Phase 2 profile
    is in control-SD units, where a control mean is zero by construction and
    would say nothing about which context it came from; a Phase 1 profile is
    standardized per gene across rows. Z-scoring across genes puts both on
    the same footing.
    """
    profile = np.asarray(profile, dtype=np.float32)
    std = float(profile.std())
    return (profile - profile.mean()) / (std if std > 1e-8 else 1.0)


def as_index(values, device=None) -> torch.Tensor:
    """Gene indices as a tensor, from a list or an array."""
    return torch.as_tensor(np.asarray(values, dtype=np.int64), dtype=torch.long, device=device)
