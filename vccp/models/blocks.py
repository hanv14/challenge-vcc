"""The pieces the gene-token core is assembled from (CLAUDE.md §4.2).

Attention is written out with plain `nn.Linear` projections rather than using
`nn.MultiheadAttention`, because every projection has to be wrappable by a
LoRA adapter (§4.3) and the fused implementation hides its own.

Nothing here knows about genes or phases; that is `core.py`'s job.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    """Multi-head attention from `query` to `context`."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, query: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`query` (B, Q, D), `context` (B, C, D), `mask` (B, C) — True to keep."""
        q = self._split(self.to_q(query))
        k = self._split(self.to_k(context))
        v = self._split(self.to_v(context))

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].expand(-1, self.n_heads, q.shape[2], -1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        batch, _, length, _ = out.shape
        out = out.transpose(1, 2).reshape(batch, length, self.n_heads * self.head_dim)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverBlock(nn.Module):
    """One latent block: read the tokens, then think among the latents.

    The cross-attention is what makes cost independent of how many genes are
    fed in, and the latent self-attention is where the interaction between
    genes actually happens (CLAUDE.md §4.2).
    """

    def __init__(self, dim: int, n_heads: int, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.cross_norm_latent = nn.LayerNorm(dim)
        self.cross_norm_tokens = nn.LayerNorm(dim)
        self.cross = Attention(dim, n_heads, dropout)
        self.cross_ff_norm = nn.LayerNorm(dim)
        self.cross_ff = FeedForward(dim, expansion, dropout)

        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = Attention(dim, n_heads, dropout)
        self.self_ff_norm = nn.LayerNorm(dim)
        self.self_ff = FeedForward(dim, expansion, dropout)

    def forward(
        self, latents: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        latents = latents + self.cross(
            self.cross_norm_latent(latents), self.cross_norm_tokens(tokens), mask
        )
        latents = latents + self.cross_ff(self.cross_ff_norm(latents))
        normed = self.self_norm(latents)
        latents = latents + self.self_attention(normed, normed)
        return latents + self.self_ff(self.self_ff_norm(latents))


class FiLM(nn.Module):
    """Scale and shift, from a conditioning vector.

    Starts as the identity: `to_params` is zero-initialized, so a freshly
    built model behaves as though the conditioning were not there.
    """

    def __init__(self, dim: int, conditioning_dim: int) -> None:
        super().__init__()
        self.to_params = nn.Linear(conditioning_dim, 2 * dim)
        nn.init.zeros_(self.to_params.weight)
        nn.init.zeros_(self.to_params.bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_params(conditioning).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)
