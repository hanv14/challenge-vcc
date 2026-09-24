"""Adapting the panel→rest mapping on a context's control cells alone.

This is Phase 3's procedure (CLAUDE.md §4.7): in the challenge, a context
arrives as control cells and nothing else, so the only thing that can be
fitted on it is the mapping from the panel to the rest — and both halves are
observed in a control cell, which is what makes that possible at all.

The rehearsal runs the *same* routine on a Replogle context (§4.6), which is
why it lives here rather than inside either caller. What adapts is set by
`phase3_policy.json`: context adapters and `delta`, with the perturbation
module and the core frozen and verified frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from ..logging_utils import get_logger
from ..train.freeze import FrozenCheck, set_trainable
from ..train.l2sp import L2SP
from ..train.loop import masked_mse, pearson, run_training, sample_output_genes


class ControlSource(Protocol):
    """A context that can hand over control cells in control-SD units."""

    name: str
    panel_idx: torch.Tensor
    rest_idx: torch.Tensor
    context_profile: torch.Tensor

    def draw_controls(
        self, n: int, rng: np.random.Generator, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(panel, rest)` for `n` control cells, in control-SD units."""
        ...


@dataclass
class AdaptationResult:
    """What adapting on one context's controls produced."""

    context: str
    adapter: str
    steps: int
    loss_before: float
    loss_after: float
    pearson_before: float
    pearson_after: float
    curve: list[dict[str, Any]]
    trainable: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "adapter": self.adapter,
            "steps": self.steps,
            "loss_before": self.loss_before,
            "loss_after": self.loss_after,
            "loss_improvement": self.loss_before - self.loss_after,
            "pearson_before": self.pearson_before,
            "pearson_after": self.pearson_after,
            "trainable": self.trainable,
            "curve": self.curve,
        }


def context_adapter(context: str) -> str:
    return f"context:{context}"


def map_panel_to_rest(
    model, source: ControlSource, panel_values: torch.Tensor, output_idx=None
) -> torch.Tensor:
    """The mapping, with no perturbation token — a statement about a cell's
    transcriptome, not about what was done to it."""
    context = source.context_profile.unsqueeze(0).expand(panel_values.shape[0], -1, -1)
    latents = model.encode(
        panel_values.unsqueeze(-1), source.panel_idx, context_profile=context
    )
    return model.decode(latents, source.rest_idx if output_idx is None else output_idx)


def evaluate_mapping(
    model, source: ControlSource, cfg, device: str, rng: np.random.Generator, n_cells: int
) -> tuple[float, float]:
    """`(mse, pearson)` of the mapping on a fresh draw of control cells."""
    model.eval()
    with torch.no_grad():
        panel, rest = source.draw_controls(n_cells, rng, device)
        predicted = map_panel_to_rest(model, source, panel)
        mse = float(((predicted - rest) ** 2).mean())
        correlation = pearson(predicted.cpu().numpy(), rest.cpu().numpy())
    return mse, correlation


def adapt_on_controls(
    model,
    source: ControlSource,
    cfg,
    *,
    steps: int,
    device: str,
    rng: np.random.Generator,
    phase: str = "phase3",
    train_delta: bool = True,
    frozen_check: FrozenCheck | None = None,
    batch_size: int | None = None,
) -> AdaptationResult:
    """Fit the mapping on this context's control cells, and nothing else.

    The perturbation module is frozen — what a knockdown does was learned in
    Phases 1 and 2 from data this context does not have, and a context whose
    only evidence is unperturbed cells has nothing to say about it.
    """
    log = get_logger()
    adapter = context_adapter(source.name)
    batch_size = batch_size or cfg.phase2.batch_size

    model.add_adapter(adapter)
    model.use_adapters([adapter])

    plan = set_trainable(
        model,
        f"{phase}:{source.name}",
        core=False,
        adapters=[adapter],
        delta=train_delta,
        projections=False,
        # The heads are shared with the perturbation module; freezing them
        # keeps "the perturbation module stays frozen" (§4.7) literal.
        heads=False,
    )
    check = frozen_check or FrozenCheck()
    before_snapshot = check.begin(model, plan)

    loss_before, pearson_before = evaluate_mapping(model, source, cfg, device, rng, batch_size)

    def step(step_rng: np.random.Generator):
        panel, rest = source.draw_controls(batch_size, step_rng, device)
        output_idx, positions = sample_output_genes(
            source.rest_idx, cfg.train.output_genes_per_step, step_rng
        )
        predicted = map_panel_to_rest(model, source, panel, output_idx)
        loss = masked_mse(predicted, rest[:, positions])
        return loss + model.regularization(), {"mapping": float(loss.detach())}

    l2sp = L2SP(model, cfg.train.l2sp_weight)
    result = run_training(
        model,
        step,
        steps=steps,
        cfg=cfg,
        phase=f"{phase}:{source.name}",
        rng=rng,
        device=device,
        l2sp=l2sp,
    )

    loss_after, pearson_after = evaluate_mapping(model, source, cfg, device, rng, batch_size)
    trainable = check.end(model, plan, before_snapshot)

    log.info(
        "  adapted %s on %d control-cell steps: mapping mse %.4f -> %.4f, pearson %+.3f -> %+.3f",
        source.name,
        steps,
        loss_before,
        loss_after,
        pearson_before,
        pearson_after,
    )
    return AdaptationResult(
        context=source.name,
        adapter=adapter,
        steps=steps,
        loss_before=loss_before,
        loss_after=loss_after,
        pearson_before=pearson_before,
        pearson_after=pearson_after,
        curve=result.curve,
        trainable=trainable,
    )
