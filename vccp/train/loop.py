"""The training loop every phase runs through.

Small on purpose: a step function supplied by the phase, a budget from the
config, and the three things that are the same everywhere — gradient
clipping, mixed precision on CUDA, and the forgetting guard (replay and
L2-SP) from §4.3.

A CUDA out-of-memory is caught once and re-raised naming the config key to
lower. It is never retried with a smaller batch: silently using a different
batch size than the config says would make the run unreproducible, and
CLAUDE.md §1.2 forbids reacting to a shortage by grabbing more.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from ..logging_utils import get_logger
from .l2sp import L2SP
from .replay import ReplayMixer

#: Config keys to suggest lowering, in the order worth trying.
MEMORY_KEYS = (
    "model.gene_chunk",
    "phase<N>.batch_size",
    "phase<N>.cells_per_draw",
    "model.n_latents",
)

StepFn = Callable[[np.random.Generator], "tuple[torch.Tensor, dict[str, float]]"]
EvalFn = Callable[[], dict[str, float]]


class OutOfMemory(RuntimeError):
    """A batch did not fit, with the config keys that control its size."""


@dataclass
class TrainResult:
    """What a phase's training produced."""

    steps: int
    curve: list[dict[str, Any]] = field(default_factory=list)
    final_metrics: dict[str, float] = field(default_factory=dict)
    replay: dict[str, Any] = field(default_factory=dict)
    l2sp_drift: float | None = None
    #: Validation at the step where `select_on` was best, and that step. A
    #: run that diverges late leaves `final_metrics` describing the wreck
    #: rather than the model: the first server Phase 2 ended 2.9x worse than
    #: its own best, and every ablation read off the final number measured
    #: that divergence instead of what it meant to measure.
    best_metrics: dict[str, float] = field(default_factory=dict)
    best_step: int | None = None
    #: The *worst* the selected metric got, and when. A run that swings wildly
    #: and happens to land near its best is unstable too, and `final / best`
    #: alone cannot see it: the first 6000-step server arm hit 2.46 at step
    #: 4500 and ended at 1.25, so it read as steady at 1.15x its best.
    worst_value: float | None = None
    worst_step: int | None = None
    selected_on: str | None = None

    def _ratio(self, value: float | None) -> float | None:
        best = self.best_metrics.get(self.selected_on) if self.selected_on else None
        if best is None or value is None or abs(best) < 1e-12:
            return None
        return float(value) / float(best)

    @property
    def divergence(self) -> float | None:
        """`final / best` on the selected metric. Above 1 means it got worse."""
        if not self.selected_on or not self.best_metrics:
            return None
        return self._ratio(self.final_metrics.get(self.selected_on))

    @property
    def instability(self) -> float | None:
        """`worst / best`. How far the run swung, whatever it ended at."""
        if not self.selected_on or not self.best_metrics:
            return None
        return self._ratio(self.worst_value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "final_metrics": self.final_metrics,
            "best_metrics": self.best_metrics,
            "best_step": self.best_step,
            "worst_value": self.worst_value,
            "worst_step": self.worst_step,
            "selected_on": self.selected_on,
            "divergence_final_over_best": self.divergence,
            "instability_worst_over_best": self.instability,
            "replay": self.replay,
            "l2sp_drift": self.l2sp_drift,
            "curve": self.curve,
        }


def run_training(
    model: nn.Module,
    step_fn: StepFn,
    *,
    steps: int,
    cfg,
    phase: str,
    rng: np.random.Generator,
    device: str,
    parameters: list[nn.Parameter] | None = None,
    replay: ReplayMixer | None = None,
    l2sp: L2SP | None = None,
    select_on: str | None = None,
    select_lower_is_better: bool = True,
    evaluate: EvalFn | None = None,
    eval_every: int | None = None,
) -> TrainResult:
    """Train for `steps`, logging a curve and evaluating along the way."""
    log = get_logger()
    trainable = [p for p in (parameters or model.parameters()) if p.requires_grad]
    if not trainable:
        raise ValueError(f"{phase}: nothing is trainable")

    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    use_amp = bool(cfg.train.amp) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    result = TrainResult(steps=steps)
    eval_every = eval_every or max(1, steps // 4)
    model.train()

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)

        task = replay.pick(rng) if replay is not None else None
        active_step = task.step if task is not None else step_fn

        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, metrics = active_step(rng)
                if l2sp is not None and l2sp.enabled:
                    penalty = l2sp.penalty(model)
                    metrics["l2sp"] = float(penalty.detach())
                    loss = loss + penalty
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - needs a GPU
            torch.cuda.empty_cache()
            keys = ", ".join(key.replace("<N>", phase[-1]) for key in MEMORY_KEYS)
            raise OutOfMemory(
                f"{phase}: a batch did not fit in the GPU memory this run is allowed "
                f"({cfg.resources.gpu_memory_fraction:.0%} of the device). Lower one of "
                f"{keys} in your config and rerun. The pipeline does not retry with a "
                "different batch size, because the config would no longer describe the run."
            ) from exc

        scaler.scale(loss).backward()
        if cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.train.log_every == 0 or step == steps:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "replayed": task.name if task is not None else None,
                **{k: float(v) for k, v in metrics.items()},
            }
            result.curve.append(record)

        if evaluate is not None and (step % eval_every == 0 or step == steps):
            model.eval()
            with torch.no_grad():
                scores = evaluate()
            model.train()
            result.final_metrics = scores
            if select_on is not None and select_on in scores:
                result.selected_on = select_on
                current = float(scores[select_on])
                previous = result.best_metrics.get(select_on)
                better = (
                    previous is None
                    or (current < previous if select_lower_is_better else current > previous)
                )
                if better:
                    result.best_metrics = dict(scores)
                    result.best_step = step
                worse = result.worst_value is None or (
                    current > result.worst_value
                    if select_lower_is_better
                    else current < result.worst_value
                )
                if worse:
                    result.worst_value = current
                    result.worst_step = step
            if result.curve:
                result.curve[-1].update({f"val_{k}": v for k, v in scores.items()})
            log.info(
                "  %s step %d/%d  loss %.4f  %s",
                phase,
                step,
                steps,
                float(loss.detach()),
                "  ".join(f"val_{k} {v:.4f}" for k, v in scores.items()),
            )

    model.eval()
    if replay is not None:
        result.replay = replay.report()
    if l2sp is not None and l2sp.enabled:
        result.l2sp_drift = l2sp.drift(model)
    return result


def sample_output_genes(
    gene_idx: torch.Tensor, n: int, rng: np.random.Generator
) -> tuple[torch.Tensor, np.ndarray]:
    """A fresh subsample of output genes for one training step.

    The loss is a mean over genes, so a subsample estimates it without bias
    while making the decoder's cost independent of how many genes the phase
    could have asked for. Returns the indices into the model's gene axis and
    the positions within `gene_idx`, since the targets have to be sliced the
    same way.
    """
    total = int(gene_idx.shape[0])
    if n <= 0 or n >= total:
        return gene_idx, np.arange(total)
    positions = np.sort(rng.choice(total, n, replace=False))
    selector = torch.as_tensor(positions, dtype=torch.long, device=gene_idx.device)
    return gene_idx.index_select(0, selector), positions


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
    """Mean squared error over the entries a mask keeps.

    Phase 2's loss is masked to the genes a context measures, and Phase 1's
    `gmt` head is masked to the rows that have one — in both cases an
    unmasked mean would average in entries that carry no truth.
    """
    error = (prediction - target) ** 2
    if mask is None:
        return error.mean()
    # A per-row mask arrives as (B, 1) and broadcasts over the genes, so the
    # denominator has to be counted *after* broadcasting: counting the mask's
    # own entries would divide a sum over B x G terms by B, inflating the
    # loss by the number of genes.
    mask = mask.to(error.dtype).expand_as(error)
    denominator = mask.sum()
    if float(denominator) == 0.0:
        return error.sum() * 0.0
    return (error * mask).sum() / denominator


def pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation over flattened arrays, NaN-safe."""
    a = np.asarray(prediction, dtype=np.float64).ravel()
    b = np.asarray(target, dtype=np.float64).ravel()
    usable = np.isfinite(a) & np.isfinite(b)
    if usable.sum() < 2:
        return float("nan")
    a, b = a[usable], b[usable]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
