"""Can the perturbation module fit anything at all? (PLAN_PERCELL.md §6)

Phase 2 reports the module at `pert ÷ no-change = 1.003` on validation and
**1.036 on its own training targets** — no better than predicting that
nothing happened, on data it has already seen. Before redesigning what it is
supervised against, this asks the prior question: *can it fit a handful of
targets it is allowed to memorize?*

The setup is deliberately the easiest possible. A few targets from one
screen, no validation, no mapping loss, no replay, no freezing — every
parameter trainable, the same targets every step, as many steps as it takes.
If the module cannot drive the training error well below the no-change line
here, the problem is not the data and not the supervision: it is capacity,
conditioning or optimisation, and any redesign would inherit it.

Three numbers are reported per checkpoint:

* **ratio** — error ÷ what predicting no change would cost. Below 1 means it
  is doing something; near 0 means it has memorized the targets.
* **pearson** — whether the predicted response points the right way.
* **magnitude ratio** — the size of the predicted change against the size of
  the true one. This is the one that catches the failure mode the pipeline
  has been showing: a module that predicts *almost nothing* scores a ratio
  near 1 while looking like it is training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..config import Config
from ..logging_utils import get_logger

#: Below this, the module has clearly fitted the targets.
FITTED = 0.5
#: A predicted change this much smaller than the truth is a collapse to
#: no-change however the error reads.
COLLAPSED_MAGNITUDE = 0.2


@dataclass
class Checkpoint:
    step: int
    loss: float
    ratio: float
    pearson: float
    magnitude_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "ratio_to_no_change": self.ratio,
            "pearson": self.pearson,
            "magnitude_ratio": self.magnitude_ratio,
        }


@dataclass
class CapacityResult:
    screen: str
    targets: list[str]
    steps: int
    curve: list[Checkpoint] = field(default_factory=list)

    @property
    def best(self) -> Checkpoint | None:
        return min(self.curve, key=lambda c: c.ratio) if self.curve else None

    def verdict(self) -> tuple[str, str]:
        """`(verdict, what it means)`."""
        best = self.best
        if best is None:
            return "no-data", "nothing was measured"
        if best.ratio < FITTED:
            return (
                "can-fit",
                f"the module drove training error to {best.ratio:.3f} of no-change on "
                f"{len(self.targets)} memorizable targets, so it has the capacity to "
                "represent a response. The failure on real training targets is about "
                "signal or supervision, not about the module — the per-cell redesign "
                "is worth building.",
            )
        if best.magnitude_ratio < COLLAPSED_MAGNITUDE:
            return (
                "collapses",
                f"the module cannot fit even these targets (best ratio {best.ratio:.3f}) "
                f"and its predicted changes are {best.magnitude_ratio:.2f}x the size of "
                "the true ones — it is converging to predicting no change. Something in "
                "the perturbation path is preventing it from expressing a response at "
                "all; find that before redesigning the supervision.",
            )
        return (
            "cannot-fit",
            f"the module cannot fit these targets (best ratio {best.ratio:.3f}) although "
            f"it does predict changes of a plausible size ({best.magnitude_ratio:.2f}x). "
            "That points at conditioning — the perturbation token may not be reaching "
            "the output — rather than at magnitude collapse.",
        )

    def as_dict(self) -> dict[str, Any]:
        verdict, meaning = self.verdict()
        return {
            "question": "can the perturbation module overfit a handful of targets?",
            "screen": self.screen,
            "targets": self.targets,
            "n_targets": len(self.targets),
            "steps": self.steps,
            "verdict": verdict,
            "meaning": meaning,
            "best": self.best.as_dict() if self.best else None,
            "curve": [c.as_dict() for c in self.curve],
            "thresholds": {"fitted_below": FITTED, "collapsed_magnitude": COLLAPSED_MAGNITUDE},
        }


def evaluate(model, tensors, target_idx, truth) -> tuple[float, float, float]:
    """`(ratio, pearson, magnitude_ratio)` on the fitted targets."""
    from ..phases.phase2 import predict_perturbed_panel
    from ..train.loop import pearson as pearson_of

    model.eval()
    with torch.no_grad():
        baseline = tensors.control_panel.unsqueeze(0).expand(truth.shape[0], -1)
        predicted = predict_perturbed_panel(model, tensors, baseline, target_idx)
    model.train()

    error = float(((predicted - truth) ** 2).mean())
    no_change = float((truth**2).mean())
    magnitude = float(predicted.abs().mean()) / max(float(truth.abs().mean()), 1e-9)
    return (
        error / max(no_change, 1e-12),
        pearson_of(predicted.cpu().numpy(), truth.cpu().numpy()),
        magnitude,
    )


def run_capacity_check(
    cfg: Config, *, screen: str | None = None, n_targets: int = 8,
    steps: int = 2000, log_every: int = 100, lr: float | None = None,
) -> dict[str, Any]:
    """Train on a few targets with nothing held back, and report the curve."""
    from ..data import reference
    from ..models.checkpoint import load_core
    from ..models.core import as_index, build_model
    from ..paths import DataPaths, RunPaths
    from ..phases import phase1, phase2
    from ..priors.build import PriorFeatures
    from ..runtime import resolve_device
    from ..train.loop import masked_mse

    log = get_logger()
    run_paths = RunPaths(cfg)
    paths = DataPaths(cfg)
    device = resolve_device(cfg.device)

    axis = reference.load_gene_names(paths.gene_names)
    gene_index = {symbol: i for i, symbol in enumerate(axis)}
    features = PriorFeatures.load(run_paths.prior_features)
    contexts = phase2.load_contexts(cfg, device, gene_index)

    usable = {n: t for n, t in contexts.items() if len(t.pert_train_targets) >= n_targets}
    if not usable:
        raise RuntimeError(
            f"no screen has {n_targets} targets that are challenge genes; "
            f"available: { {n: len(t.pert_train_targets) for n, t in contexts.items()} }"
        )
    name = screen if screen in usable else sorted(usable)[0]
    tensors = usable[name]

    rng = np.random.default_rng(cfg.seed)
    targets = sorted(
        rng.choice(tensors.pert_train_targets, n_targets, replace=False).tolist()
    )
    target_idx = as_index([gene_index[t] for t in targets], device)
    truth = torch.stack([tensors.pseudobulk[t] for t in targets])

    model = build_model(cfg, features).to(device)
    for adapter in (phase1.ADAPTER, phase2.ADAPTER):
        model.add_adapter(adapter)
    model.use_adapters([phase2.ADAPTER])
    checkpoint = run_paths.core_checkpoint(phase1.PHASE)
    if checkpoint.is_file():
        load_core(checkpoint, model, layout_hash=features.layout_hash(), strict=False)
        log.info("warm-started from %s", checkpoint)

    # Everything trainable: the question is capacity, so nothing is held back.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr or cfg.train.lr)

    result = CapacityResult(screen=name, targets=targets, steps=steps)
    log.info(
        "capacity check: %d targets from %s, %d steps, every parameter trainable",
        len(targets), name, steps,
    )
    log.info("  targets: %s", ", ".join(targets))

    from ..phases.phase2 import predict_perturbed_panel

    for step in range(1, steps + 1):
        baseline = tensors.control_panel.unsqueeze(0).expand(len(targets), -1)
        predicted = predict_perturbed_panel(model, tensors, baseline, target_idx)
        loss = masked_mse(predicted, truth)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        if step % log_every == 0 or step == 1:
            ratio, correlation, magnitude = evaluate(model, tensors, target_idx, truth)
            result.curve.append(
                Checkpoint(step, float(loss.detach()), ratio, correlation, magnitude)
            )
            log.info(
                "  step %5d  loss %.5f  ratio %.4f  pearson %+.3f  |pred|/|true| %.3f",
                step, float(loss.detach()), ratio, correlation, magnitude,
            )

    verdict, meaning = result.verdict()
    log.info("")
    log.info("verdict: %s", verdict.upper())
    log.info("  %s", meaning)

    report = result.as_dict()
    path = run_paths.reports / "capacity_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    log.info("written -> %s", path)
    return report
