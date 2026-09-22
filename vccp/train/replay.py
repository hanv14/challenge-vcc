"""Replay: keep training on the earlier phase while learning the later one.

The other half of the guard against forgetting (CLAUDE.md §4.3). A fraction
of a later phase's steps (`train.replay_fraction`) run an earlier phase's
batch and loss instead of the current phase's, so the earlier objective keeps
producing gradients rather than only being tested at the end.

A replay task carries its own sampler *and* its own loss, because replaying
Phase 1 batches through Phase 2's loss would measure nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class ReplayTask:
    """An earlier phase's objective, kept alive."""

    name: str
    #: `(rng) -> loss, metrics`
    step: Callable[[np.random.Generator], tuple[torch.Tensor, dict[str, float]]]


@dataclass
class ReplayMixer:
    """Decides, per step, whether to replay and which task to replay."""

    tasks: list[ReplayTask] = field(default_factory=list)
    fraction: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.tasks) and self.fraction > 0

    def pick(self, rng: np.random.Generator) -> ReplayTask | None:
        if not self.enabled or rng.random() >= self.fraction:
            return None
        task = self.tasks[int(rng.integers(len(self.tasks)))]
        self.counts[task.name] = self.counts.get(task.name, 0) + 1
        return task

    def report(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fraction": self.fraction,
            "tasks": [task.name for task in self.tasks],
            "steps_replayed": dict(self.counts),
        }
