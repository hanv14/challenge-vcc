"""L2-SP: pull a later phase's weights toward the previous phase's.

Part of the guard against forgetting (CLAUDE.md §4.3). Plain weight decay
pulls toward zero, which for a phase that starts from a trained model means
pulling toward forgetting it. L2-SP pulls toward the *starting point* instead,
so a later phase moves only as far as its own data justifies.

On by default in Phases 2 and 3 (`train.l2sp_weight`).
"""

from __future__ import annotations

import torch
from torch import nn


class L2SP:
    """A quadratic pull toward a reference state."""

    def __init__(self, model: nn.Module, weight: float) -> None:
        self.weight = float(weight)
        # The reference is the previous phase's values, detached so nothing
        # backpropagates into it.
        self.reference = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @property
    def enabled(self) -> bool:
        return self.weight > 0 and bool(self.reference)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        if not self.enabled:
            return torch.zeros((), device=device)

        total = torch.zeros((), device=device)
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            reference = self.reference.get(name)
            if reference is None or reference.shape != parameter.shape:
                continue
            total = total + ((parameter - reference.to(parameter.device)) ** 2).sum()
        return self.weight * total

    def drift(self, model: nn.Module) -> float:
        """How far the trainable weights have moved, in L2. Reported so the
        guard's effect is visible rather than assumed."""
        with torch.no_grad():
            total = 0.0
            for name, parameter in model.named_parameters():
                reference = self.reference.get(name)
                if reference is None or reference.shape != parameter.shape:
                    continue
                total += float(((parameter - reference.to(parameter.device)) ** 2).sum())
        return total**0.5
