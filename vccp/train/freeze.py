"""What trains in a phase, and proof that the rest did not (item 6).

CLAUDE.md §4.3 divides the model into a **core** and small **adapters**, and
says later phases train the adapters and `delta` while the core moves only if
the config allows it. A claim like that is worth nothing unless it is checked,
so every phase snapshots a digest of each parameter it declared frozen and
re-checks it afterwards; the result is `reports/frozen_check.json`.

The digest is of the tensor's bytes, so a parameter that was updated and then
happened to be restored to a similar value still registers as changed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

#: Parameter-name prefixes, used to decide what a name belongs to.
VOCABULARY = "vocabulary."
PROJECTIONS = "vocabulary.projections."
DELTAS = "vocabulary.deltas."
HEADS = "heads."


def parameter_group(name: str) -> str:
    """Which trainable group a parameter name belongs to."""
    if ".up." in name or ".down." in name:
        return "adapters"
    if name.startswith(DELTAS):
        return "delta"
    if name.startswith(PROJECTIONS):
        return "projections"
    if name.startswith(HEADS):
        return "heads"
    return "core"


@dataclass
class TrainablePlan:
    """What a phase declared it would train."""

    phase: str
    groups: dict[str, bool]
    adapters: list[str] = field(default_factory=list)
    trainable: list[str] = field(default_factory=list)
    frozen: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "groups": self.groups,
            "adapters": self.adapters,
            "n_trainable": len(self.trainable),
            "n_frozen": len(self.frozen),
            "n_trainable_parameters": None,  # filled by `describe`
        }


def set_trainable(
    model: nn.Module,
    phase: str,
    *,
    core: bool,
    adapters: list[str],
    delta: bool = True,
    projections: bool = False,
    heads: bool = True,
) -> TrainablePlan:
    """Set `requires_grad` across the model and return what was decided.

    `adapters` names the adapters this phase trains; every other adapter is
    frozen even though it exists, so an earlier phase's specialization cannot
    drift while a later phase runs.
    """
    from ..models.adapters import _key

    wanted_keys = {_key(name) for name in adapters}
    groups = {
        "core": core,
        "adapters": bool(adapters),
        "delta": delta,
        "projections": projections,
        "heads": heads,
    }

    plan = TrainablePlan(phase=phase, groups=groups, adapters=sorted(adapters))
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        if group == "adapters":
            key = name.rsplit(".", 1)[-1]
            trainable = key in wanted_keys
        else:
            trainable = groups[group]
        parameter.requires_grad_(trainable)
        (plan.trainable if trainable else plan.frozen).append(name)
    return plan


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()[:16]


def snapshot(model: nn.Module, names: list[str]) -> dict[str, str]:
    lookup = dict(model.named_parameters())
    return {name: digest(lookup[name]) for name in names if name in lookup}


def verify(model: nn.Module, before: dict[str, str]) -> dict[str, Any]:
    """Compare the frozen parameters against their snapshot."""
    lookup = dict(model.named_parameters())
    changed = [
        name for name, before_digest in before.items() if digest(lookup[name]) != before_digest
    ]
    return {
        "n_checked": len(before),
        "n_changed": len(changed),
        "changed": sorted(changed)[:20],
        "ok": not changed,
    }


def describe(model: nn.Module, plan: TrainablePlan) -> dict[str, Any]:
    """The plan, with parameter counts — for the report."""
    lookup = dict(model.named_parameters())
    by_group: dict[str, dict[str, int]] = {}
    for name, parameter in lookup.items():
        group = parameter_group(name)
        bucket = by_group.setdefault(group, {"trainable": 0, "frozen": 0})
        key = "trainable" if parameter.requires_grad else "frozen"
        bucket[key] += parameter.numel()

    return {
        "phase": plan.phase,
        "groups": plan.groups,
        "adapters": plan.adapters,
        "n_trainable_tensors": len(plan.trainable),
        "n_frozen_tensors": len(plan.frozen),
        "n_trainable_parameters": sum(
            lookup[name].numel() for name in plan.trainable if name in lookup
        ),
        "parameters_by_group": by_group,
    }


class FrozenCheck:
    """Snapshot on entry, verify on exit, accumulate across phases."""

    def __init__(self) -> None:
        self.phases: dict[str, Any] = {}

    def begin(self, model: nn.Module, plan: TrainablePlan) -> dict[str, str]:
        return snapshot(model, plan.frozen)

    def end(self, model: nn.Module, plan: TrainablePlan, before: dict[str, str]) -> dict[str, Any]:
        result = {**describe(model, plan), "frozen_verified": verify(model, before)}
        self.phases[plan.phase] = result
        return result

    def report(self) -> dict[str, Any]:
        return {
            "description": "parameters each phase declared frozen, and a digest check "
            "that they are bit-for-bit unchanged afterwards (CLAUDE.md §4.3)",
            "phases": self.phases,
            "all_ok": all(p["frozen_verified"]["ok"] for p in self.phases.values()),
        }
