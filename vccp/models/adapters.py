"""Per-phase and per-context adapters (CLAUDE.md §4.3, checklist item 6).

The core is one network shared by all three phases. What differs between a
LINCS signature, a Replogle cell and a challenge context is carried by small
low-rank adapters bolted onto the core's linear layers, so a later phase can
specialize without overwriting what an earlier one learned.

A `LoRALinear` starts as an exact identity — its `down` matrix is zero — so
adding an adapter never changes what the model does until it is trained.

Which adapters are active is model state, set by `use_adapters`, not a global:
two models in one process must not be able to interfere.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A linear layer with named low-rank adapters.

    `y = base(x) + sum over active adapters of (x @ up^T) @ down^T * scale`
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / max(self.rank, 1)
        self.up = nn.ParameterDict()
        self.down = nn.ParameterDict()
        self._active: tuple[str, ...] = ()

    @property
    def adapter_names(self) -> list[str]:
        return sorted(self.up)

    def add_adapter(self, name: str) -> None:
        if name in self.up:
            return
        key = _key(name)
        up = torch.empty(self.rank, self.base.in_features)
        nn.init.kaiming_uniform_(up, a=5**0.5)
        self.up[key] = nn.Parameter(up)
        # Zero, so the adapter is an exact no-op until it is trained.
        self.down[key] = nn.Parameter(torch.zeros(self.base.out_features, self.rank))

    def set_active(self, names: Iterable[str]) -> None:
        self._active = tuple(_key(n) for n in names if _key(n) in self.up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        for key in self._active:
            out = out + (x @ self.up[key].T) @ self.down[key].T * self.scale
        return out

    def extra_repr(self) -> str:
        return f"rank={self.rank}, adapters={self.adapter_names}"


def _key(name: str) -> str:
    """`ParameterDict` keys cannot contain a dot."""
    return name.replace(".", "_").replace(":", "__")


def wrap_linears(module: nn.Module, rank: int, alpha: float = 1.0) -> int:
    """Replace every `nn.Linear` under `module` with a `LoRALinear`.

    Returns how many were wrapped. Called once when the core is built, so
    that any adapter can be added later without touching the architecture.
    """
    wrapped = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            wrapped += 1
        elif not isinstance(child, LoRALinear):
            wrapped += wrap_linears(child, rank, alpha)
    return wrapped


def iter_lora(module: nn.Module) -> Iterator[LoRALinear]:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def add_adapter(module: nn.Module, name: str) -> None:
    """Add one named adapter to every wrapped linear layer."""
    for layer in iter_lora(module):
        layer.add_adapter(name)


def use_adapters(module: nn.Module, names: Iterable[str]) -> None:
    """Make exactly `names` active; anything else is switched off."""
    names = list(names)
    for layer in iter_lora(module):
        layer.set_active(names)


def adapter_parameters(module: nn.Module, names: Iterable[str]) -> list[nn.Parameter]:
    """The parameters belonging to `names`, for an optimizer."""
    keys = {_key(n) for n in names}
    found = []
    for layer in iter_lora(module):
        for key in keys & set(layer.up):
            found.append(layer.up[key])
            found.append(layer.down[key])
    return found


def adapter_names(module: nn.Module) -> list[str]:
    names: set[str] = set()
    for layer in iter_lora(module):
        names.update(layer.up)
    return sorted(names)


def is_adapter_parameter(parameter_name: str) -> bool:
    """Whether a `named_parameters()` name belongs to an adapter."""
    return ".up." in parameter_name or ".down." in parameter_name
