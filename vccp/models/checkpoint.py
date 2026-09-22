"""Saving and loading the shared core.

A checkpoint records the **prior layout hash** it was trained against. The
block set is not fixed — the server's third Replogle screen adds blocks and
shifts every offset after them — so a checkpoint loaded against a different
layout would silently read the wrong columns of `W`. `load_core` refuses that
rather than producing plausible nonsense.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

FORMAT_VERSION = 1


def save_core(
    path: Path,
    model: nn.Module,
    *,
    phase: str,
    layout_hash: str,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "phase": phase,
            "layout_hash": layout_hash,
            "state_dict": model.state_dict(),
            "adapters": sorted(
                {
                    name.rsplit(".", 1)[-1]
                    for name, _ in model.named_parameters()
                    if ".up." in name
                }
            ),
            "extra": extra or {},
        },
        path,
    )


def load_core(path: Path, model: nn.Module, *, layout_hash: str, strict: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored = payload.get("layout_hash")
    if stored != layout_hash:
        raise ValueError(
            f"{path} was trained against prior layout {stored}, but the priors in this "
            f"run have layout {layout_hash}. Rerun the `priors` stage and the phases "
            "together, or point at the checkpoint that matches."
        )
    model.load_state_dict(payload["state_dict"], strict=strict)
    return payload
