"""Process-level settings: threads, seeds, device, GPU memory.

The thread limits matter before anything else: numpy, torch, polars and numba
fix their pool sizes at import, so `set_thread_env` has to run before those
imports. `vccp/__main__.py` calls it first, which is why the heavy imports in
this package all live inside functions rather than at module top level.

The rule for every variable here is: never override what the environment
already says (CLAUDE.md §1.2). `os.environ.setdefault` is the whole policy.
"""

from __future__ import annotations

import os
import random
from typing import Any

#: Thread pools that must be capped before the libraries reading them import.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
)

#: Used when the environment is silent and no config has been read yet. The
#: config's `resources.cpu_threads` is applied afterwards by `apply_resources`,
#: which can only lower a pool further, never raise one the environment set.
DEFAULT_THREADS = 1


def set_thread_env(threads: int = DEFAULT_THREADS) -> dict[str, str]:
    """Set every thread variable that is unset. Returns the effective values."""
    for var in THREAD_VARS:
        os.environ.setdefault(var, str(threads))
    return {var: os.environ[var] for var in THREAD_VARS}


def effective_thread_env() -> dict[str, str]:
    """The thread variables as they currently stand, for logging."""
    return {var: os.environ.get(var, "<unset>") for var in THREAD_VARS}


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if imported) torch."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    import numpy as np

    np.random.seed(seed)

    torch = _torch_if_available()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _torch_if_available() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def resolve_device(requested: str) -> str:
    """Turn the config's `device` into a concrete torch device string.

    Only ever `cuda:0` — the physical GPU is whatever `CUDA_VISIBLE_DEVICES`
    says it is, and the pipeline never picks an index itself (CLAUDE.md §1.2).
    """
    if requested == "cpu":
        return "cpu"

    torch = _torch_if_available()
    available = torch is not None and torch.cuda.is_available()

    if requested == "cuda":
        if not available:
            raise RuntimeError(
                "device: cuda was requested but torch reports no CUDA device. "
                "Check CUDA_VISIBLE_DEVICES, or set device: cpu in the config."
            )
        return "cuda:0"

    if requested == "auto":
        return "cuda:0" if available else "cpu"

    raise ValueError(f"unknown device {requested!r}")


def apply_resources(device: str, cpu_threads: int, gpu_memory_fraction: float) -> None:
    """Apply the config's resource limits to torch, once the device is known."""
    torch = _torch_if_available()
    if torch is None:
        return

    torch.set_num_threads(cpu_threads)
    if device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=0)


def release_gpu_memory() -> None:
    """Hand back GPU memory the pipeline no longer needs, between stages."""
    torch = _torch_if_available()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
