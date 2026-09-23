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

#: cuBLAS needs a fixed workspace for its reductions to be reproducible, and
#: it reads this **once, when CUDA initializes** — so like the thread
#: variables it has to be set before torch is imported, not when the config
#: is read. `:4096:8` is the setting torch's own documentation names.
CUBLAS_WORKSPACE = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_VALUE = ":4096:8"


def set_thread_env(threads: int = DEFAULT_THREADS) -> dict[str, str]:
    """Set every thread variable that is unset. Returns the effective values."""
    for var in THREAD_VARS:
        os.environ.setdefault(var, str(threads))
    # Never overridden either: whatever the environment says, it keeps.
    os.environ.setdefault(CUBLAS_WORKSPACE, CUBLAS_WORKSPACE_VALUE)
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


def set_determinism(enabled: bool, device: str) -> dict[str, Any]:
    """Make two runs of the same checkpoint agree, bit for bit.

    Seeds alone do not do this. They fix which cells are drawn and which way
    a coin lands; they say nothing about the order a CUDA matmul reduces in,
    or whether it rounds through TF32. Those choices move the predicted
    values in their last bits, and a threshold applied afterwards turns that
    into thousands of genes moving or not moving.

    `warn_only` because a few ops have no deterministic implementation: a
    warning names them and the run continues, which is better than failing
    at hour three of twelve.
    """
    import torch

    if not enabled:
        return {"deterministic": False}

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TF32 trades mantissa bits for speed on matmuls — the single largest
    # source of run-to-run drift on an Ampere or later GPU.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    workspace = os.environ.get(CUBLAS_WORKSPACE)
    report = {
        "deterministic": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "tf32": False,
        CUBLAS_WORKSPACE: workspace or "<unset>",
        "warn_only": True,
        "note": "seeds fix the sampling; this fixes the arithmetic",
    }
    if not workspace and device == "cuda":
        # Saying "deterministic" while cuBLAS is free to reduce as it likes
        # would be worse than saying nothing. `vccp/__main__.py` sets it
        # before torch is imported; anything that bypasses that entry point
        # has to set it itself, and only before CUDA initializes.
        report["warning"] = (
            f"{CUBLAS_WORKSPACE} is unset, so cuBLAS reductions are still free to "
            f"vary. Set {CUBLAS_WORKSPACE}={CUBLAS_WORKSPACE_VALUE} before the "
            "process imports torch — `python -m vccp` does this for you."
        )
    return report


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
