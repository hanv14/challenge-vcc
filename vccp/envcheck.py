"""`check-env` — is this machine set up to run the pipeline? (CLAUDE.md §9)

Run before the first `all` on the server. It answers, in seconds, the
questions that otherwise surface hours into a run:

* are the packages the pipeline imports installed, and which versions;
* is `cell-eval2` there, with its `vcc2026` preset and the DE backend the
  config pins — without it the rehearsal cannot score and sanity checks 6
  and 7 have nothing to read;
* is the challenge's own `vcc` tool on `PATH` — without it `validate` skips
  `vcc prep --dry-run` and no `.vcc` is written, which is expected in the
  cloud and is the user's own step on the server;
* is CUDA visible, which device the config resolves to, and what
  `CUDA_VISIBLE_DEVICES` says;
* are the thread limits of §1.2 in force — the values the process will
  actually use, after `scripts/server_env.sh` or the conservative defaults;
* do the configured data roots exist, and is there room under `output_root`.

Nothing here writes anything or reads a matrix.
"""

from __future__ import annotations

import importlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config

OK, WARN, FAIL = "ok", "warn", "fail"

#: What the pipeline imports. A missing one of these is a hard failure.
REQUIRED_PACKAGES = (
    "numpy", "scipy", "pandas", "sklearn", "torch", "anndata", "h5py", "yaml",
    "matplotlib",
)
#: Used by some stages, or by the data the server carries.
OPTIONAL_PACKAGES = ("scanpy", "polars", "pyarrow")

#: Rough floor for a full run's outputs: checkpoints, predictions and a
#: submission of the validation round's size.
MIN_FREE_GB = 20.0


@dataclass
class Finding:
    name: str
    status: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "message": self.message,
                **self.detail}


class Report:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(self, name: str, status: str, message: str, **detail: Any) -> None:
        self.findings.append(Finding(name, status, message, detail))

    @property
    def failed(self) -> list[Finding]:
        return [f for f in self.findings if f.status == FAIL]

    @property
    def warned(self) -> list[Finding]:
        return [f for f in self.findings if f.status == WARN]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": not self.failed,
            "n_fail": len(self.failed),
            "n_warn": len(self.warned),
            "findings": [f.as_dict() for f in self.findings],
        }

    def summary(self) -> str:
        lines = ["Environment check (CLAUDE.md §1, §1.2)", ""]
        for finding in self.findings:
            lines.append(f"  [{finding.status.upper():4}] {finding.name}: {finding.message}")
        lines.append("")
        lines.append(
            f"{len(self.findings) - len(self.failed) - len(self.warned)} ok, "
            f"{len(self.warned)} warning(s), {len(self.failed)} failure(s)"
        )
        if self.failed:
            lines.append("This machine cannot run the pipeline until the failures are fixed.")
        return "\n".join(lines)


#: Import name -> distribution name, where they differ.
DISTRIBUTIONS = {"sklearn": "scikit-learn", "yaml": "PyYAML"}


def _version(name: str, module) -> str:
    """The installed version, from the package metadata where possible.

    `module.__version__` is deprecated in several of these packages and warns
    when it is read, so the metadata is asked first.
    """
    import importlib.metadata as metadata

    try:
        return metadata.version(DISTRIBUTIONS.get(name, name))
    except Exception:  # noqa: BLE001 — a version is never worth failing over
        value = getattr(module, "__version__", None)
        return value if isinstance(value, str) else "unknown"


def check_packages(report: Report) -> None:
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            report.add(f"package:{name}", FAIL, f"not importable: {exc}")
        else:
            report.add(f"package:{name}", OK, _version(name, module))
    for name in OPTIONAL_PACKAGES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            report.add(f"package:{name}", WARN, "not installed")
        else:
            report.add(f"package:{name}", OK, _version(name, module))


def check_scorer(cfg: Config, report: Report) -> None:
    """`cell-eval2`, its preset, and the DE backend the config pins."""
    from .eval import official

    if not official.available():
        report.add(
            "cell-eval2", WARN,
            "not installed — the rehearsal cannot score with the official metrics and "
            "sanity checks 6 and 7 have nothing to read. `pip install cell-eval2` "
            "(CPU base install; no [gpu] or [gpudge] extras, CLAUDE.md §1)",
        )
        return

    try:
        config = official.build_config(cfg, "target_gene", "non-targeting")
    except Exception as exc:  # noqa: BLE001
        report.add("cell-eval2", FAIL, f"installed but unusable: {exc}")
        return

    report.add(
        "cell-eval2", OK,
        f"{official.scorer_version()}, preset vcc2026, de.backend={config.de.backend}, "
        f"{config.num_threads} thread(s)",
        version=official.scorer_version(), de_backend=config.de.backend,
        num_threads=config.num_threads,
    )

    backend = cfg.eval.de_backend
    module = {"pdex": "pdex", "scanpy": "scanpy", "deseq2": "pydeseq2"}.get(backend)
    if module:
        try:
            importlib.import_module(module)
        except ImportError:
            report.add(
                f"de-backend:{backend}", FAIL,
                f"eval.de_backend is {backend!r} but {module!r} is not installed",
            )
        else:
            report.add(f"de-backend:{backend}", OK, f"{module} importable")


def check_challenge_tool(report: Report) -> None:
    """The challenge's own `vcc`, which packages the `.vcc` (§8)."""
    from .submit import package

    path = package.tool_path()
    if path is None:
        report.add(
            "vcc", WARN,
            "not on PATH — `validate` will skip `vcc prep --dry-run` and no .vcc will "
            "be written. `submission/prediction.h5ad` is still the deliverable",
        )
        return
    report.add("vcc", OK, f"on PATH at {path}", path=path)


def check_device(cfg: Config, report: Report) -> None:
    from .runtime import resolve_device

    device = resolve_device(cfg.device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    try:
        import torch
    except ImportError:
        report.add("device", FAIL, "torch is not installed, so no device could be resolved")
        return

    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cap = cfg.resources.gpu_memory_fraction
        report.add(
            "device", OK,
            f"cuda:0 = {name}, {total:.0f} GB total, capped at "
            f"{cap:.0%} ({total * cap:.0f} GB) by resources.gpu_memory_fraction; "
            f"CUDA_VISIBLE_DEVICES={visible}",
            device=device, gpu=name, total_gb=round(total, 1),
        )
    elif cfg.device == "cuda":
        report.add("device", FAIL, "the config asks for cuda but torch reports none available")
    else:
        report.add(
            "device", WARN if cfg.device == "auto" else OK,
            f"running on CPU (config: {cfg.device}, CUDA_VISIBLE_DEVICES={visible}). "
            "A full run on the server is expected to use a GPU",
            device=device,
        )


def check_threads(cfg: Config, report: Report) -> None:
    """§1.2: the limits must be in force, and the pipeline never raises them."""
    from .runtime import effective_thread_env

    env = effective_thread_env()
    unset = [name for name, value in env.items() if value == "<unset>"]
    message = " ".join(f"{k}={v}" for k, v in env.items())
    if unset:
        report.add(
            "threads", WARN,
            f"{len(unset)} thread variable(s) unset ({', '.join(unset)}); the pipeline "
            "will set a conservative default at start-up, but `source scripts/"
            "server_env.sh` is what §1.2 asks for. Effective: " + message,
            unset=unset, effective=env,
        )
        return

    polars_threads = env.get("POLARS_MAX_THREADS")
    if polars_threads and polars_threads.isdigit() and int(polars_threads) < cfg.resources.cpu_threads:
        report.add(
            "threads", WARN,
            f"POLARS_MAX_THREADS={polars_threads} is below resources.cpu_threads="
            f"{cfg.resources.cpu_threads}; the scorer will use the smaller of the two. "
            + message,
            effective=env,
        )
        return
    report.add("threads", OK, message, effective=env)


def check_paths(cfg: Config, report: Report) -> None:
    from .paths import DataPaths

    paths = DataPaths(cfg)
    for label, path in (
        ("data_root", cfg.data_root), ("vcc_root", cfg.vcc_root),
        ("ref", cfg.ref_dir), ("processed/phases", cfg.phases_dir),
    ):
        if path.is_dir():
            report.add(f"path:{label}", OK, str(path))
        else:
            report.add(f"path:{label}", FAIL, f"{path} does not exist")

    if paths.manifest.is_file():
        from .data import manifest as manifest_mod

        manifest = manifest_mod.load_manifest(paths.manifest)
        report.add(
            "round", OK,
            f"{len(manifest.contexts)} context(s) {list(manifest.contexts)}, "
            f"{manifest.cells_per_pert} cells per perturbation, {manifest.n_genes} genes",
            contexts=list(manifest.contexts), cells_per_pert=manifest.cells_per_pert,
        )
    else:
        report.add("round", FAIL, f"{paths.manifest} does not exist")


def check_disk(cfg: Config, report: Report) -> None:
    root = cfg.output_root
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gb = usage.free / 1024**3
    status = OK if free_gb >= MIN_FREE_GB else WARN
    report.add(
        "disk", status,
        f"{free_gb:.0f} GB free under {root} (a full round's outputs need "
        f"roughly {MIN_FREE_GB:.0f} GB)",
        free_gb=round(free_gb, 1), output_root=str(root),
    )


def check_environment(cfg: Config) -> dict[str, Any]:
    """Every check above, in one report."""
    report = Report()
    check_packages(report)
    check_scorer(cfg, report)
    check_challenge_tool(report)
    check_device(cfg, report)
    check_threads(cfg, report)
    check_paths(cfg, report)
    check_disk(cfg, report)
    return report.as_dict() | {"summary": report.summary()}
