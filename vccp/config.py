"""Typed configuration.

One config object drives every stage. Nothing in the pipeline is sized by
hand: gene counts, perturbation counts, cell counts and context names all
come from the data, and everything that *is* a choice (budgets, widths,
chunk sizes, thresholds) comes from here.

Unknown keys are rejected rather than ignored, so a typo in a YAML file
fails at load instead of silently leaving a default in place.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """A config file is malformed, or names a key the schema does not have."""


@dataclass(frozen=True)
class Resources:
    """Limits that keep the pipeline a good neighbour on the shared server.

    Every worker pool in the pipeline is sized from ``cpu_threads`` or
    ``num_workers``; nothing calls ``os.cpu_count()`` or passes ``n_jobs=-1``.
    """

    cpu_threads: int = 4
    num_workers: int = 2
    gpu_memory_fraction: float = 0.5
    max_ram_gb: float = 64.0

    def validate(self) -> None:
        if self.cpu_threads < 1:
            raise ConfigError("resources.cpu_threads must be at least 1")
        if self.num_workers < 0:
            raise ConfigError("resources.num_workers must not be negative")
        if not 0.0 < self.gpu_memory_fraction <= 1.0:
            raise ConfigError("resources.gpu_memory_fraction must be in (0, 1]")
        if self.max_ram_gb <= 0:
            raise ConfigError("resources.max_ram_gb must be positive")


@dataclass(frozen=True)
class Config:
    """The whole configuration, as loaded from a YAML file.

    ``source`` is the file it came from; ``repo_root`` is what relative paths
    in that file resolve against (the repository root, not the process's
    working directory, so a run from elsewhere still finds ``mini_data``).
    """

    data_root: Path
    vcc_root: Path
    output_root: Path
    device: str = "auto"
    run_name: str = "run"
    seed: int = 0
    resources: Resources = field(default_factory=Resources)

    source: Path | None = None
    repo_root: Path | None = None

    # ---- derived paths ---------------------------------------------------
    @property
    def ref_dir(self) -> Path:
        return self.data_root / "ref"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def phases_dir(self) -> Path:
        return self.processed_dir / "phases"

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    def validate(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigError(f"device must be one of auto, cpu, cuda (got {self.device!r})")
        if not self.run_name:
            raise ConfigError("run_name must not be empty")
        self.resources.validate()


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def _reject_unknown(section: str, given: dict[str, Any], cls: type) -> None:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(given) - known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {section}: {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(known))}"
        )


def load_config(path: str | Path, repo_root: str | Path | None = None) -> Config:
    """Load a YAML config, resolving relative paths against the repo root."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    base = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent

    resources_raw = raw.pop("resources", {}) or {}
    if not isinstance(resources_raw, dict):
        raise ConfigError(f"{path}: 'resources' must be a mapping")
    _reject_unknown("resources", resources_raw, Resources)
    resources = Resources(**resources_raw)

    # `source` and `repo_root` are set by the loader, never by the file.
    for reserved in ("source", "repo_root"):
        if reserved in raw:
            raise ConfigError(f"{path}: '{reserved}' is set by the loader, not by the config")

    _reject_unknown(str(path), raw, Config)
    for required in ("data_root", "vcc_root", "output_root"):
        if required not in raw:
            raise ConfigError(f"{path}: missing required key '{required}'")

    cfg = Config(
        data_root=_resolve(raw.pop("data_root"), base),
        vcc_root=_resolve(raw.pop("vcc_root"), base),
        output_root=_resolve(raw.pop("output_root"), base),
        resources=resources,
        source=path.resolve(),
        repo_root=base,
        **raw,
    )
    cfg.validate()
    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """A plain dict of the config, for writing into the run directory."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        if isinstance(value, Path):
            out[f.name] = str(value)
        elif dataclasses.is_dataclass(value):
            out[f.name] = dataclasses.asdict(value)
        else:
            out[f.name] = value
    return out
