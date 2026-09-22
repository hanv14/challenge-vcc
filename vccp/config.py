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
class Priors:
    """Gene vocabulary and prior blocks (CLAUDE.md §4.1)."""

    #: Each prior block is reduced to this many components before it is
    #: standardized and concatenated.
    block_dim: int = 32
    #: Cells sampled per source when estimating a co-expression block. The
    #: estimate is over genes, so more cells buy accuracy, not coverage.
    coexpr_max_cells: int = 20000
    #: Power iterations in the randomized SVD. More is more accurate and
    #: costs one streaming pass over the cells each.
    coexpr_n_iter: int = 2
    #: Gene-group families smaller than this carry no usable signal.
    hgnc_min_group_size: int = 2
    #: A knockdown's response is z-scored per target before the direction
    #: decomposition, which makes a near-null response look like a confident
    #: random direction. These two keys are how that is handled: the floor is
    #: a fraction of the sub-block's median response magnitude, below which a
    #: target is reported as low-magnitude, and, when weighting is on, the
    #: point at which its influence on the basis is halved.
    phenotype_magnitude_floor: float = 0.5
    #: Learn the direction basis from the targets that actually responded,
    #: then project every target into it. Off means every target, however
    #: null, shapes the basis equally.
    phenotype_weight_by_magnitude: bool = True
    #: Protein language model embeddings (block 7). Built only when this
    #: points at a file that exists; never downloaded (CLAUDE.md §4.1).
    plm_path: str | None = None
    #: Rows read at a time when streaming cells for a co-expression block.
    cell_block_size: int = 2048
    #: Nearest neighbours examined by the prior checks.
    check_n_neighbors: int = 10
    #: Targets sampled for the response-correlation check. The check is over
    #: pairs, so this caps a quadratic; the server has thousands of targets.
    check_max_targets: int = 500
    #: Gene-group name fragments the neighbour check looks for, matched
    #: case-insensitively against HGNC `gene_group`.
    check_families: tuple[str, ...] = (
        "ribosomal",
        "proteasome",
        "mitochondrial complex",
        "mitochondrial respiratory chain",
    )

    def validate(self) -> None:
        if self.block_dim < 1:
            raise ConfigError("priors.block_dim must be at least 1")
        if self.coexpr_max_cells < 2:
            raise ConfigError("priors.coexpr_max_cells must be at least 2")
        if self.coexpr_n_iter < 0:
            raise ConfigError("priors.coexpr_n_iter must not be negative")
        if self.cell_block_size < 1:
            raise ConfigError("priors.cell_block_size must be at least 1")
        if self.check_n_neighbors < 1:
            raise ConfigError("priors.check_n_neighbors must be at least 1")
        if self.check_max_targets < 2:
            raise ConfigError("priors.check_max_targets must be at least 2")
        if self.phenotype_magnitude_floor <= 0:
            raise ConfigError("priors.phenotype_magnitude_floor must be positive")


@dataclass(frozen=True)
class Model:
    """The shared gene-token model (CLAUDE.md §4.2)."""

    #: Width of a gene embedding: `W . prior(g) + delta(g)`.
    embedding_dim: int = 128
    #: L2 pull on the free per-gene vector, which starts at zero.
    delta_l2: float = 1e-4

    def validate(self) -> None:
        if self.embedding_dim < 1:
            raise ConfigError("model.embedding_dim must be at least 1")
        if self.delta_l2 < 0:
            raise ConfigError("model.delta_l2 must not be negative")


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
    priors: Priors = field(default_factory=Priors)
    model: Model = field(default_factory=Model)

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
        self.priors.validate()
        self.model.validate()


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def _build_section(cls: type, given: dict[str, Any]):
    """Build a config section, keeping tuple fields tuples so the whole
    config stays hashable and comparable."""
    coerced = dict(given)
    for f in dataclasses.fields(cls):
        if f.name in coerced and isinstance(coerced[f.name], list):
            coerced[f.name] = tuple(coerced[f.name])
    return cls(**coerced)


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

    sections = {}
    for name, cls in (("resources", Resources), ("priors", Priors), ("model", Model)):
        section_raw = raw.pop(name, {}) or {}
        if not isinstance(section_raw, dict):
            raise ConfigError(f"{path}: {name!r} must be a mapping")
        _reject_unknown(name, section_raw, cls)
        sections[name] = _build_section(cls, section_raw)

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
        source=path.resolve(),
        repo_root=base,
        **sections,
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
