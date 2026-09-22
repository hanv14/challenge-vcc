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
    #: Latent bottleneck size. Cost does not grow with the number of output
    #: genes, so this is the main capacity knob.
    n_latents: int = 64
    n_blocks: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    #: Output genes decoded at a time. Lower this first if the GPU runs out.
    gene_chunk: int = 4096
    #: Rank of the per-phase and per-context LoRA adapters (§4.3).
    adapter_rank: int = 8

    def validate(self) -> None:
        if self.embedding_dim < 1:
            raise ConfigError("model.embedding_dim must be at least 1")
        if self.delta_l2 < 0:
            raise ConfigError("model.delta_l2 must not be negative")
        if self.embedding_dim % self.n_heads:
            raise ConfigError(
                f"model.embedding_dim ({self.embedding_dim}) must be divisible by "
                f"model.n_heads ({self.n_heads})"
            )
        for key in ("n_latents", "n_blocks", "n_heads", "gene_chunk", "adapter_rank"):
            if getattr(self, key) < 1:
                raise ConfigError(f"model.{key} must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1)")


@dataclass(frozen=True)
class Train:
    """Budgets and the guard against forgetting (CLAUDE.md §4.3)."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    #: Mixed precision, when the device is CUDA. Ignored on CPU.
    amp: bool = True
    #: L2-SP: how hard a later phase is pulled toward the previous phase's
    #: weights. On by default in Phases 2 and 3 (§4.3).
    l2sp_weight: float = 1e-3
    #: Fraction of a later phase's steps that replay an earlier phase's
    #: batches instead.
    replay_fraction: float = 0.25
    #: Train a second Phase 2 arm with the core frozen, and score Phase 1
    #: validation under both, so the `unfreeze_core` default rests on a
    #: measurement rather than on a judgement (PLAN.md §8.9).
    core_freeze_ablation: bool = True
    #: The ablation arm's budget, as a fraction of the main arm's.
    ablation_steps_fraction: float = 1.0
    #: Steps between logged curve points.
    log_every: int = 25
    #: Output genes decoded per training step, sampled fresh each time. The
    #: loss is a mean over genes, so a subsample is an unbiased estimate of
    #: it, and the decoder's cost is linear in this. 0 means every gene —
    #: which on the server would be 18,533 of them per step. Evaluation
    #: always uses every gene.
    output_genes_per_step: int = 2048

    def validate(self) -> None:
        if self.lr <= 0:
            raise ConfigError("train.lr must be positive")
        if self.l2sp_weight < 0:
            raise ConfigError("train.l2sp_weight must not be negative")
        if not 0.0 <= self.replay_fraction < 1.0:
            raise ConfigError("train.replay_fraction must be in [0, 1)")
        if not 0.0 < self.ablation_steps_fraction <= 1.0:
            raise ConfigError("train.ablation_steps_fraction must be in (0, 1]")
        if self.log_every < 1:
            raise ConfigError("train.log_every must be at least 1")
        if self.output_genes_per_step < 0:
            raise ConfigError("train.output_genes_per_step must not be negative")


@dataclass(frozen=True)
class Phase1:
    """LINCS: control + perturbation -> signature -> perturbed (§4.4)."""

    steps: int = 400
    batch_size: int = 32
    #: Held-out fraction of *target genes*, so validation is unseen
    #: perturbations rather than unseen rows.
    val_fraction: float = 0.2
    loss_sig: float = 1.0
    loss_gmt: float = 0.5
    loss_pert: float = 1.0
    #: Feed the *true* signature into the second step instead of the
    #: predicted one. Off by default: the two steps train jointly.
    teacher_forcing: bool = False

    def validate(self) -> None:
        _check_phase("phase1", self.steps, self.batch_size, self.val_fraction)


@dataclass(frozen=True)
class Phase2:
    """Replogle: Siamese panel->rest, plus the perturbation module (§4.5)."""

    steps: int = 600
    batch_size: int = 32
    #: Cells drawn per target when sampling a batch.
    cells_per_draw: int = 16
    val_fraction: float = 0.2
    loss_mapping: float = 1.0
    loss_perturbation: float = 1.0
    #: Phase 2 has the data to support training the core, and is where the
    #: forgetting guard is worth having (DECISIONS.md D3).
    unfreeze_core: bool = True

    def validate(self) -> None:
        _check_phase("phase2", self.steps, self.batch_size, self.val_fraction)
        if self.cells_per_draw < 1:
            raise ConfigError("phase2.cells_per_draw must be at least 1")


def _check_phase(name: str, steps: int, batch_size: int, val_fraction: float) -> None:
    if steps < 1:
        raise ConfigError(f"{name}.steps must be at least 1")
    if batch_size < 1:
        raise ConfigError(f"{name}.batch_size must be at least 1")
    if not 0.0 < val_fraction < 1.0:
        raise ConfigError(f"{name}.val_fraction must be in (0, 1)")


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
    train: Train = field(default_factory=Train)
    phase1: Phase1 = field(default_factory=Phase1)
    phase2: Phase2 = field(default_factory=Phase2)

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
        self.train.validate()
        self.phase1.validate()
        self.phase2.validate()


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
    for name, cls in (
        ("resources", Resources),
        ("priors", Priors),
        ("model", Model),
        ("train", Train),
        ("phase1", Phase1),
        ("phase2", Phase2),
    ):
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
