"""The prior block interface.

A block turns one data source into `(n_genes, d)` features plus a coverage
mask saying which genes it actually informs. `build.py` reduces, standardizes
and concatenates them; the block itself only has to produce its features.

Two roles (CLAUDE.md §4.1): a gene is a **measured feature** and a gene is a
**perturbation target**, and the same source does not say the same thing
about both. Co-expression and gene-group membership describe a gene either
way, so those blocks serve both roles. A knockdown phenotype only describes a
gene as a target. Nothing describes a gene only as a feature, which is why
the target-role prior is the wider of the two.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..config import Config
from ..paths import DataPaths
from .scope import PriorScope

FEATURE = "feature"
TARGET = "target"


@dataclass
class BlockResult:
    """What a block produces for one role.

    `features` is `(n_genes, d)` with zeros wherever `covered` is false;
    `build.py` keeps that convention and adds the missing-flag column.

    `column_names`, `source_context`, `reads_responses` and `sampling_seed`
    are provenance: they travel into `coverage.csv` and into
    `prior_features.npz` so that a checkpoint can be matched to the prior
    layout it was trained with, and so the rehearsal can say which blocks it
    had to rebuild.
    """

    name: str
    features: np.ndarray
    covered: np.ndarray
    #: One label per feature column. Blocks that mix a decomposition with
    #: scalar features need this to stay readable.
    column_names: list[str] = field(default_factory=list)
    #: The source instance this result came from (a screen name), or None
    #: for a block with a single source.
    source_context: str | None = None
    #: Whether this result read perturbation responses — what decides
    #: whether a rehearsal scope forces it to be rebuilt.
    reads_responses: bool = False
    #: The seed used for any sampling inside the block, so the same cells
    #: are drawn again on a rerun.
    sampling_seed: int | None = None
    #: Anything worth putting in `coverage.csv` — the source it read, how
    #: many rows it saw, how much variance the components carry.
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.features.shape[0] != self.covered.shape[0]:
            raise ValueError(
                f"{self.name}: features has {self.features.shape[0]} rows but the "
                f"coverage mask has {self.covered.shape[0]}"
            )
        self.features = np.where(self.covered[:, None], self.features, 0.0).astype(np.float32)
        if not self.column_names:
            self.column_names = [f"c{i}" for i in range(self.features.shape[1])]
        if len(self.column_names) != self.features.shape[1]:
            raise ValueError(
                f"{self.name}: {len(self.column_names)} column name(s) for "
                f"{self.features.shape[1]} column(s)"
            )

    @property
    def n_covered(self) -> int:
        return int(self.covered.sum())


@dataclass(frozen=True)
class BlockInputs:
    """Everything a block may read, resolved once and shared."""

    cfg: Config
    paths: DataPaths
    #: Challenge genes in challenge order — the axis every block returns on.
    axis: list[str]
    scope: PriorScope
    rng: np.random.Generator

    @property
    def n_genes(self) -> int:
        return len(self.axis)

    @property
    def index(self) -> dict[str, int]:
        return {symbol: i for i, symbol in enumerate(self.axis)}

    def block_seed(self, source: str) -> int:
        """A seed for `source`, derived from the run seed and the scope.

        Derived per block rather than drawn from one shared generator, so
        the cells a block samples do not depend on how many blocks ran
        before it — and so the seed can be written down next to the block.
        """
        material = f"{self.cfg.seed}:{self.scope.cache_key()}:{source}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32)

    def block_rng(self, source: str) -> tuple[np.random.Generator, int]:
        seed = self.block_seed(source)
        return np.random.default_rng(seed), seed

    def empty(self, width: int) -> tuple[np.ndarray, np.ndarray]:
        """An all-uncovered result, for a source this scope excludes."""
        return (
            np.zeros((self.n_genes, width), dtype=np.float32),
            np.zeros(self.n_genes, dtype=bool),
        )

    def apply_gene_restriction(self, covered: np.ndarray, source: str) -> np.ndarray:
        """Hide the scope's restricted genes from every source but one.

        This is what makes the rehearsal's "unseen genes" variant honest: a
        gene pretended unmeasured must not reach the model through a prior
        built from the data that measured it.
        """
        if not self.scope.restricted_genes:
            return covered
        if source == self.scope.restricted_genes_source:
            return covered
        hidden = np.array(
            [symbol in self.scope.restricted_genes for symbol in self.axis], dtype=bool
        )
        return covered & ~hidden


class PriorBlock(Protocol):
    """One source of gene features."""

    name: str
    roles: tuple[str, ...]
    #: True when the block reads perturbation responses, and so must be
    #: rebuilt under a rehearsal scope (CLAUDE.md §4.1, leakage rule).
    reads_responses: bool

    def build(self, inputs: BlockInputs) -> list[BlockResult]:
        """Return one result per sub-block.

        A source that exists once per context (the Replogle screens) returns
        one result per context rather than trying to merge screens that
        measure different genes into one basis.
        """
        ...
