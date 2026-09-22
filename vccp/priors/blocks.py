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
    """

    name: str
    features: np.ndarray
    covered: np.ndarray
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
