"""Assemble the gene vocabulary (CLAUDE.md §4.1, checklist items 1–3).

    embedding(g) = W . prior(g) + delta(g)

This module builds `prior(g)`: every block's features, each reduced to
`priors.block_dim`, standardized, zero-filled where the block does not cover
the gene, and followed by one missing flag per block. `W` and `delta` live in
`vccp/models/embedding.py`.

Two priors come out, not one. A gene is a measured feature and a gene is a
perturbation target, and the sources say different things about the two; the
model gets both and combines them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data import reference
from ..logging_utils import get_logger
from ..paths import DataPaths
from .blocks import FEATURE, TARGET, BlockInputs, BlockResult
from .coexpr import ChallengeControlCoexpression, ReplogleControlCoexpression
from .decompose import MIN_STD
from .hgnc_block import HgncGeneGroups
from .lincs_blocks import GmtCoMembership, LincsSignatureCovariation
from .phenotype import LincsKnockdownPhenotype, ReplogleKnockdownPhenotype
from .plm import ProteinLanguageModel
from .scope import FULL_SCOPE, PriorScope

ROLES = (FEATURE, TARGET)

#: The blocks of CLAUDE.md §4.1, in the order that section lists them.
#: Block 7 returns nothing unless `priors.plm_path` is configured.
def default_blocks() -> list:
    return [
        ChallengeControlCoexpression(),
        ReplogleControlCoexpression(),
        LincsSignatureCovariation(),
        GmtCoMembership(),
        HgncGeneGroups(),
        ReplogleKnockdownPhenotype(),
        LincsKnockdownPhenotype(),
        ProteinLanguageModel(),
    ]


@dataclass
class BlockSpec:
    """One block's place in a role's prior vector.

    Written into `coverage.csv` and into `prior_features.npz` so that a
    checkpoint can be matched to the layout it was trained with. The block
    set is not fixed — the server's third Replogle screen adds two blocks —
    so the layout has to travel with the artifact rather than be assumed.
    """

    name: str
    offset: int
    #: Feature columns, not counting the missing flag.
    width: int
    #: Column holding this block's missing flag.
    flag_column: int
    column_names: list[str]
    n_covered: int
    fraction_covered: float
    source_context: str | None = None
    reads_responses: bool = False
    sampling_seed: int | None = None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class RoleLayout:
    """Where every block sits in a role's prior vector."""

    blocks: list[BlockSpec]
    total_width: int

    # Convenience views, so callers need not unpack `blocks` every time.
    @property
    def block_names(self) -> list[str]:
        return [b.name for b in self.blocks]

    @property
    def offsets(self) -> list[int]:
        return [b.offset for b in self.blocks]

    @property
    def widths(self) -> list[int]:
        return [b.width for b in self.blocks]

    @property
    def flag_columns(self) -> list[int]:
        return [b.flag_column for b in self.blocks]

    @property
    def column_names(self) -> list[str]:
        """One label per column of the role's prior, flags included."""
        names: list[str] = []
        for block in self.blocks:
            names.extend(f"{block.name}/{column}" for column in block.column_names)
            names.append(f"{block.name}/is_missing")
        return names

    def as_dict(self) -> dict:
        return {
            "blocks": [b.as_dict() for b in self.blocks],
            "total_width": self.total_width,
        }

    @classmethod
    def from_dict(cls, spec: dict) -> RoleLayout:
        return cls(
            blocks=[BlockSpec(**b) for b in spec["blocks"]],
            total_width=spec["total_width"],
        )


@dataclass
class PriorFeatures:
    """The built vocabulary: one prior matrix per role, plus its layout."""

    gene_symbols: list[str]
    priors: dict[str, np.ndarray]
    layouts: dict[str, RoleLayout]
    coverage: pd.DataFrame
    scope: PriorScope
    #: The run seed the sampling seeds were derived from.
    seed: int = 0

    @property
    def n_genes(self) -> int:
        return len(self.gene_symbols)

    def width(self, role: str) -> int:
        return int(self.priors[role].shape[1])

    def layout_hash(self) -> str:
        """A digest of the layout, for matching a checkpoint to its priors.

        Covers every block's name, width, offset and column labels in both
        roles — everything a trained `W` depends on. It deliberately does not
        cover the prior *values*, so re-running the priors with the same
        sources and config keeps a checkpoint loadable.
        """
        payload = json.dumps(
            {role: layout.as_dict() for role, layout in sorted(self.layouts.items())},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def build_meta(self) -> dict:
        return {
            "n_genes": self.n_genes,
            "seed": self.seed,
            "scope": self.scope.describe(),
            "layout_hash": self.layout_hash(),
            "roles": {role: self.width(role) for role in sorted(self.priors)},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "gene_symbols": np.asarray(self.gene_symbols, dtype=object),
            "layout_json": np.asarray(
                json.dumps({role: layout.as_dict() for role, layout in self.layouts.items()})
            ),
            "scope_json": np.asarray(json.dumps(self.scope.as_dict())),
            "build_meta_json": np.asarray(json.dumps(self.build_meta())),
            "coverage_json": np.asarray(
                self.coverage.to_json(orient="records") if not self.coverage.empty else "[]"
            ),
        }
        for role, matrix in self.priors.items():
            payload[f"prior_{role}"] = matrix
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> PriorFeatures:
        loaded = np.load(path, allow_pickle=True)
        layouts_raw = json.loads(str(loaded["layout_json"]))
        meta = json.loads(str(loaded["build_meta_json"]))
        scope_raw = json.loads(str(loaded["scope_json"]))
        coverage = pd.read_json(io.StringIO(str(loaded["coverage_json"])), orient="records")

        features = cls(
            gene_symbols=[str(s) for s in loaded["gene_symbols"]],
            priors={role: loaded[f"prior_{role}"] for role in ROLES if f"prior_{role}" in loaded},
            layouts={role: RoleLayout.from_dict(spec) for role, spec in layouts_raw.items()},
            coverage=coverage,
            scope=PriorScope(
                exclude_targets=frozenset(scope_raw["exclude_targets"]),
                exclude_contexts=frozenset(scope_raw["exclude_contexts"]),
                exclude_response_contexts=frozenset(scope_raw["exclude_response_contexts"]),
                restricted_genes=frozenset(scope_raw["restricted_genes"]),
                restricted_genes_source=scope_raw["restricted_genes_source"],
                label=scope_raw["label"],
            ),
            seed=int(meta["seed"]),
        )
        if features.layout_hash() != meta["layout_hash"]:
            raise ValueError(
                f"{path}: the stored layout hash {meta['layout_hash']} does not match "
                f"the layout it holds ({features.layout_hash()})"
            )
        return features


def _standardize(result: BlockResult) -> np.ndarray:
    """Z-score a block's features over the genes it covers.

    Uncovered genes stay at exactly zero, which is what "this block says
    nothing about this gene" has to mean once the flag column is there.
    """
    features = result.features.astype(np.float32, copy=True)
    covered = result.covered
    if not covered.any():
        return np.zeros_like(features)

    present = features[covered]
    mean = present.mean(axis=0)
    std = present.std(axis=0)
    features[covered] = (present - mean) / np.where(std < MIN_STD, 1.0, std)
    features[~covered] = 0.0
    return features


def _assemble_role(results: list[BlockResult], n_genes: int) -> tuple[np.ndarray, RoleLayout]:
    """Concatenate the blocks of one role, each followed by its missing flag."""
    columns: list[np.ndarray] = []
    specs: list[BlockSpec] = []
    position = 0

    for result in results:
        standardized = _standardize(result)
        # 1.0 means "this block does not cover this gene", so the model can
        # tell a real zero from an absent source.
        missing = (~result.covered).astype(np.float32)[:, None]
        width = standardized.shape[1]

        columns.append(standardized)
        columns.append(missing)
        specs.append(
            BlockSpec(
                name=result.name,
                offset=position,
                width=width,
                flag_column=position + width,
                column_names=list(result.column_names),
                n_covered=result.n_covered,
                fraction_covered=round(result.n_covered / n_genes, 6),
                source_context=result.source_context,
                reads_responses=result.reads_responses,
                sampling_seed=result.sampling_seed,
            )
        )
        position += width + 1

    if not columns:
        return np.zeros((n_genes, 0), dtype=np.float32), RoleLayout([], 0)

    return np.concatenate(columns, axis=1).astype(np.float32), RoleLayout(specs, position)


def build_priors(cfg: Config, scope: PriorScope = FULL_SCOPE, blocks=None) -> PriorFeatures:
    """Build both role priors under `scope`."""
    log = get_logger()
    paths = DataPaths(cfg)
    axis = reference.load_gene_names(paths.gene_names)
    rng = np.random.default_rng(cfg.seed)

    inputs = BlockInputs(cfg=cfg, paths=paths, axis=axis, scope=scope, rng=rng)
    blocks = default_blocks() if blocks is None else blocks

    log.info("building gene priors over %d genes (scope: %s)", len(axis), scope.label)

    produced: list[tuple[object, BlockResult]] = []
    coverage_rows = []
    for block in blocks:
        results = block.build(inputs)
        if not results:
            log.info("  %-28s (not built)", block.name)
            coverage_rows.append(
                {
                    "block": block.name,
                    "roles": "|".join(block.roles),
                    "built": False,
                    "n_covered": 0,
                    "fraction_covered": 0.0,
                    "width": 0,
                    "reads_responses": block.reads_responses,
                    "source_context": None,
                    "sampling_seed": None,
                    "column_names": "",
                    "detail": "",
                }
            )
            continue

        for result in results:
            produced.append((block, result))
            fraction = result.n_covered / len(axis)
            log.info(
                "  %-28s %6d genes (%5.1f%%)  d=%d",
                result.name,
                result.n_covered,
                100 * fraction,
                result.features.shape[1],
            )
            coverage_rows.append(
                {
                    "block": result.name,
                    "roles": "|".join(block.roles),
                    "built": True,
                    "n_covered": result.n_covered,
                    "fraction_covered": round(fraction, 6),
                    "width": int(result.features.shape[1]),
                    "reads_responses": block.reads_responses,
                    "source_context": result.source_context,
                    "sampling_seed": result.sampling_seed,
                    "column_names": "|".join(result.column_names),
                    "detail": json.dumps(result.detail, default=str),
                }
            )

    priors, layouts = {}, {}
    for role in ROLES:
        role_results = [result for block, result in produced if role in block.roles]
        matrix, layout = _assemble_role(role_results, len(axis))
        priors[role] = matrix
        layouts[role] = layout
        log.info(
            "  %s-role prior: %d blocks, width %d",
            role,
            len(layout.blocks),
            layout.total_width,
        )

    for role, matrix in priors.items():
        if matrix.shape[1] == 0:
            raise RuntimeError(
                f"the {role}-role prior has no blocks — every source was missing or "
                "excluded by the scope, so the vocabulary would carry no information"
            )
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"the {role}-role prior holds non-finite values")

    coverage = pd.DataFrame(coverage_rows)
    coverage = _add_layout_columns(coverage, layouts)

    features = PriorFeatures(
        gene_symbols=axis,
        priors=priors,
        layouts=layouts,
        coverage=coverage,
        scope=scope,
        seed=cfg.seed,
    )
    log.info("  layout hash: %s", features.layout_hash())
    return features


def _add_layout_columns(coverage: pd.DataFrame, layouts: dict[str, RoleLayout]) -> pd.DataFrame:
    """Record each block's column range per role, so `coverage.csv` alone is
    enough to read a prior matrix back."""
    if coverage.empty:
        return coverage
    for role, layout in sorted(layouts.items()):
        positions = {spec.name: spec for spec in layout.blocks}
        coverage[f"{role}_offset"] = [
            positions[name].offset if name in positions else None for name in coverage["block"]
        ]
        coverage[f"{role}_flag_column"] = [
            positions[name].flag_column if name in positions else None
            for name in coverage["block"]
        ]
    return coverage
