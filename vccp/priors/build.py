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
class RoleLayout:
    """Where each block sits in a role's prior vector."""

    block_names: list[str]
    offsets: list[int]
    widths: list[int]
    #: Column holding the missing flag for each block.
    flag_columns: list[int]
    total_width: int


@dataclass
class PriorFeatures:
    """The built vocabulary: one prior matrix per role, plus its layout."""

    gene_symbols: list[str]
    priors: dict[str, np.ndarray]
    layouts: dict[str, RoleLayout]
    coverage: pd.DataFrame
    scope: PriorScope

    @property
    def n_genes(self) -> int:
        return len(self.gene_symbols)

    def width(self, role: str) -> int:
        return int(self.priors[role].shape[1])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "gene_symbols": np.asarray(self.gene_symbols, dtype=object),
            "layout_json": np.asarray(
                json.dumps(
                    {
                        role: {
                            "block_names": layout.block_names,
                            "offsets": layout.offsets,
                            "widths": layout.widths,
                            "flag_columns": layout.flag_columns,
                            "total_width": layout.total_width,
                        }
                        for role, layout in self.layouts.items()
                    }
                )
            ),
            "scope_json": np.asarray(json.dumps(self.scope.describe())),
        }
        for role, matrix in self.priors.items():
            payload[f"prior_{role}"] = matrix
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> PriorFeatures:
        loaded = np.load(path, allow_pickle=True)
        layouts_raw = json.loads(str(loaded["layout_json"]))
        return cls(
            gene_symbols=[str(s) for s in loaded["gene_symbols"]],
            priors={role: loaded[f"prior_{role}"] for role in ROLES if f"prior_{role}" in loaded},
            layouts={role: RoleLayout(**spec) for role, spec in layouts_raw.items()},
            coverage=pd.DataFrame(),
            scope=FULL_SCOPE,
        )


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
    names, offsets, widths, flags = [], [], [], []
    position = 0

    for result in results:
        standardized = _standardize(result)
        # 1.0 means "this block does not cover this gene", so the model can
        # tell a real zero from an absent source.
        missing = (~result.covered).astype(np.float32)[:, None]

        columns.append(standardized)
        columns.append(missing)
        names.append(result.name)
        offsets.append(position)
        widths.append(standardized.shape[1])
        flags.append(position + standardized.shape[1])
        position += standardized.shape[1] + 1

    if not columns:
        return np.zeros((n_genes, 0), dtype=np.float32), RoleLayout([], [], [], [], 0)

    return np.concatenate(columns, axis=1).astype(np.float32), RoleLayout(
        names, offsets, widths, flags, position
    )


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
            "  %s-role prior: %d blocks, width %d", role, len(layout.block_names), layout.total_width
        )

    for role, matrix in priors.items():
        if matrix.shape[1] == 0:
            raise RuntimeError(
                f"the {role}-role prior has no blocks — every source was missing or "
                "excluded by the scope, so the vocabulary would carry no information"
            )
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"the {role}-role prior holds non-finite values")

    return PriorFeatures(
        gene_symbols=axis,
        priors=priors,
        layouts=layouts,
        coverage=pd.DataFrame(coverage_rows),
        scope=scope,
    )
