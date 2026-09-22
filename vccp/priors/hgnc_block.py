"""Prior block 5 — HGNC gene groups.

Curated membership: ribosomal proteins, proteasome subunits, respiratory
chain complexes, and several thousand smaller families. It is the one block
that knows a gene belongs with another without having measured either, which
is why it describes a gene in both roles.

A gene x group multi-hot is reduced by a truncated SVD. It is not centered:
centering a membership matrix makes it dense for no gain, and what is wanted
here — which genes share groups — survives without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import reference
from .blocks import FEATURE, TARGET, BlockInputs, BlockResult
from .decompose import sparse_svd

#: HGNC packs several groups into one cell, separated by this.
GROUP_SEPARATOR = "|"


def _symbol_lookup(frame: pd.DataFrame, axis: list[str]) -> dict[str, int]:
    """Challenge symbol -> HGNC row, by current symbol then by previous and
    alias symbols, so a gene HGNC has since renamed is still found."""
    wanted = set(axis)
    lookup: dict[str, int] = {}

    for position, symbol in enumerate(frame["symbol"].astype(str)):
        if symbol in wanted and symbol not in lookup:
            lookup[symbol] = position

    remaining = wanted - set(lookup)
    if not remaining:
        return lookup

    for column in ("prev_symbol", "alias_symbol"):
        if column not in frame.columns:
            continue
        for position, packed in enumerate(frame[column].fillna("")):
            for name in str(packed).split(GROUP_SEPARATOR):
                name = name.strip()
                if name in remaining and name not in lookup:
                    lookup[name] = position
        remaining = wanted - set(lookup)
        if not remaining:
            break

    return lookup


class HgncGeneGroups:
    """Block 5 — curated family membership."""

    name = "hgnc_gene_groups"
    roles = (FEATURE, TARGET)
    reads_responses = False

    def build(self, inputs: BlockInputs) -> list[BlockResult]:
        import scipy.sparse as sp

        try:
            path = inputs.paths.hgnc
        except FileNotFoundError:
            return []

        frame = reference.load_hgnc(path)
        lookup = _symbol_lookup(frame, inputs.axis)
        groups = frame["gene_group"].fillna("")

        # Collect membership, then drop the groups too small to say anything.
        members: dict[str, list[int]] = {}
        for gene_position, symbol in enumerate(inputs.axis):
            row = lookup.get(symbol)
            if row is None:
                continue
            for group in str(groups.iloc[row]).split(GROUP_SEPARATOR):
                group = group.strip()
                if group:
                    members.setdefault(group, []).append(gene_position)

        min_size = inputs.cfg.priors.hgnc_min_group_size
        kept = {g: rows for g, rows in members.items() if len(rows) >= min_size}
        if len(kept) < 2:
            return []

        rows, cols = [], []
        for column, (_, gene_positions) in enumerate(sorted(kept.items())):
            rows.extend(gene_positions)
            cols.extend([column] * len(gene_positions))
        multihot = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(inputs.n_genes, len(kept)),
        )

        features, variance_ratio = sparse_svd(multihot, inputs.cfg.priors.block_dim)
        covered = np.asarray(multihot.getnnz(axis=1) > 0)
        covered = inputs.apply_gene_restriction(covered, source=self.name)

        return [
            BlockResult(
                name=self.name,
                features=features,
                covered=covered,
                column_names=[f"sv{i}" for i in range(features.shape[1])],
                source_context=None,
                reads_responses=False,
                detail={
                    "source": path.name,
                    "n_groups": len(kept),
                    "n_groups_dropped_as_too_small": len(members) - len(kept),
                    "min_group_size": min_size,
                    "n_symbols_matched": len(lookup),
                    "explained_variance_ratio": float(variance_ratio.sum()),
                },
            )
        ]
