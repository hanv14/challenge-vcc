"""Prior blocks 3 and 4 — what LINCS says about a gene as a measured feature.

Block 3 reads `layers['sig']` and block 4 `layers['gmt']` with genes as the
**columns**: two genes are alike when they move together across signatures,
or when they belong to the same up/down sets. Only the panel is covered,
because that is all LINCS measures.

Both read perturbation responses, so both honour the scope's held-out
targets — a rehearsal that scored a target whose LINCS signature had shaped
the gene embeddings would be measuring memory (CLAUDE.md §4.1).
"""

from __future__ import annotations

import numpy as np

from ..data import lincs as lincs_data
from .blocks import FEATURE, BlockInputs, BlockResult
from .decompose import resident_pca


class _LincsLayerBlock:
    """Shared body: one layer of the Phase 1 file, genes as columns."""

    layer: str
    row_mask_column: str
    roles = (FEATURE,)
    reads_responses = True

    def build(self, inputs: BlockInputs) -> list[BlockResult]:
        path = inputs.paths.phase1_lincs
        if not path.is_file():
            return []

        data = lincs_data.load_phase1(path)
        usable = data.obs[self.row_mask_column].to_numpy(bool)
        visible = np.array(
            [inputs.scope.allows_target(str(t)) for t in data.obs["target_gene"]], dtype=bool
        )
        rows = usable & visible
        if rows.sum() < 2:
            return []

        matrix = data.layers[self.layer][rows]
        components, variance_ratio = resident_pca(
            matrix, inputs.cfg.priors.block_dim, rng=inputs.rng
        )

        measured_idx = data.challenge_idx
        features = np.zeros((inputs.n_genes, components.shape[1]), dtype=np.float32)
        features[measured_idx] = components
        covered = np.zeros(inputs.n_genes, dtype=bool)
        covered[measured_idx] = True
        covered = inputs.apply_gene_restriction(covered, source=self.name)

        return [
            BlockResult(
                name=self.name,
                features=features,
                covered=covered,
                column_names=[f"pc{i}" for i in range(components.shape[1])],
                source_context=None,
                reads_responses=True,
                detail={
                    "source": f"phase1_lincs.h5ad layers[{self.layer!r}]",
                    "n_rows": int(rows.sum()),
                    "n_rows_dropped_by_scope": int((usable & ~visible).sum()),
                    "n_measured_genes": int(measured_idx.size),
                    "explained_variance_ratio": float(variance_ratio.sum()),
                },
            )
        ]


class LincsSignatureCovariation(_LincsLayerBlock):
    """Block 3 — genes that co-vary across Level 5 signatures."""

    name = "lincs_sig_covariation"
    layer = "sig"
    row_mask_column = "has_sig"


class GmtCoMembership(_LincsLayerBlock):
    """Block 4 — genes that share up/down gene-set membership.

    Masked to the rows that have a GMT signature at all; `has_gmt` is false
    for a good fraction of the Phase 1 rows.
    """

    name = "gmt_comembership"
    layer = "gmt"
    row_mask_column = "has_gmt"
