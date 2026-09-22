"""Prior blocks 1 and 2 — co-expression across control cells.

What varies together in unperturbed cells is the strongest thing we know
about a gene that no perturbation screen has ever touched, and about 10,000
challenge genes are in exactly that position on the server. Block 1 reads
the challenge's own controls, which measure every gene; block 2 reads the
Replogle controls, which measure a few thousand.

Neither forms a gene-gene correlation matrix. Both stream cells from disk in
blocks and take a randomized SVD of the standardized data matrix, whose right
singular vectors are the same principal directions (see `decompose.py`).

Cells are standardized **within their own context** first, using the
`ctrl_mean` / `ctrl_std` each file already carries, so that a difference in
level between two contexts does not read as covariance between genes.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..data import challenge as challenge_data
from ..data import replogle as replogle_data
from .blocks import FEATURE, TARGET, BlockInputs, BlockResult
from .decompose import streaming_pca


def _sample_rows(n_available: int, n_max: int, rng: np.random.Generator) -> np.ndarray:
    """Up to `n_max` row indices, ascending. Reading in order keeps the
    on-disk access sequential."""
    if n_available <= n_max:
        return np.arange(n_available)
    return np.sort(rng.choice(n_available, n_max, replace=False))


def _chunks(rows: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


class ChallengeControlCoexpression:
    """Block 1 — co-expression over every challenge context's control cells.

    This is the only block that covers every challenge gene, which makes it
    the one that matters most for the genes measured nowhere else.
    """

    name = "challenge_coexpr"
    roles = (FEATURE, TARGET)
    reads_responses = False

    def build(self, inputs: BlockInputs) -> list[BlockResult]:
        cfg = inputs.cfg
        control_files = inputs.paths.discover_phase3_controls()
        contexts = {
            context: path
            for context, path in control_files.items()
            if inputs.scope.allows_context(context)
        }
        if not contexts:
            return []

        # Share the cell budget across the contexts that are in scope.
        per_context = max(2, cfg.priors.coexpr_max_cells // len(contexts))
        rng, seed = inputs.block_rng(self.name)
        plan = []
        for context, path in sorted(contexts.items()):
            controls = challenge_data.load_challenge_controls(path, context)
            rows = _sample_rows(controls.n_cells, per_context, rng)
            mean, std = controls.control_stats()
            plan.append((controls, rows, mean, std))

        n_genes = inputs.n_genes
        for controls, *_ in plan:
            if controls.n_genes != n_genes:
                raise ValueError(
                    f"{controls.path}: {controls.n_genes} genes, expected {n_genes} — "
                    "the control files must be on the challenge gene axis"
                )

        def block_iter() -> Iterator[np.ndarray]:
            for controls, rows, mean, std in plan:
                for chunk in _chunks(rows, cfg.priors.cell_block_size):
                    yield (controls.lognorm(chunk) - mean) / std

        components, variance_ratio, n_rows = streaming_pca(
            block_iter,
            n_genes,
            cfg.priors.block_dim,
            n_iter=cfg.priors.coexpr_n_iter,
            rng=rng,
        )

        covered = np.ones(n_genes, dtype=bool)
        covered = inputs.apply_gene_restriction(covered, source=self.name)
        return [
            BlockResult(
                name=self.name,
                features=components,
                covered=covered,
                column_names=[f"pc{i}" for i in range(components.shape[1])],
                source_context=None,
                reads_responses=False,
                sampling_seed=seed,
                detail={
                    "source": "challenge control cells",
                    "contexts": sorted(contexts),
                    "n_cells": int(n_rows),
                    "n_cells_available": sum(int(c.n_cells) for c, *_ in plan),
                    "explained_variance_ratio": float(variance_ratio.sum()),
                },
            )
        ]


class ReplogleControlCoexpression:
    """Block 2 — co-expression over each Replogle screen's control cells.

    One sub-block per screen rather than one pooled block: the screens
    measure different gene sets, and principal directions from two different
    SVDs are not comparable, so merging them would put unrelated numbers in
    the same coordinate. A gene measured in two screens gets features from
    both; a gene measured in one gets that one and a missing flag for the
    other.
    """

    name = "replogle_coexpr"
    roles = (FEATURE, TARGET)
    reads_responses = False

    def build(self, inputs: BlockInputs) -> list[BlockResult]:
        cfg = inputs.cfg
        results = []

        for context, directory in sorted(inputs.paths.discover_phase2_contexts().items()):
            if not inputs.scope.allows_context(context):
                continue

            ctx = replogle_data.load_replogle_context(directory)
            control_rows = ctx.cells.loc[ctx.cells["is_control"], "row"].to_numpy()
            if control_rows.size < 2:
                continue

            source = f"{self.name}:{context}"
            rng, seed = inputs.block_rng(source)
            selected = control_rows[
                _sample_rows(control_rows.size, cfg.priors.coexpr_max_cells, rng)
            ]
            mean, std = ctx.control_stats()
            measured_idx = ctx.challenge_idx
            panel_mask = ctx.genes["split"].to_numpy() == replogle_data.PANEL

            def block_iter(
                ctx=ctx, selected=selected, mean=mean, std=std, panel_mask=panel_mask
            ) -> Iterator[np.ndarray]:
                for chunk in _chunks(selected, cfg.priors.cell_block_size):
                    panel, rest = ctx.reader.get(chunk)
                    merged = np.empty((panel.shape[0], panel_mask.size), dtype=np.float32)
                    merged[:, panel_mask] = panel
                    merged[:, ~panel_mask] = rest
                    yield (merged - mean) / std

            components, variance_ratio, n_rows = streaming_pca(
                block_iter,
                measured_idx.size,
                cfg.priors.block_dim,
                n_iter=cfg.priors.coexpr_n_iter,
                rng=rng,
            )

            features = np.zeros((inputs.n_genes, components.shape[1]), dtype=np.float32)
            features[measured_idx] = components
            covered = np.zeros(inputs.n_genes, dtype=bool)
            covered[measured_idx] = True
            covered = inputs.apply_gene_restriction(covered, source=source)

            results.append(
                BlockResult(
                    name=source,
                    features=features,
                    covered=covered,
                    column_names=[f"pc{i}" for i in range(components.shape[1])],
                    source_context=context,
                    reads_responses=False,
                    sampling_seed=seed,
                    detail={
                        "source": f"Replogle {context} control cells",
                        "n_cells": int(n_rows),
                        "n_cells_available": int(control_rows.size),
                        "n_measured_genes": int(measured_idx.size),
                        "explained_variance_ratio": float(variance_ratio.sum()),
                    },
                )
            )

        return results
