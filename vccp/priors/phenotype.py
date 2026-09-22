"""Prior block 6 — what happens when you knock this gene down.

**Target role only.** Everything else in the vocabulary describes a gene by
how it behaves; this describes it by what its loss does to everything else,
which is the thing a perturbation token actually has to encode.

Two sources, one sub-block each:

* Replogle — the pseudobulk difference between a target's cells and the
  context's controls, in control-SD units (CLAUDE.md §3.2), one sub-block per
  screen for the same reason as the co-expression blocks.
* LINCS — the mean Level 5 signature over a target's rows.

**Direction and magnitude are separated, and both are kept.** The response is
z-scored per target before the decomposition, because magnitude does not
transfer from a LINCS knockout to a challenge knockdown (§3.3) while direction
does. On its own that normalization is dangerous: most knockdowns do very
little, and a near-null response divided by its own tiny norm becomes a
confident-looking random direction that the model cannot distinguish from a
strong specific one. So each sub-block also carries scalar features saying how
large and how trustworthy the response was — the log magnitude, and whatever
quality signal the source provides (Replogle: knockdown efficiency, the
Anderson-Darling responsive-gene count, the cell count; LINCS: the replicate
correlation and the number of signatures).

The direction basis is additionally learned from the targets that *did*
respond (`priors.phenotype_weight_by_magnitude`), and every target is then
projected into it — so a null response lands where it lands in a basis it did
not help choose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import lincs as lincs_data
from ..data import manifest as manifest_mod
from ..data import replogle as replogle_data
from .blocks import TARGET, BlockInputs, BlockResult
from .decompose import magnitude_weights, rowwise_standardize, weighted_direction_basis

#: Quality columns read from a Replogle `*_raw_bulk.h5ad`, when present.
REPLOGLE_QUALITY = ("fold_expr", "anderson_darling_counts")

#: Reported so a reader can see the spread of response magnitudes, not just
#: the fraction under the floor.
RMS_PERCENTILES = (5, 25, 50, 75, 95)  # not-a-size: percentiles, not counts of anything
RMS_PERCENTILE_LABELS = ("p5", "p25", "p50", "p75", "p95")


def _log1p_positive(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(values, dtype=np.float64), 0.0)).astype(np.float32)


def _median_fill(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill non-finite entries with the median of the finite ones.

    The block already carries a gene-level missing flag; a per-column flag
    for every quality signal would cost more columns than it is worth, so a
    missing signal is filled with the typical one and the count reported.
    """
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values), int(values.size)
    filled = np.where(finite, values, np.median(values[finite]))
    return filled.astype(np.float32), int((~finite).sum())


class _PhenotypeBlock:
    """Shared body: responses in, direction features plus scalars out."""

    roles = (TARGET,)
    reads_responses = True

    def _assemble(
        self,
        inputs: BlockInputs,
        source: str,
        targets: list[str],
        profiles: np.ndarray,
        scalars: dict[str, np.ndarray],
        detail: dict,
        depth: tuple[str, np.ndarray] | None = None,
    ) -> BlockResult | None:
        """Turn `(n_targets, n_genes)` responses into a block result."""
        if profiles.shape[0] < 2:
            return None

        floor_fraction = inputs.cfg.priors.phenotype_magnitude_floor
        _, rms = rowwise_standardize(profiles)
        median_rms = float(np.median(rms))
        floor = floor_fraction * median_rms

        weights = None
        if inputs.cfg.priors.phenotype_weight_by_magnitude:
            weights = magnitude_weights(rms, floor)

        directions, variance_ratio, _ = weighted_direction_basis(
            profiles, inputs.cfg.priors.block_dim, weights
        )

        # The magnitude the direction was divided by, kept as its own feature.
        columns = [directions]
        names = [f"dir{i}" for i in range(directions.shape[1])]
        columns.append(_log1p_positive(rms)[:, None])
        names.append("log1p_magnitude")

        n_imputed = {}
        for label, values in scalars.items():
            filled, missing = _median_fill(values)
            columns.append(filled[:, None])
            names.append(label)
            if missing:
                n_imputed[label] = missing

        stacked = np.concatenate(columns, axis=1).astype(np.float32)

        index = inputs.index
        features = np.zeros((inputs.n_genes, stacked.shape[1]), dtype=np.float32)
        covered = np.zeros(inputs.n_genes, dtype=bool)
        for row, target in enumerate(targets):
            position = index.get(target)
            if position is None:
                continue  # a target that is not itself a challenge gene
            features[position] = stacked[row]
            covered[position] = True

        covered = inputs.apply_gene_restriction(covered, source=source)

        below = int((rms < floor).sum())
        percentiles = np.percentile(rms, RMS_PERCENTILES)

        # A shallow pseudobulk is noisy, and noise has a magnitude. Where
        # magnitude tracks depth rather than biology, the fraction below the
        # floor says more about cell counts than about which knockdowns did
        # nothing — so the correlation is reported next to it.
        depth_correlation = None
        depth_label = None
        if depth is not None:
            depth_label, depth_values = depth
            usable = np.isfinite(depth_values) & np.isfinite(rms)
            if usable.sum() > 2 and np.std(depth_values[usable]) > 0:
                depth_correlation = round(
                    float(np.corrcoef(rms[usable], depth_values[usable])[0, 1]), 4
                )

        detail = {
            **detail,
            "explained_variance_ratio": float(variance_ratio.sum()),
            "magnitude": {
                "median_rms": round(median_rms, 6),
                "rms_percentiles": {
                    label: round(float(value), 6)
                    for label, value in zip(RMS_PERCENTILE_LABELS, percentiles)
                },
                "floor_fraction_of_median": floor_fraction,
                "floor": round(float(floor), 6),
                "n_below_floor": below,
                "fraction_below_floor": round(below / len(rms), 4),
                "weighted_by_magnitude": bool(
                    inputs.cfg.priors.phenotype_weight_by_magnitude
                ),
                "depth_signal": depth_label,
                "magnitude_vs_depth_correlation": depth_correlation,
            },
            "n_imputed_quality_values": n_imputed,
        }

        return BlockResult(
            name=source,
            features=features,
            covered=covered,
            column_names=names,
            source_context=detail.get("context"),
            reads_responses=True,
            detail=detail,
        )


class ReplogleKnockdownPhenotype(_PhenotypeBlock):
    """Block 6a — the measured knockdown response, per Replogle screen."""

    name = "pert_phenotype_replogle"

    def build(self, inputs: BlockInputs) -> list[BlockResult]:
        manifest = manifest_mod.load_manifest(inputs.paths.manifest)
        bulk_files = inputs.paths.discover_replogle_bulk()
        results = []

        for context, directory in sorted(inputs.paths.discover_phase2_contexts().items()):
            if not inputs.scope.allows_responses_from(context):
                continue

            ctx = replogle_data.load_replogle_context(directory)
            obs, matrix, _ = replogle_data.load_pseudobulk(directory, manifest.control_label)

            labels = obs.index.astype(str).tolist()
            control_row = labels.index(manifest.control_label)
            control_profile = matrix[control_row]
            _, ctrl_std = ctx.control_stats()

            rows, targets = [], []
            n_dropped = 0
            for row, label in enumerate(labels):
                if label == manifest.control_label:
                    continue
                if not inputs.scope.allows_target(label):
                    n_dropped += 1
                    continue
                rows.append(row)
                targets.append(label)
            if len(rows) < 2:
                continue

            # Control-SD units: the bridge between sources (CLAUDE.md §3.2).
            deltas = (matrix[rows] - control_profile) / ctrl_std

            scalars = {}
            depth = None
            if "n_cells" in obs.columns:
                n_cells = obs["n_cells"].to_numpy(np.float64)[rows]
                scalars["log1p_n_cells"] = _log1p_positive(n_cells)
                depth = ("n_cells", n_cells)

            quality_path = bulk_files.get(f"{context}_raw_bulk")
            quality_columns: list[str] = []
            if quality_path is not None:
                quality = replogle_data.load_bulk_quality(quality_path, REPLOGLE_QUALITY)
                aligned = quality.reindex(targets)
                quality_columns = list(aligned.columns)
                if "fold_expr" in aligned:
                    scalars["fold_expr"] = aligned["fold_expr"].to_numpy(np.float32)
                if "anderson_darling_counts" in aligned:
                    scalars["log1p_anderson_darling"] = np.where(
                        np.isfinite(aligned["anderson_darling_counts"].to_numpy(np.float64)),
                        _log1p_positive(aligned["anderson_darling_counts"].to_numpy()),
                        np.nan,
                    ).astype(np.float32)

            source = f"{self.name}:{context}"
            result = self._assemble(
                inputs,
                source,
                targets,
                deltas,
                scalars,
                {
                    "source": f"Replogle {context} pseudobulk delta, control-SD units",
                    "context": context,
                    "n_targets": len(targets),
                    "n_targets_dropped_by_scope": n_dropped,
                    "n_response_genes": int(deltas.shape[1]),
                    "quality_source": quality_path.name if quality_path else None,
                    "quality_columns": quality_columns,
                },
                depth=depth,
            )
            if result is not None:
                results.append(result)

        return results


class LincsKnockdownPhenotype(_PhenotypeBlock):
    """Block 6b — the mean LINCS signature of a target, over its rows.

    LINCS is CRISPR knockout and the challenge is interference, so this is
    used for which genes move and in which direction, not for how much — and
    `cc_q75_median` (how well the replicate signatures agreed) says how much
    to trust even the direction.
    """

    name = "pert_phenotype_lincs"

    def build(self, inputs: BlockInputs) -> list[BlockResult]:
        path = inputs.paths.phase1_lincs
        if not path.is_file():
            return []

        data = lincs_data.load_phase1(path)
        usable = data.sig_mask()
        labels = data.obs["target_gene"].astype(str).to_numpy()

        profiles, targets = [], []
        cc_q75, n_sigs, n_rows = [], [], []
        n_dropped = 0
        for target in sorted(set(labels)):
            rows = usable & (labels == target)
            if not rows.any():
                continue
            if not inputs.scope.allows_target(target):
                n_dropped += 1
                continue
            # Several cell lines and time points per target; their mean is
            # the part of the response that is not cell-line specific.
            profiles.append(data.layers["sig"][rows].mean(axis=0))
            targets.append(target)

            subset = data.obs.loc[rows]
            cc_q75.append(pd.to_numeric(subset["cc_q75_median"], errors="coerce").mean())
            n_sigs.append(pd.to_numeric(subset["n_sigs"], errors="coerce").sum())
            n_rows.append(int(rows.sum()))

        if len(targets) < 2:
            return []

        scalars = {
            "cc_q75": np.asarray(cc_q75, dtype=np.float32),
            "log1p_n_sigs": _log1p_positive(np.asarray(n_sigs)),
            "log1p_n_rows": _log1p_positive(np.asarray(n_rows)),
        }

        result = self._assemble(
            inputs,
            self.name,
            targets,
            np.asarray(profiles, dtype=np.float32),
            scalars,
            {
                "source": "phase1_lincs.h5ad layers['sig'], averaged per target",
                "context": None,
                "n_targets": len(targets),
                "n_targets_dropped_by_scope": n_dropped,
                "n_response_genes": int(data.n_genes),
                "quality_columns": ["cc_q75_median", "n_sigs"],
            },
            depth=("n_sigs", np.asarray(n_sigs, dtype=np.float64)),
        )
        return [result] if result is not None else []
