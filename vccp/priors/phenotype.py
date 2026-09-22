"""Prior block 6 — what happens when you knock this gene down.

**Target role only.** Everything else in the vocabulary describes a gene by
how it behaves; this describes it by what its loss does to everything else,
which is the thing a perturbation token actually has to encode.

Two sources, one sub-block each:

* Replogle — the pseudobulk difference between a target's cells and the
  context's controls, in control-SD units (CLAUDE.md §3.2), one sub-block per
  screen for the same reason as the co-expression blocks.
* LINCS — the mean Level 5 signature over a target's rows.

The response profile is standardized per target before the decomposition, so
two targets are alike when their responses point the same way rather than
when they are the same size. Magnitude does not transfer from a LINCS
knockout to a challenge knockdown (§3.3); direction does.

Both read perturbation responses, so both honour the scope's held-out
targets.
"""

from __future__ import annotations

import numpy as np

from ..data import lincs as lincs_data
from ..data import manifest as manifest_mod
from ..data import replogle as replogle_data
from .blocks import TARGET, BlockInputs, BlockResult
from .decompose import resident_pca


def _features_for_targets(
    profiles: np.ndarray, targets: list[str], inputs: BlockInputs, source: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Decompose `(n_targets, n_genes)` responses into per-target features.

    The decomposition runs on the transpose so that the *targets* are the
    columns being described: `resident_pca` standardizes and reduces columns,
    and a column here is one target's whole response profile.
    """
    if profiles.shape[0] < 2:
        return None

    components, variance_ratio = resident_pca(
        profiles.T, inputs.cfg.priors.block_dim, rng=inputs.rng
    )

    index = inputs.index
    features = np.zeros((inputs.n_genes, components.shape[1]), dtype=np.float32)
    covered = np.zeros(inputs.n_genes, dtype=bool)
    for row, target in enumerate(targets):
        position = index.get(target)
        if position is None:
            continue  # a target that is not itself a challenge gene
        features[position] = components[row]
        covered[position] = True

    covered = inputs.apply_gene_restriction(covered, source=source)
    return features, covered, variance_ratio


class ReplogleKnockdownPhenotype:
    """Block 6a — the measured knockdown response, per Replogle screen."""

    name = "pert_phenotype_replogle"
    roles = (TARGET,)
    reads_responses = True

    def build(self, inputs: BlockInputs) -> list[BlockResult]:
        manifest = manifest_mod.load_manifest(inputs.paths.manifest)
        results = []

        for context, directory in sorted(inputs.paths.discover_phase2_contexts().items()):
            if not inputs.scope.allows_context(context):
                continue

            ctx = replogle_data.load_replogle_context(directory)
            obs, matrix, _ = replogle_data.load_pseudobulk(directory, manifest.control_label)

            labels = obs.index.astype(str).tolist()
            control_row = labels.index(manifest.control_label)
            control_profile = matrix[control_row]
            _, ctrl_std = ctx.control_stats()

            rows, targets = [], []
            for row, label in enumerate(labels):
                if label == manifest.control_label:
                    continue
                if not inputs.scope.allows_target(label):
                    continue
                rows.append(row)
                targets.append(label)
            if len(rows) < 2:
                continue

            # Control-SD units: the bridge between sources (CLAUDE.md §3.2).
            deltas = (matrix[rows] - control_profile) / ctrl_std

            source = f"{self.name}:{context}"
            built = _features_for_targets(deltas, targets, inputs, source)
            if built is None:
                continue
            features, covered, variance_ratio = built

            results.append(
                BlockResult(
                    name=source,
                    features=features,
                    covered=covered,
                    detail={
                        "source": f"Replogle {context} pseudobulk delta, control-SD units",
                        "n_targets": len(targets),
                        "n_targets_dropped_by_scope": len(
                            [t for t in labels if not inputs.scope.allows_target(t)]
                        ),
                        "n_response_genes": int(deltas.shape[1]),
                        "explained_variance_ratio": float(variance_ratio.sum()),
                    },
                )
            )

        return results


class LincsKnockdownPhenotype:
    """Block 6b — the mean LINCS signature of a target, over its rows.

    LINCS is CRISPR knockout and the challenge is interference, so this is
    used for which genes move and in which direction, not for how much.
    """

    name = "pert_phenotype_lincs"
    roles = (TARGET,)
    reads_responses = True

    def build(self, inputs: BlockInputs) -> list[BlockResult]:
        path = inputs.paths.phase1_lincs
        if not path.is_file():
            return []

        data = lincs_data.load_phase1(path)
        usable = data.sig_mask()
        labels = data.obs["target_gene"].astype(str).to_numpy()

        profiles, targets = [], []
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

        if len(targets) < 2:
            return []

        built = _features_for_targets(np.asarray(profiles), targets, inputs, self.name)
        if built is None:
            return []
        features, covered, variance_ratio = built

        return [
            BlockResult(
                name=self.name,
                features=features,
                covered=covered,
                detail={
                    "source": "phase1_lincs.h5ad layers['sig'], averaged per target",
                    "n_targets": len(targets),
                    "n_targets_dropped_by_scope": n_dropped,
                    "n_response_genes": int(data.n_genes),
                    "explained_variance_ratio": float(variance_ratio.sum()),
                },
            )
        ]
