"""What this round's submission must contain (CLAUDE.md §8).

Every field is read from the data: the gene axis from `gene_names.csv`, the
perturbations from `pert_counts.csv` (which §8 names as the authority), the
contexts and their labels from the control files themselves, and the cell
count from the manifest or from `pert_counts.csv` where it carries one. The
final round's D/E/F and its different target list therefore need no code
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..data import challenge as challenge_data
from ..data import manifest as manifest_mod
from ..data import reference
from ..paths import DataPaths


@dataclass(frozen=True)
class SubmissionSpec:
    """The round's requirements, as read from the release."""

    axis: tuple[str, ...]
    contexts: tuple[str, ...]
    targets: tuple[str, ...]
    cells_per_target: dict[str, int]
    pert_col: str
    context_col: str
    control_label: str

    @property
    def n_genes(self) -> int:
        return len(self.axis)

    @property
    def required_pairs(self) -> set[tuple[str, str]]:
        return {(c, t) for c in self.contexts for t in self.targets}

    @property
    def total_cells(self) -> int:
        return len(self.contexts) * sum(self.cells_per_target[t] for t in self.targets)

    def cells_for(self, target: str) -> int:
        return self.cells_per_target[target]

    def describe(self) -> dict[str, Any]:
        return {
            "n_genes": self.n_genes,
            "contexts": list(self.contexts),
            "n_perturbations": len(self.targets),
            "cells_per_perturbation": sorted(set(self.cells_per_target.values())),
            "expected_rows": self.total_cells,
            "obs_columns": [self.pert_col, self.context_col],
            "control_label_excluded": self.control_label,
        }


def load_spec(cfg: Config) -> SubmissionSpec:
    """Read the round's requirements from the release files."""
    paths = DataPaths(cfg)
    manifest = manifest_mod.load_manifest(paths.manifest)
    axis = reference.load_gene_names(paths.gene_names)
    perts = reference.load_pert_counts(paths.pert_counts, manifest.pert_col)

    # §8: the context label must be reused exactly as the control file spells
    # it, so it is read from there rather than from the file name.
    contexts = []
    for context in manifest.contexts:
        obs, _ = challenge_data.read_release_context(paths.context_h5ad(context))
        labels = sorted(set(obs[manifest.context_col].astype(str)))
        if labels != [context]:
            raise ValueError(
                f"{paths.context_h5ad(context)} carries context label(s) {labels}, "
                f"but the manifest calls this context {context!r}"
            )
        contexts.append(labels[0])

    counts = {
        target: int(
            perts.cells_per_pert[target] if perts.cells_per_pert else manifest.cells_per_pert
        )
        for target in perts.targets
    }
    return SubmissionSpec(
        axis=tuple(axis),
        contexts=tuple(contexts),
        targets=tuple(perts.targets),
        cells_per_target=counts,
        pert_col=manifest.pert_col,
        context_col=manifest.context_col,
        control_label=manifest.control_label,
    )
