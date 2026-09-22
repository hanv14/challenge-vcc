"""One gene coordinate system, checked across every file that carries genes.

`challenge_idx = i` must mean the i-th gene of `gene_names.csv` in every file
(CLAUDE.md §3.1). Files holding the full axis must hold it in order; files
holding a subset must carry a `challenge_idx` that lands on the right symbol.
A single disagreement here silently corrupts every downstream index, which is
why it is tested on its own rather than only inside `check-data`.
"""

from __future__ import annotations

import numpy as np
import pytest

from vccp.data import challenge, gene_split, lincs, manifest as manifest_mod, reference, replogle
from vccp.paths import DataPaths


@pytest.fixture
def paths(mini_cfg) -> DataPaths:
    return DataPaths(mini_cfg)


@pytest.fixture
def axis(paths) -> list[str]:
    """The one true gene order: the submission's own gene axis."""
    return reference.load_gene_names(paths.gene_names)


def assert_subset_lands_on_axis(symbols, challenge_idx, axis, label):
    idx = np.asarray(challenge_idx)
    assert idx.min() >= 0 and idx.max() < len(axis), f"{label}: challenge_idx out of range"
    assert [axis[i] for i in idx] == list(symbols), f"{label}: challenge_idx points elsewhere"


# --- files carrying the full axis, in order -------------------------------- #
def test_challenge_genes_tsv_matches_the_axis(paths, axis):
    assert list(reference.load_gene_reference(paths.challenge_genes).symbols) == axis


def test_gene_split_matches_the_axis(paths, axis):
    assert gene_split.load_gene_split(paths.gene_split).symbols == axis


def test_phase3_genes_matches_the_axis(paths, axis):
    frame = challenge.load_phase3_genes(paths.phase3_genes)
    assert frame["gene_symbol"].astype(str).tolist() == axis
    assert frame["challenge_idx"].tolist() == list(range(len(axis)))


def test_every_control_file_matches_the_axis(paths, axis):
    found = paths.discover_phase3_controls()
    assert found, "expected control files"
    for context, path in found.items():
        controls = challenge.load_challenge_controls(path, context)
        assert controls.var.index.astype(str).tolist() == axis, context
        assert controls.var["challenge_idx"].tolist() == list(range(len(axis))), context


def test_every_release_file_matches_the_axis(paths, axis):
    found = paths.discover_context_files()
    assert found, "expected release context files"
    for context, path in found.items():
        _, var_names = challenge.read_release_context(path)
        assert [str(v) for v in var_names] == axis, context


# --- files carrying a subset ------------------------------------------------ #
def test_phase1_panel_lands_on_the_axis(paths, axis):
    obs, var = lincs.read_phase1_metadata(paths.phase1_lincs)
    assert_subset_lands_on_axis(
        var.index.astype(str), var["challenge_idx"].to_numpy(), axis, "phase1_lincs.h5ad"
    )


def test_every_replogle_context_lands_on_the_axis(paths, axis):
    found = paths.discover_phase2_contexts()
    assert found, "expected Replogle context directories"
    for name, directory in found.items():
        ctx = replogle.load_replogle_context(directory)
        assert_subset_lands_on_axis(
            ctx.genes["gene_symbol"].astype(str), ctx.challenge_idx, axis, name
        )


def test_replogle_panel_genes_are_panel_in_the_split(paths):
    """A context's panel/rest labelling must agree with the global split —
    otherwise Phase 2 would feed rest genes in as if they were panel."""
    split = gene_split.load_gene_split(paths.gene_split)
    is_panel = split.is_panel()
    for name, directory in paths.discover_phase2_contexts().items():
        ctx = replogle.load_replogle_context(directory)
        assert is_panel[ctx.panel_challenge_idx].all(), f"{name}: a non-panel gene labelled panel"
        assert not is_panel[ctx.rest_challenge_idx].any(), f"{name}: a panel gene labelled rest"


def test_phase1_genes_are_all_panel(paths):
    """Phase 1 measures the LINCS landmarks, which are exactly the panel."""
    split = gene_split.load_gene_split(paths.gene_split)
    _, var = lincs.read_phase1_metadata(paths.phase1_lincs)
    assert split.is_panel()[var["challenge_idx"].to_numpy()].all()


# --- the perturbation axis -------------------------------------------------- #
def test_targets_table_uses_the_manifest_labels(paths):
    man = manifest_mod.load_manifest(paths.manifest)
    targets = challenge.load_targets(paths.phase3_targets)
    perts = reference.load_pert_counts(paths.pert_counts, man.pert_col)

    assert set(targets["context"].astype(str)) == set(man.contexts)
    assert set(targets["target_gene"].astype(str)) == set(perts.targets)
    assert man.control_label not in set(targets["target_gene"].astype(str))


def test_target_labels_are_gene_symbols_not_construct_ids(paths):
    """`ADNP`, never `ADNP-1` (CLAUDE.md §8)."""
    man = manifest_mod.load_manifest(paths.manifest)
    perts = reference.load_pert_counts(paths.pert_counts, man.pert_col)
    axis = set(reference.load_gene_names(paths.gene_names))

    suffixed = [t for t in perts.targets if t.rsplit("-", 1)[-1].isdigit() and t not in axis]
    assert not suffixed, f"construct ids in pert_counts.csv: {suffixed[:5]}"
