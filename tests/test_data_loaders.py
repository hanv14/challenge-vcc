"""Every loader of CLAUDE.md §3, against mini_data.

The assertions are about *shape and contract*, never about mini_data's
particular sizes — the same tests must hold on the server, where every count
is different.
"""

from __future__ import annotations

import numpy as np
import pytest

from vccp.data import challenge, coverage, gene_split, lincs, manifest as manifest_mod, reference
from vccp.data import replogle
from vccp.paths import DataPaths


@pytest.fixture
def paths(mini_cfg) -> DataPaths:
    return DataPaths(mini_cfg)


# --------------------------------------------------------------------------- #
# reference
# --------------------------------------------------------------------------- #
def test_gene_reference_is_ordered_and_unique(paths):
    ref = reference.load_gene_reference(paths.challenge_genes)
    assert len(ref) > 0
    assert len(set(ref.symbols)) == len(ref), "challenge symbols must be unique"
    assert len(ref.ensembl) == len(ref)
    assert ref.index[ref.symbols[0]] == 0
    assert ref.index[ref.symbols[-1]] == len(ref) - 1


def test_gene_reference_positions_round_trip(paths):
    ref = reference.load_gene_reference(paths.challenge_genes)
    sample = [ref.symbols[0], ref.symbols[len(ref) // 2], ref.symbols[-1]]
    assert np.array_equal(ref.positions(sample), [0, len(ref) // 2, len(ref) - 1])


def test_gene_reference_rejects_unknown_symbol(paths):
    ref = reference.load_gene_reference(paths.challenge_genes)
    with pytest.raises(KeyError, match="not challenge genes"):
        ref.positions(["NOT_A_REAL_GENE"])


def test_gene_names_matches_challenge_genes(paths):
    ref = reference.load_gene_reference(paths.challenge_genes)
    assert reference.load_gene_names(paths.gene_names) == list(ref.symbols)


def test_pert_counts_and_challenge_perts_agree(paths):
    perts = reference.load_challenge_perts(paths.challenge_perts)
    counts = reference.load_pert_counts(paths.pert_counts)
    assert set(perts) == set(counts.targets)
    assert len(counts) == len(set(counts.targets))


def test_pert_counts_detects_a_per_target_count_column(tmp_path):
    path = tmp_path / "pert_counts.csv"
    path.write_text("target_gene,n_cells\nA,7\nB,7\n")
    counts = reference.load_pert_counts(path)
    assert counts.count_column == "n_cells"
    assert counts.cells_per_pert == {"A": 7, "B": 7}


def test_pert_counts_without_a_count_column_reports_none(paths):
    counts = reference.load_pert_counts(paths.pert_counts)
    assert counts.cells_per_pert is None or counts.count_column is not None


def test_hgnc_loads_under_either_name(paths):
    frame = reference.load_hgnc(paths.hgnc)
    for column in reference.HGNC_COLUMNS:
        assert column in frame.columns


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def test_manifest_supplies_the_round_shape(paths):
    man = manifest_mod.load_manifest(paths.manifest)
    assert man.pert_col and man.context_col and man.control_label
    assert man.cells_per_pert > 0
    assert len(man.contexts) > 0
    for context in man.contexts:
        assert man.control_cells(context) is not None


def test_manifest_rejects_a_missing_key(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"pert_col": "target_gene"}')
    with pytest.raises(ValueError, match="missing key"):
        manifest_mod.load_manifest(path)


# --------------------------------------------------------------------------- #
# gene split
# --------------------------------------------------------------------------- #
def test_gene_split_partitions_the_gene_axis(paths):
    split = gene_split.load_gene_split(paths.gene_split)
    assert split.n_panel + split.n_rest == split.n_genes
    assert not set(split.panel_idx) & set(split.rest_idx)
    assert split.is_panel().sum() == split.n_panel
    assert split.symbols == reference.load_gene_names(paths.gene_names)


def test_gene_split_panel_and_rest_symbols_line_up(paths):
    split = gene_split.load_gene_split(paths.gene_split)
    symbols = np.asarray(split.symbols)
    assert split.panel_symbols == symbols[split.panel_idx].tolist()
    assert split.rest_symbols == symbols[split.rest_idx].tolist()


def test_gene_split_rejects_a_reordered_index(tmp_path, paths):
    import pandas as pd

    frame = pd.read_csv(paths.gene_split)
    frame["challenge_idx"] = frame["challenge_idx"][::-1].to_numpy()
    path = tmp_path / "gene_split.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="challenge_idx"):
        gene_split.load_gene_split(path)


# --------------------------------------------------------------------------- #
# phase 1
# --------------------------------------------------------------------------- #
def test_phase1_layers_and_masks(paths):
    data = lincs.load_phase1(paths.phase1_lincs)
    assert set(data.layers) == set(lincs.LAYERS)
    for name, layer in data.layers.items():
        assert layer.shape == (data.n_rows, data.n_genes), name
    assert data.gmt_mask().shape == (data.n_rows,)
    assert data.sig_mask().shape == (data.n_rows,)
    assert len(data.cell_lines) > 0
    assert len(data.targets) > 0


def test_phase1_gmt_layer_is_bounded(paths):
    """`gmt` is (#up - #down)/#signatures, so it lives in [-1, 1] (§3.3)."""
    data = lincs.load_phase1(paths.phase1_lincs)
    gmt = data.layers["gmt"][data.gmt_mask()]
    assert gmt.min() >= -1.0 - 1e-6
    assert gmt.max() <= 1.0 + 1e-6


def test_phase1_challenge_idx_lands_on_the_gene_axis(paths):
    data = lincs.load_phase1(paths.phase1_lincs)
    axis = reference.load_gene_names(paths.gene_names)
    idx = data.challenge_idx
    assert idx.min() >= 0 and idx.max() < len(axis)
    assert [axis[i] for i in idx] == data.var.index.astype(str).tolist()


def test_phase1_metadata_read_skips_the_layers(paths):
    obs, var = lincs.read_phase1_metadata(paths.phase1_lincs)
    assert len(obs) > 0 and len(var) > 0


# --------------------------------------------------------------------------- #
# phase 2 / Replogle
# --------------------------------------------------------------------------- #
def test_replogle_contexts_are_discovered(paths):
    contexts = paths.discover_phase2_contexts()
    assert contexts, "expected at least one Replogle context directory"
    for name, directory in contexts.items():
        ctx = replogle.load_replogle_context(directory)
        assert ctx.name == name
        assert ctx.n_panel + ctx.n_rest == len(ctx.genes)
        assert len(ctx.targets) > 0


def test_replogle_measured_mask_is_on_the_full_axis(paths):
    axis = reference.load_gene_names(paths.gene_names)
    name, directory = next(iter(paths.discover_phase2_contexts().items()))
    ctx = replogle.load_replogle_context(directory)
    mask = ctx.measured_mask(len(axis))
    assert mask.shape == (len(axis),)
    assert mask.sum() == len(ctx.genes)
    assert np.array_equal(np.flatnonzero(mask), np.sort(ctx.challenge_idx))


def test_replogle_control_stats_are_floored(paths):
    name, directory = next(iter(paths.discover_phase2_contexts().items()))
    ctx = replogle.load_replogle_context(directory)
    mean, std = ctx.control_stats()
    assert mean.shape == std.shape == (len(ctx.genes),)
    assert std.min() >= replogle.MIN_CTRL_STD, "ctrl_std must never reach zero (§3.2)"


def test_replogle_cells_come_back_split_panel_rest(paths):
    """Cells are sampled through `ReplogleCells`, never loaded whole (§3.3)."""
    name, directory = next(iter(paths.discover_phase2_contexts().items()))
    ctx = replogle.load_replogle_context(directory)
    rng = np.random.default_rng(0)

    n = 8
    panel, rest = ctx.control_cells(n, rng)
    assert panel.shape == (n, ctx.n_panel)
    assert rest.shape == (n, ctx.n_rest)
    assert np.isfinite(panel).all() and np.isfinite(rest).all()
    assert panel.min() >= 0.0, "log1p(CP10K) is non-negative"

    target = ctx.targets[0]
    kd_panel, kd_rest = ctx.cells_for_target(target, n, rng)
    assert kd_panel.shape[1] == ctx.n_panel
    assert kd_rest.shape[1] == ctx.n_rest


def test_replogle_pseudobulk_carries_the_control(paths, mini_cfg):
    man = manifest_mod.load_manifest(DataPaths(mini_cfg).manifest)
    name, directory = next(iter(paths.discover_phase2_contexts().items()))
    obs, X, var = replogle.load_pseudobulk(directory, man.control_label)
    assert man.control_label in obs.index
    assert X.shape == (len(obs), len(var))
    assert np.isfinite(X).all()


def test_fold_expr_is_positive_and_per_target(paths):
    bulk = {n: p for n, p in paths.discover_replogle_bulk().items() if n.endswith("_raw_bulk")}
    assert bulk, "expected at least one *_raw_bulk.h5ad"
    for path in bulk.values():
        series = replogle.load_fold_expr(path)
        assert len(series) > 0
        assert series.index.is_unique, "one fold_expr per target, promoters reduced"
        assert (series > 0).all() and np.isfinite(series).all()


# --------------------------------------------------------------------------- #
# phase 3 / challenge
# --------------------------------------------------------------------------- #
def test_challenge_controls_expose_both_unit_systems(paths, mini_cfg):
    man = manifest_mod.load_manifest(paths.manifest)
    context = man.contexts[0]
    controls = challenge.load_challenge_controls(paths.phase3_controls(context), context)

    rows = np.arange(min(5, controls.n_cells))
    counts = controls.counts(rows)
    lognorm = controls.lognorm(rows)

    assert counts.shape == lognorm.shape == (len(rows), controls.n_genes)
    assert np.all(counts >= 0)
    assert np.allclose(counts, np.round(counts)), "layers['counts'] must be integral"
    assert lognorm.min() >= 0.0


def test_challenge_controls_stream_in_blocks(paths, mini_cfg):
    man = manifest_mod.load_manifest(paths.manifest)
    context = man.contexts[0]
    controls = challenge.load_challenge_controls(paths.phase3_controls(context), context)

    block_size = max(1, controls.n_cells // 3)
    blocks = list(controls.iter_lognorm_blocks(block_size))
    assert sum(b.shape[0] for b in blocks) == controls.n_cells
    assert all(b.shape[1] == controls.n_genes for b in blocks)


def test_challenge_controls_var_is_the_full_gene_axis(paths):
    axis = reference.load_gene_names(paths.gene_names)
    for context, path in paths.discover_phase3_controls().items():
        controls = challenge.load_challenge_controls(path, context)
        assert controls.var.index.astype(str).tolist() == axis


def test_targets_table_covers_contexts_times_perturbations(paths):
    man = manifest_mod.load_manifest(paths.manifest)
    targets = challenge.load_targets(paths.phase3_targets)
    perts = reference.load_pert_counts(paths.pert_counts, man.pert_col)
    assert len(targets) == len(man.contexts) * len(perts)


def test_release_context_holds_controls_only(paths):
    man = manifest_mod.load_manifest(paths.manifest)
    axis = reference.load_gene_names(paths.gene_names)
    for context, path in paths.discover_context_files().items():
        obs, var_names = challenge.read_release_context(path)
        assert [str(v) for v in var_names] == axis
        assert set(obs[man.pert_col].astype(str)) == {man.control_label}
        assert set(obs[man.context_col].astype(str)) == {context}


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #
def test_coverage_tables_expose_their_sources(paths):
    pert = coverage.load_pert_coverage(paths.pert_coverage)
    gene = coverage.load_gene_coverage(paths.gene_coverage)
    assert pert.sources and gene.sources
    assert set(pert.counts_per_source()) <= set(pert.sources)
    assert pert.covered_by_any().shape[0] == len(pert.frame)
