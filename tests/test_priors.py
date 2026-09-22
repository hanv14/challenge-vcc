"""The gene vocabulary: blocks, roles, coverage, leakage (checklist 1–3).

Every assertion is about the contract, not about mini_data's sizes — the
gene count, the block widths and which sources cover what all differ on the
server.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from vccp.data import reference
from vccp.paths import DataPaths, RunPaths
from vccp.priors.blocks import FEATURE, TARGET, BlockInputs, BlockResult
from vccp.priors.build import PriorFeatures, build_priors, default_blocks
from vccp.priors.coexpr import ChallengeControlCoexpression, ReplogleControlCoexpression
from vccp.priors.hgnc_block import HgncGeneGroups
from vccp.priors.lincs_blocks import GmtCoMembership, LincsSignatureCovariation
from vccp.priors.phenotype import LincsKnockdownPhenotype, ReplogleKnockdownPhenotype
from vccp.priors.plm import ProteinLanguageModel
from vccp.priors.scope import FULL_SCOPE, PriorScope
from vccp.priors.stage import run_priors


def block_inputs(cfg, scope=FULL_SCOPE) -> BlockInputs:
    paths = DataPaths(cfg)
    return BlockInputs(
        cfg=cfg,
        paths=paths,
        axis=reference.load_gene_names(paths.gene_names),
        scope=scope,
        rng=np.random.default_rng(cfg.seed),
    )


# --------------------------------------------------------------------------- #
# item 1 — the vocabulary covers every challenge gene, in order
# --------------------------------------------------------------------------- #
def test_vocabulary_covers_every_challenge_gene_in_order(session_cfg, built_priors):
    axis = reference.load_gene_names(DataPaths(session_cfg).gene_names)
    assert built_priors.gene_symbols == axis
    for role in (FEATURE, TARGET):
        assert built_priors.priors[role].shape[0] == len(axis)


def test_priors_are_finite_and_non_trivial(built_priors):
    for role in (FEATURE, TARGET):
        matrix = built_priors.priors[role]
        assert np.isfinite(matrix).all()
        assert matrix.shape[1] > 0
        assert matrix.std() > 0, f"the {role} prior carries no variation"


# --------------------------------------------------------------------------- #
# item 3 — two roles
# --------------------------------------------------------------------------- #
def test_feature_role_and_target_role_matrices_differ_and_are_both_stored(built_priors):
    feature = built_priors.priors[FEATURE]
    target = built_priors.priors[TARGET]

    assert feature.shape[1] != target.shape[1] or not np.allclose(feature, target)
    # The knockdown phenotype describes a gene only as a target, so the
    # target role is the wider of the two (CLAUDE.md §4.1).
    assert set(built_priors.layouts[FEATURE].block_names) < set(
        built_priors.layouts[TARGET].block_names
    ) or set(built_priors.layouts[TARGET].block_names) - set(
        built_priors.layouts[FEATURE].block_names
    )


def test_only_the_target_role_carries_the_knockdown_phenotype(built_priors):
    feature_blocks = built_priors.layouts[FEATURE].block_names
    target_blocks = built_priors.layouts[TARGET].block_names

    assert any(name.startswith("pert_phenotype") for name in target_blocks)
    assert not any(name.startswith("pert_phenotype") for name in feature_blocks)


# --------------------------------------------------------------------------- #
# item 2 — the blocks, their widths, their missing flags
# --------------------------------------------------------------------------- #
def test_every_specified_block_is_built(built_priors):
    """Blocks 1–6 of CLAUDE.md §4.1; block 7 only when configured."""
    built = set(built_priors.coverage.loc[built_priors.coverage["built"], "block"])

    def present(prefix):
        return any(name == prefix or name.startswith(prefix + ":") for name in built)

    for prefix in (
        "challenge_coexpr",
        "replogle_coexpr",
        "lincs_sig_covariation",
        "gmt_comembership",
        "hgnc_gene_groups",
        "pert_phenotype_replogle",
        "pert_phenotype_lincs",
    ):
        assert present(prefix), f"block {prefix} was not built"


def test_layout_accounts_for_every_column(built_priors):
    for role in (FEATURE, TARGET):
        layout = built_priors.layouts[role]
        assert layout.total_width == built_priors.priors[role].shape[1]
        # Each block occupies its width plus exactly one flag column.
        for offset, width, flag in zip(layout.offsets, layout.widths, layout.flag_columns):
            assert flag == offset + width
        expected = sum(w + 1 for w in layout.widths)
        assert layout.total_width == expected


def test_missing_flag_marks_the_genes_a_block_does_not_cover(built_priors):
    """A gene with no data from a source must be distinguishable from a gene
    whose value from that source happens to be zero."""
    role = FEATURE
    layout = built_priors.layouts[role]
    matrix = built_priors.priors[role]

    for name, offset, width, flag in zip(
        layout.block_names, layout.offsets, layout.widths, layout.flag_columns
    ):
        flags = matrix[:, flag]
        assert set(np.unique(flags)) <= {0.0, 1.0}, f"{name}: flag is not an indicator"
        uncovered = flags == 1.0
        if uncovered.any():
            block = matrix[uncovered, offset : offset + width]
            assert np.allclose(block, 0.0), f"{name}: uncovered genes carry values"


def test_covered_features_are_standardized(built_priors):
    """Each block is z-scored over the genes it covers, so no one block
    dominates the projection by being on a larger scale."""
    role = FEATURE
    layout = built_priors.layouts[role]
    matrix = built_priors.priors[role]

    for name, offset, width, flag in zip(
        layout.block_names, layout.offsets, layout.widths, layout.flag_columns
    ):
        covered = matrix[:, flag] == 0.0
        if covered.sum() < 2:
            continue
        block = matrix[covered, offset : offset + width]
        assert np.abs(block.mean(axis=0)).max() < 1e-3, name
        std = block.std(axis=0)
        assert np.abs(std[std > 0] - 1.0).max() < 1e-2, name


def test_coverage_table_has_a_row_per_block(built_priors):
    coverage = built_priors.coverage
    assert not coverage.empty
    for column in ("block", "roles", "built", "n_covered", "fraction_covered", "width"):
        assert column in coverage.columns
    assert coverage["fraction_covered"].between(0.0, 1.0).all()


def test_challenge_coexpression_covers_every_gene(session_cfg):
    """The only block that can reach a gene measured nowhere else."""
    inputs = block_inputs(session_cfg)
    results = ChallengeControlCoexpression().build(inputs)
    assert len(results) == 1
    assert results[0].n_covered == inputs.n_genes


def test_replogle_coexpression_returns_one_block_per_screen(session_cfg):
    """Screens measure different genes, so their components are not
    comparable and are kept apart."""
    inputs = block_inputs(session_cfg)
    contexts = inputs.paths.discover_phase2_contexts()
    results = ReplogleControlCoexpression().build(inputs)

    assert len(results) == len(contexts)
    for result in results:
        assert result.name.startswith("replogle_coexpr:")
        assert 0 < result.n_covered < inputs.n_genes


def test_a_block_covers_exactly_the_genes_its_source_measures(session_cfg):
    inputs = block_inputs(session_cfg)
    from vccp.data import replogle as replogle_data

    for result in ReplogleControlCoexpression().build(inputs):
        context = result.name.split(":", 1)[1]
        ctx = replogle_data.load_replogle_context(inputs.paths.phase2_context_dir(context))
        assert np.array_equal(np.flatnonzero(result.covered), np.sort(ctx.challenge_idx))


# --------------------------------------------------------------------------- #
# block 7 — optional, and never downloaded
# --------------------------------------------------------------------------- #
def test_block7_is_absent_unless_configured(session_cfg):
    assert ProteinLanguageModel().build(block_inputs(session_cfg)) == []


def test_block7_fails_loudly_rather_than_downloading(session_cfg, tmp_path):
    cfg = dataclasses.replace(
        session_cfg,
        priors=dataclasses.replace(session_cfg.priors, plm_path=str(tmp_path / "nope.npz")),
    )
    with pytest.raises(FileNotFoundError, match="never downloads"):
        ProteinLanguageModel().build(block_inputs(cfg))


def test_block7_is_built_when_a_file_is_supplied(session_cfg, tmp_path):
    axis = reference.load_gene_names(DataPaths(session_cfg).gene_names)
    chosen = axis[:40]
    path = tmp_path / "plm.npz"
    np.savez(
        path,
        gene_symbols=np.asarray(chosen, dtype=object),
        embeddings=np.random.default_rng(0).standard_normal((len(chosen), 8)).astype(np.float32),
    )

    cfg = dataclasses.replace(
        session_cfg, priors=dataclasses.replace(session_cfg.priors, plm_path=str(path))
    )
    results = ProteinLanguageModel().build(block_inputs(cfg))
    assert len(results) == 1
    assert results[0].n_covered == len(chosen)


# --------------------------------------------------------------------------- #
# leakage — the rehearsal's scopes are honoured (CLAUDE.md §4.1)
# --------------------------------------------------------------------------- #
def test_blocks_declare_whether_they_read_responses():
    response_readers = {
        LincsSignatureCovariation,
        GmtCoMembership,
        ReplogleKnockdownPhenotype,
        LincsKnockdownPhenotype,
    }
    for block in default_blocks():
        expected = type(block) in response_readers
        assert block.reads_responses is expected, type(block).__name__


def test_excluded_targets_are_dropped_from_response_blocks(session_cfg):
    """A held-out target must not shape the vocabulary through a prior built
    from its own knockdown."""
    inputs = block_inputs(session_cfg)
    full = ReplogleKnockdownPhenotype().build(inputs)
    assert full

    covered_symbols = [
        inputs.axis[i] for i in np.flatnonzero(full[0].covered)
    ]
    held_out = set(covered_symbols[:3])

    scope = PriorScope(exclude_targets=frozenset(held_out), label="test")
    restricted = ReplogleKnockdownPhenotype().build(block_inputs(session_cfg, scope))

    for result in restricted:
        still_covered = {inputs.axis[i] for i in np.flatnonzero(result.covered)}
        assert not (still_covered & held_out), result.name


def test_excluded_contexts_are_skipped(session_cfg):
    inputs = block_inputs(session_cfg)
    contexts = list(inputs.paths.discover_phase2_contexts())
    dropped = contexts[0]

    scope = PriorScope(exclude_contexts=frozenset({dropped}), label="test")
    results = ReplogleControlCoexpression().build(block_inputs(session_cfg, scope))
    assert all(not r.name.endswith(f":{dropped}") for r in results)
    assert len(results) == len(contexts) - 1


def test_restricted_genes_are_hidden_from_every_source_but_one(session_cfg):
    """Rehearsal variant 3 pretends genes are unmeasured; a prior built from
    the data that measured them would undo that."""
    inputs = block_inputs(session_cfg)
    hidden = set(inputs.axis[:25])

    scope = PriorScope(
        restricted_genes=frozenset(hidden),
        restricted_genes_source="challenge_coexpr",
        label="unseen-genes",
    )
    restricted_inputs = block_inputs(session_cfg, scope)

    allowed = ChallengeControlCoexpression().build(restricted_inputs)[0]
    assert allowed.n_covered == inputs.n_genes, "the permitted source still sees them"

    for result in HgncGeneGroups().build(restricted_inputs):
        covered = {inputs.axis[i] for i in np.flatnonzero(result.covered)}
        assert not (covered & hidden), result.name


def test_scope_cache_keys_distinguish_scopes():
    a = PriorScope(label="a")
    b = PriorScope(exclude_targets=frozenset({"X"}), label="a")
    c = PriorScope(label="b")
    assert len({a.cache_key(), b.cache_key(), c.cache_key()}) == 3
    assert FULL_SCOPE.is_unrestricted
    assert not b.is_unrestricted


# --------------------------------------------------------------------------- #
# the stage and its artifacts
# --------------------------------------------------------------------------- #
def test_priors_stage_writes_its_three_artifacts(mini_cfg):
    summary = run_priors(mini_cfg)
    run_paths = RunPaths(mini_cfg)

    assert run_paths.prior_features.is_file()
    assert run_paths.prior_coverage.is_file()
    assert run_paths.prior_checks.is_file()
    assert summary["n_blocks_built"] > 0

    checks = json.loads(run_paths.prior_checks.read_text())
    assert "neighbour_check" in checks
    assert "response_correlation_check" in checks


def test_saved_priors_round_trip(tmp_path, built_priors):
    path = tmp_path / "prior_features.npz"
    built_priors.save(path)
    loaded = PriorFeatures.load(path)

    assert loaded.gene_symbols == built_priors.gene_symbols
    for role in (FEATURE, TARGET):
        assert np.array_equal(loaded.priors[role], built_priors.priors[role])
        assert loaded.layouts[role].block_names == built_priors.layouts[role].block_names


def test_block_result_zeroes_uncovered_rows():
    features = np.ones((5, 3), dtype=np.float32)
    covered = np.array([True, False, True, False, True])
    result = BlockResult("t", features, covered)

    assert np.allclose(result.features[~covered], 0.0)
    assert np.allclose(result.features[covered], 1.0)
    assert result.n_covered == 3


def test_block_result_rejects_a_mismatched_mask():
    with pytest.raises(ValueError, match="coverage mask"):
        BlockResult("t", np.ones((5, 3)), np.ones(4, dtype=bool))
