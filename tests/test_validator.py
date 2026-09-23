"""The submission validator: one test per rule of CLAUDE.md §8.

Each test hands the validator a deliberately broken submission and asserts it
names the rule that is broken. The spec is built here rather than read from
`mini_data`, so nothing in the validator can be right only for one round's
sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from vccp.submit import validator
from vccp.submit.spec import SubmissionSpec

GENES = ("G1", "G2", "G3", "G4", "G5")
CONTEXTS = ("A", "B")
TARGETS = ("ADNP", "TAF4", "TDRD7")
CELLS = 4


@pytest.fixture
def spec():
    return SubmissionSpec(
        axis=GENES,
        contexts=CONTEXTS,
        targets=TARGETS,
        cells_per_target={t: CELLS for t in TARGETS},
        pert_col="target_gene",
        context_col="context",
        control_label="non-targeting",
    )


@pytest.fixture
def cfg(mini_cfg):
    return mini_cfg


def block(rows: int = CELLS, genes: int = len(GENES), seed: int = 0):
    """A well-formed block: sparse, integral, non-negative, no stored zeros."""
    rng = np.random.default_rng(seed)
    dense = rng.integers(0, 9, size=(rows, genes))
    matrix = sp.csr_matrix(dense.astype(np.int32))
    matrix.eliminate_zeros()
    return matrix


def good_blocks(spec, overrides=None):
    """Every (context, target) pair, with named pairs replaced."""
    overrides = overrides or {}
    blocks = []
    for i, context in enumerate(spec.contexts):
        for j, target in enumerate(spec.targets):
            key = (context, target)
            if key in overrides:
                replacement = overrides[key]
                if replacement is None:
                    continue
                blocks.append((context, target, replacement))
            else:
                blocks.append((context, target, block(seed=10 * i + j)))
    return blocks


def rules(result) -> dict[str, bool]:
    return {rule.rule: rule.ok for rule in result.rules}


def run(cfg, spec, blocks, axis=None):
    return validator.validate_blocks(
        cfg, spec, blocks, source="test", axis=axis or spec.axis
    )


def test_a_correct_submission_passes(cfg, spec):
    result = run(cfg, spec, good_blocks(spec))
    assert result.ok, result.summary()
    assert result.totals["n_rows"] == len(CONTEXTS) * len(TARGETS) * CELLS


def test_a_control_row_is_rejected(cfg, spec):
    blocks = good_blocks(spec) + [("A", spec.control_label, block())]
    result = run(cfg, spec, blocks)
    assert rules(result)["no_control_rows"] is False


def test_a_construct_id_is_rejected(cfg, spec):
    """§8: `ADNP`, never `ADNP-1`."""
    blocks = good_blocks(spec, {("A", "ADNP"): None}) + [("A", "ADNP-1", block())]
    result = run(cfg, spec, blocks)
    assert rules(result)["gene_symbols_not_construct_ids"] is False
    offending = next(r for r in result.rules if r.rule == "gene_symbols_not_construct_ids")
    assert offending.detail["examples"]["ADNP-1"] == "ADNP"


def test_a_label_outside_pert_counts_is_rejected(cfg, spec):
    blocks = good_blocks(spec) + [("A", "NOT_A_TARGET", block())]
    result = run(cfg, spec, blocks)
    assert rules(result)["labels_within_pert_counts"] is False


def test_a_missing_target_is_rejected(cfg, spec):
    result = run(cfg, spec, good_blocks(spec, {("B", "TAF4"): None}))
    assert rules(result)["every_context_times_every_perturbation"] is False


def test_a_wrong_cell_count_is_rejected(cfg, spec):
    result = run(cfg, spec, good_blocks(spec, {("A", "TAF4"): block(rows=CELLS + 1)}))
    assert rules(result)["cells_per_perturbation"] is False


def test_a_context_label_outside_the_release_is_rejected(cfg, spec):
    blocks = good_blocks(spec) + [("Z", "ADNP", block())]
    result = run(cfg, spec, blocks)
    assert rules(result)["context_labels"] is False


def test_a_non_integer_value_is_rejected(cfg, spec):
    broken = block().astype(np.float64)
    broken.data[0] += 0.5
    result = run(cfg, spec, good_blocks(spec, {("A", "ADNP"): broken}))
    assert rules(result)["finite_non_negative_integers"] is False


def test_a_negative_value_is_rejected(cfg, spec):
    broken = block().astype(np.float64)
    broken.data[0] = -1.0
    result = run(cfg, spec, good_blocks(spec, {("A", "ADNP"): broken}))
    assert rules(result)["finite_non_negative_integers"] is False


def test_a_non_finite_value_is_rejected(cfg, spec):
    broken = block().astype(np.float64)
    broken.data[0] = np.nan
    result = run(cfg, spec, good_blocks(spec, {("A", "ADNP"): broken}))
    assert rules(result)["finite_non_negative_integers"] is False


def test_a_cell_over_the_count_cap_is_rejected(cfg, spec):
    broken = block().astype(np.int64)
    broken[0, 0] = cfg.submission.max_counts_per_cell + 1
    result = run(cfg, spec, good_blocks(spec, {("A", "ADNP"): broken.tocsr()}))
    assert rules(result)["counts_per_cell"] is False


def test_explicitly_stored_zeros_are_rejected(cfg, spec):
    """§8: they count toward the cap, so `eliminate_zeros()` is not optional."""
    broken = block().astype(np.int32)
    broken.data[0] = 0  # stored, not removed
    result = run(cfg, spec, good_blocks(spec, {("A", "ADNP"): broken}))
    assert rules(result)["no_explicit_zeros"] is False


def test_a_dense_block_is_rejected(cfg, spec):
    dense = np.ones((CELLS, len(GENES)), dtype=np.int32)
    result = run(cfg, spec, good_blocks(spec, {("A", "ADNP"): dense}))
    assert rules(result)["sparse_storage"] is False


def test_the_wrong_gene_order_is_rejected(cfg, spec):
    shuffled = list(spec.axis)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    result = run(cfg, spec, good_blocks(spec), axis=shuffled)
    assert rules(result)["gene_axis"] is False
    rule = next(r for r in result.rules if r.rule == "gene_axis")
    assert rule.detail["position"] == 0


def test_a_different_gene_set_is_rejected(cfg, spec):
    result = run(cfg, spec, good_blocks(spec), axis=[*spec.axis[:-1], "OTHER"])
    assert rules(result)["gene_axis"] is False


def test_the_stored_entry_cap_is_enforced(cfg, spec):
    """The cap is a config key, so a test can shrink it and see it bite."""
    import dataclasses

    tightened = dataclasses.replace(
        cfg, submission=dataclasses.replace(cfg.submission, max_stored_entries=3)
    )
    result = run(tightened, spec, good_blocks(spec))
    assert rules(result)["stored_entries"] is False


def test_the_spec_is_read_from_the_release_not_hardcoded(mini_cfg):
    from vccp.submit.spec import load_spec

    spec = load_spec(mini_cfg)
    assert spec.n_genes > 0
    assert len(spec.contexts) == len(set(spec.contexts))
    assert spec.total_cells == len(spec.contexts) * sum(spec.cells_per_target.values())
