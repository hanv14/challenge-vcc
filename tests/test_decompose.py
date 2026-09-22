"""The dimension reduction underneath every prior block.

The streaming PCA is the piece most able to be subtly wrong: it accumulates
a randomized SVD over row blocks it never holds together, so an off-by-one in
the accumulation would still return plausible-looking numbers. These tests
compare it against an exact decomposition and against itself at different
block sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from vccp.priors.decompose import (
    column_stats,
    resident_pca,
    sparse_svd,
    streaming_pca,
)


def low_rank_matrix(n_rows=400, n_cols=60, rank=5, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((n_rows, rank))
    right = rng.standard_normal((rank, n_cols))
    signal = left @ (right * np.linspace(4.0, 1.0, rank)[:, None])
    return (signal + noise * rng.standard_normal((n_rows, n_cols))).astype(np.float32)


def blocks_of(matrix, size):
    def factory():
        for start in range(0, matrix.shape[0], size):
            yield matrix[start : start + size]

    return factory


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine of the principal angles between two column spaces.

    1.0 means the two sets of components span the same subspace. Components
    are only defined up to sign and rotation, so this is the right
    comparison rather than element-wise equality.
    """
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    return float(np.linalg.svd(qa.T @ qb, compute_uv=False).mean())


def test_column_stats_match_numpy():
    matrix = low_rank_matrix()
    mean, std, n_rows = column_stats(blocks_of(matrix, 37), matrix.shape[1])

    assert n_rows == matrix.shape[0]
    assert np.allclose(mean, matrix.mean(axis=0), atol=1e-4)
    assert np.allclose(std, matrix.std(axis=0), atol=1e-4)


def test_column_stats_are_independent_of_block_size():
    matrix = low_rank_matrix()
    a = column_stats(blocks_of(matrix, 400), matrix.shape[1])
    b = column_stats(blocks_of(matrix, 7), matrix.shape[1])
    assert np.allclose(a[0], b[0], atol=1e-5)
    assert np.allclose(a[1], b[1], atol=1e-5)


def test_streaming_pca_recovers_the_true_subspace():
    """Against an exact SVD of the same standardized matrix."""
    matrix = low_rank_matrix(rank=5)
    k = 5

    standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    _, _, vt = np.linalg.svd(standardized, full_matrices=False)
    exact = vt[:k].T

    components, ratio, n_rows = streaming_pca(
        blocks_of(matrix, 64), matrix.shape[1], k, n_iter=2, rng=np.random.default_rng(0)
    )

    assert components.shape == (matrix.shape[1], k)
    assert n_rows == matrix.shape[0]
    assert subspace_overlap(components, exact) > 0.99
    assert ratio.sum() > 0.5, "a rank-5 matrix's top 5 components should carry most variance"


def test_streaming_pca_is_independent_of_block_size():
    """The accumulation must not depend on how the rows were chunked."""
    matrix = low_rank_matrix()
    kwargs = dict(n_iter=2)

    wide, _, _ = streaming_pca(
        blocks_of(matrix, 400), matrix.shape[1], 6, rng=np.random.default_rng(1), **kwargs
    )
    narrow, _, _ = streaming_pca(
        blocks_of(matrix, 13), matrix.shape[1], 6, rng=np.random.default_rng(1), **kwargs
    )
    assert subspace_overlap(wide, narrow) > 0.999


def test_streaming_and_resident_pca_agree():
    matrix = low_rank_matrix()
    streamed, _, _ = streaming_pca(
        blocks_of(matrix, 50), matrix.shape[1], 6, n_iter=3, rng=np.random.default_rng(2)
    )
    resident, _ = resident_pca(matrix, 6)
    assert subspace_overlap(streamed, resident) > 0.99


def test_streaming_pca_rejects_a_block_iterator_that_changes_its_mind():
    """The two passes must see the same rows; a factory that yields a
    different number the second time is a bug, not something to average."""
    matrix = low_rank_matrix()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        rows = matrix if calls["n"] == 1 else matrix[:-10]
        for start in range(0, rows.shape[0], 50):
            yield rows[start : start + 50]

    with pytest.raises(ValueError, match="rows"):
        streaming_pca(flaky, matrix.shape[1], 4, rng=np.random.default_rng(0))


def test_constant_columns_do_not_produce_nan():
    """A gene with no variance in a context's controls must not divide by
    something near zero (CLAUDE.md §3.2)."""
    matrix = low_rank_matrix()
    matrix[:, 0] = 3.0

    components, _, _ = streaming_pca(
        blocks_of(matrix, 40), matrix.shape[1], 4, rng=np.random.default_rng(0)
    )
    assert np.isfinite(components).all()

    resident, _ = resident_pca(matrix, 4)
    assert np.isfinite(resident).all()


def test_fewer_components_than_asked_for_is_allowed():
    """A source with few rows cannot support many components; the caller
    pads rather than the decomposition inventing them."""
    matrix = low_rank_matrix(n_rows=4, n_cols=20)
    components, _, _ = streaming_pca(
        blocks_of(matrix, 2), matrix.shape[1], 32, rng=np.random.default_rng(0)
    )
    assert components.shape[0] == matrix.shape[1]
    assert components.shape[1] <= 32


def test_sparse_svd_on_a_membership_matrix():
    import scipy.sparse as sp

    # Two disjoint families of five genes each.
    rows = list(range(5)) + list(range(5, 10))
    cols = [0] * 5 + [1] * 5
    multihot = sp.csr_matrix((np.ones(10, dtype=np.float32), (rows, cols)), shape=(12, 2))

    features, ratio = sparse_svd(multihot, 4)
    assert features.shape[0] == 12
    assert np.isfinite(features).all()
    assert ratio.sum() <= 1.0 + 1e-6
    # The two genes outside both families carry no membership.
    assert np.allclose(features[10:], 0.0)


# --------------------------------------------------------------------------- #
# direction / magnitude separation, for the knockdown phenotype blocks
# --------------------------------------------------------------------------- #
def test_rowwise_standardize_separates_direction_from_size():
    from vccp.priors.decompose import rowwise_standardize

    direction = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    matrix = np.stack([direction * 10.0, direction * 0.01])

    standardized, rms = rowwise_standardize(matrix)

    # Same direction, wildly different size: the directions coincide and the
    # size survives only in `rms`.
    assert np.allclose(standardized[0], standardized[1], atol=1e-5)
    assert rms[0] > 100 * rms[1]


def test_rowwise_standardize_survives_a_flat_row():
    from vccp.priors.decompose import rowwise_standardize

    standardized, rms = rowwise_standardize(np.ones((2, 5), dtype=np.float32))
    assert np.isfinite(standardized).all()
    assert np.allclose(rms, 0.0)


def test_magnitude_weights_halve_at_the_floor():
    from vccp.priors.decompose import magnitude_weights

    weights = magnitude_weights(np.array([0.0, 1.0, 100.0]), floor=1.0)
    assert weights[0] == pytest.approx(0.0)
    assert weights[1] == pytest.approx(0.5)
    assert weights[2] > 0.99


def test_weighted_basis_is_set_by_the_targets_that_responded():
    """The point of the weighting: a crowd of null responses must not bend
    the basis toward its own noise (D17)."""
    from vccp.priors.decompose import (
        magnitude_weights,
        rowwise_standardize,
        weighted_direction_basis,
    )

    rng = np.random.default_rng(0)
    axis = rng.standard_normal(40)
    strong = np.outer(np.ones(3), axis) * 3 + 0.1 * rng.standard_normal((3, 40))
    null = 0.01 * rng.standard_normal((20, 40))
    profiles = np.vstack([strong, null]).astype(np.float32)

    _, rms = rowwise_standardize(profiles)
    weights = magnitude_weights(rms, 0.5 * float(np.median(rms)))

    _, _, weighted_basis = weighted_direction_basis(profiles, 4, weights)
    _, _, flat_basis = weighted_direction_basis(profiles, 4, None)

    unit_axis = axis / np.linalg.norm(axis)
    assert abs(weighted_basis[:, 0] @ unit_axis) > abs(flat_basis[:, 0] @ unit_axis)


def test_every_row_is_projected_into_the_basis_including_the_weak_ones():
    """Weighting chooses the basis; it does not drop rows."""
    from vccp.priors.decompose import weighted_direction_basis

    rng = np.random.default_rng(1)
    profiles = rng.standard_normal((12, 30)).astype(np.float32)
    weights = np.concatenate([np.ones(6), np.full(6, 1e-6)]).astype(np.float32)

    features, _, _ = weighted_direction_basis(profiles, 4, weights)
    assert features.shape[0] == profiles.shape[0]
    assert np.isfinite(features).all()
    assert np.abs(features[6:]).sum() > 0, "down-weighted rows still get features"


def test_weighted_basis_rejects_mismatched_weights():
    from vccp.priors.decompose import weighted_direction_basis

    with pytest.raises(ValueError, match="one entry per row"):
        weighted_direction_basis(np.zeros((5, 3), np.float32), 2, np.ones(4))
