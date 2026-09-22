"""Dimension reduction for the prior blocks.

Two shapes come up:

* **Tall and streamed** — cells x genes, where the cells arrive in blocks
  from disk and the gene count is the full challenge axis. A gene-gene
  correlation matrix would be 18,533^2 (1.4 GB before its eigendecomposition),
  so this never forms one: a randomized SVD of the standardized data matrix
  gives the same principal directions, and its right singular vectors *are*
  the PCA of that correlation matrix.
* **Small and resident** — signatures x genes, targets x genes, or a sparse
  gene x group multi-hot, all of which fit comfortably.

Both return `(n_cols, k)` gene features.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

#: Extra directions sampled beyond the ones asked for. Randomized SVD is
#: much more accurate with a little oversampling.
OVERSAMPLE = 10

#: Columns whose standard deviation is below this are treated as constant:
#: they carry no correlation structure and are scaled by 1 rather than by
#: something near zero.
MIN_STD = 1e-8

BlockIterFactory = Callable[[], Iterator[np.ndarray]]


class ColumnStats:
    """Streaming mean and standard deviation per column."""

    def __init__(self, n_cols: int) -> None:
        self.n = 0
        self._sum = np.zeros(n_cols, dtype=np.float64)
        self._sumsq = np.zeros(n_cols, dtype=np.float64)

    def update(self, block: np.ndarray) -> None:
        self.n += block.shape[0]
        self._sum += block.sum(axis=0, dtype=np.float64)
        self._sumsq += np.einsum("ij,ij->j", block, block, dtype=np.float64)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n == 0:
            raise ValueError("no rows were seen")
        mean = self._sum / self.n
        var = np.maximum(self._sumsq / self.n - mean**2, 0.0)
        std = np.sqrt(var)
        return mean, np.where(std < MIN_STD, 1.0, std)


def column_stats(block_iter: BlockIterFactory, n_cols: int) -> tuple[np.ndarray, np.ndarray, int]:
    stats = ColumnStats(n_cols)
    for block in block_iter():
        stats.update(np.asarray(block, dtype=np.float32))
    mean, std = stats.finish()
    return mean, std, stats.n


def streaming_pca(
    block_iter: BlockIterFactory,
    n_cols: int,
    k: int,
    *,
    n_iter: int = 2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """PCA over the columns of a row-streamed matrix.

    Returns `(components, explained_variance_ratio, n_rows)` where
    `components` is `(n_cols, k_effective)` — one vector per column, which
    for a cells x genes matrix is one vector per gene.

    `k_effective` can be smaller than `k` when the matrix has fewer rows
    than components asked for; the caller pads.
    """
    rng = rng or np.random.default_rng(0)
    mean, std, n_rows = column_stats(block_iter, n_cols)
    if n_rows < 2:
        raise ValueError(f"need at least 2 rows to estimate covariance, got {n_rows}")

    width = min(k + OVERSAMPLE, n_cols, n_rows)

    def standardized() -> Iterator[np.ndarray]:
        for block in block_iter():
            yield (np.asarray(block, dtype=np.float32) - mean) / std

    # Randomized range finder, working on the (n_rows x n_cols) matrix
    # without ever holding more than one block of it.
    omega = rng.standard_normal((n_cols, width)).astype(np.float32)
    sketch = _left_multiply(standardized, omega, n_rows)

    for _ in range(n_iter):
        projected = _right_multiply(standardized, sketch, n_cols)
        projected, _ = np.linalg.qr(projected)
        sketch = _left_multiply(standardized, projected, n_rows)

    basis, _ = np.linalg.qr(sketch)  # (n_rows, width)
    small = _right_multiply(standardized, basis, n_cols).T  # (width, n_cols)

    _, singular, vt = np.linalg.svd(small, full_matrices=False)
    k_effective = min(k, vt.shape[0])
    components = vt[:k_effective].T.astype(np.float32)

    total = float((singular**2).sum())
    ratio = (singular[:k_effective] ** 2 / total) if total > 0 else np.zeros(k_effective)
    return components, np.asarray(ratio, dtype=np.float32), n_rows


def _left_multiply(standardized: BlockIterFactory, right: np.ndarray, n_rows: int) -> np.ndarray:
    """X @ right, accumulated block by block. Returns (n_rows, width)."""
    out = np.empty((n_rows, right.shape[1]), dtype=np.float32)
    start = 0
    for block in standardized():
        stop = start + block.shape[0]
        out[start:stop] = block @ right
        start = stop
    if start != n_rows:
        raise ValueError(f"the block iterator yielded {start} rows, expected {n_rows}")
    return out


def _right_multiply(standardized: BlockIterFactory, left: np.ndarray, n_cols: int) -> np.ndarray:
    """X.T @ left, accumulated block by block. Returns (n_cols, width)."""
    out = np.zeros((n_cols, left.shape[1]), dtype=np.float32)
    start = 0
    for block in standardized():
        stop = start + block.shape[0]
        out += block.T @ left[start:stop]
        start = stop
    return out


def resident_pca(
    matrix: np.ndarray, k: int, *, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """PCA over the columns of a matrix that fits in memory.

    Used for the LINCS layers (signatures x genes) and the perturbation
    phenotypes (targets x genes), which are orders of magnitude smaller than
    the single-cell sources.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-d matrix, got shape {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError(f"need at least 2 rows, got {matrix.shape[0]}")

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    standardized = (matrix - mean) / np.where(std < MIN_STD, 1.0, std)

    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    k_effective = min(k, vt.shape[0])
    components = vt[:k_effective].T.astype(np.float32)

    total = float((singular**2).sum())
    ratio = (singular[:k_effective] ** 2 / total) if total > 0 else np.zeros(k_effective)
    return components, np.asarray(ratio, dtype=np.float32)


def sparse_svd(matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Truncated SVD of a sparse indicator matrix, without centering.

    Centering a multi-hot membership matrix would make it dense, and the
    structure wanted here — which genes share groups — survives without it.
    Returns `(rows, k_effective)` features for the matrix's rows.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds

    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    k_effective = min(k, min(matrix.shape) - 1)
    if k_effective < 1:
        raise ValueError(f"matrix too small for an SVD: shape {matrix.shape}")

    u, singular, _ = svds(matrix, k=k_effective)
    order = np.argsort(singular)[::-1]  # svds returns ascending
    u, singular = u[:, order], singular[order]

    total = float((singular**2).sum())
    ratio = (singular**2 / total) if total > 0 else np.zeros(k_effective)
    return (u * singular).astype(np.float32), np.asarray(ratio, dtype=np.float32)
