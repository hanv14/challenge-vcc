"""Per-group CP10K moments of a Replogle screen, in one streaming pass.

The lookup's per-entry denoisers (D128) and the offline instrument that
chooses them (`scripts/measure/lookup_proxy.py`) need, for many groups of
cells (a target's cells, a quarter of them, the controls), the per-gene mean
of per-cell CP10K and the mean of its square. This computes both for any
number of groups in a single pass over the screen's single-cell file, read
in contiguous row spans, so a pass over every target of K562 genome-wide
(about 2 million cells, 65 GB on the server) streams the file once and never
holds more than one block of cells.

A cell's CP10K is its counts over the screen's measured genes divided by
their sum, times 1e4: exactly what `sanity.checks.mean_cp10k`, and so the
lookup, computes. A group's mean CP10K here is the lookup's
`mean_cp10k(counts)` for the same cells, up to floating-point summation order.

Rows are read with h5py directly (CSR groups and dense datasets both), from
the file `meta.json` points at, through the screen's `source_col`. The file
is opened once per pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..logging_utils import get_logger

TARGET_SUM = 1e4
#: Bounds on the rows read per block; within them the block size is derived
#: from `resources.max_ram_gb` (`block_rows_for`).
MAX_BLOCK_ROWS = 50_000
MIN_BLOCK_ROWS = 256
#: A progress line every this many blocks.
LOG_EVERY_BLOCKS = 20
#: Requested rows closer than this share one contiguous read ...
MAX_GAP = 64
#: ... of at most this many file rows, so one read stays a few hundred MB even
#: when the requested cells are scattered through a 2-million-row file.
MAX_RUN_ROWS = 4096


@dataclass
class GroupMoments:
    """`n`, mean CP10K and mean squared CP10K per group, over the screen's genes."""

    n: np.ndarray        # (groups,) int64
    mean: np.ndarray     # (groups, screen genes) float64
    meansq: np.ndarray   # (groups, screen genes) float64

    def merged(self, groups) -> tuple[int, np.ndarray, np.ndarray]:
        """The moments of the union of `groups` (indices), weighted by size."""
        groups = np.atleast_1d(np.asarray(groups))
        n = self.n[groups].astype(np.float64)
        total = float(n.sum())
        if total <= 0:
            zeros = np.zeros(self.mean.shape[1])
            return 0, zeros, zeros
        mean = (n[:, None] * self.mean[groups]).sum(axis=0) / total
        meansq = (n[:, None] * self.meansq[groups]).sum(axis=0) / total
        return int(total), mean, meansq


def block_rows_for(n_genes: int, max_ram_gb: float) -> int:
    """Rows per block, so a dense float64 block and four copies fit a sixteenth
    of `resources.max_ram_gb`."""
    budget = max_ram_gb * 1e9 / 16
    rows = int(budget / max(1, n_genes * 8 * 5))
    return int(min(MAX_BLOCK_ROWS, max(MIN_BLOCK_ROWS, rows)))


class _RowReader:
    """Contiguous row spans of the screen's single-cell matrix, as counts over
    the screen's measured genes."""

    def __init__(self, context):
        import h5py

        self.columns = context.genes["source_col"].to_numpy()
        self._file = h5py.File(context.source_path, "r", rdcc_nbytes=256 * 1024**2)
        X = self._file["X"]
        self._dense = isinstance(X, h5py.Dataset)
        self._X = X
        if not self._dense:
            self._data = X["data"]
            self._indices = X["indices"]
            self._indptr = X["indptr"][:].astype(np.int64)
            self._n_cols = int(X.attrs["shape"][1])

    def close(self) -> None:
        self._file.close()

    def span(self, rows: np.ndarray) -> np.ndarray:
        """Counts of `rows` (ascending file rows), float64.

        Rows are read in runs: neighbours closer than `MAX_GAP` rows share one
        contiguous read, so a dense selection (every target of a screen) is a
        few large sequential reads and a sparse one (the round's targets) reads
        only what it needs.
        """
        runs = split_runs(rows)
        if self._dense:
            parts = []
            for run in runs:
                lo, hi = int(run[0]), int(run[-1]) + 1
                parts.append(np.asarray(self._X[lo:hi])[run - lo][:, self.columns])
            return np.rint(np.concatenate(parts)).astype(np.float64)
        import scipy.sparse as sp

        parts = []
        for run in runs:
            lo, hi = int(run[0]), int(run[-1]) + 1
            start, stop = self._indptr[lo], self._indptr[hi]
            block = sp.csr_matrix(
                (self._data[start:stop], self._indices[start:stop],
                 self._indptr[lo:hi + 1] - start),
                shape=(hi - lo, self._n_cols),
            )
            parts.append(block[run - lo][:, self.columns])
        return np.rint(sp.vstack(parts).toarray()).astype(np.float64)


def split_runs(rows: np.ndarray) -> list[np.ndarray]:
    """Ascending rows cut into runs: a new run where the gap to the previous
    row exceeds `MAX_GAP`, or where the run would span more than
    `MAX_RUN_ROWS` file rows."""
    runs, start = [], 0
    for end in np.flatnonzero(np.diff(rows) > MAX_GAP) + 1:
        runs.extend(_cap_extent(rows[start:end]))
        start = end
    runs.extend(_cap_extent(rows[start:]))
    return [r for r in runs if r.size]


def _cap_extent(run: np.ndarray) -> list[np.ndarray]:
    if run.size == 0 or run[-1] - run[0] < MAX_RUN_ROWS:
        return [run]
    cuts = np.flatnonzero(np.diff((run - run[0]) // MAX_RUN_ROWS)) + 1
    return np.split(run, cuts)


def stream_moments(context, groups: list[np.ndarray], max_ram_gb: float,
                   label: str = "") -> GroupMoments:
    """CP10K moments of every group of rows (a row may sit in one group only).

    `groups[i]` holds file rows (`cells.parquet`'s `row`). One pass over the
    file in ascending row order, `block_rows_for` requested rows at a time;
    rows nobody asked for are read only when they sit between close
    neighbours (`MAX_GAP`).
    """
    log = get_logger()
    n_groups = len(groups)
    n_genes = int(len(context.genes))
    rows = np.concatenate([np.asarray(g, dtype=np.int64) for g in groups]) if groups else \
        np.zeros(0, np.int64)
    owner = np.concatenate([np.full(len(g), i, dtype=np.int64) for i, g in enumerate(groups)]) \
        if groups else np.zeros(0, np.int64)
    if np.unique(rows).size != rows.size:
        raise ValueError(f"{label or context.name}: a cell was assigned to two groups")
    order = np.argsort(rows, kind="stable")
    rows, owner = rows[order], owner[order]

    need = 2 * n_groups * n_genes * 8 / 1e9
    if need > max_ram_gb / 2:
        raise MemoryError(
            f"{label or context.name}: {n_groups} groups x {n_genes} genes need {need:.1f} GB "
            f"of accumulators, over half of resources.max_ram_gb ({max_ram_gb:g}); "
            "raise resources.max_ram_gb or read fewer targets"
        )
    sums = np.zeros((n_groups, n_genes), dtype=np.float64)
    sumsq = np.zeros((n_groups, n_genes), dtype=np.float64)
    counts = np.bincount(owner, minlength=n_groups).astype(np.int64)

    step = block_rows_for(n_genes, max_ram_gb)
    reader = _RowReader(context)
    started, done, blocks = time.time(), 0, 0
    try:
        position = 0
        while position < rows.size:
            end = min(position + step, rows.size)
            span_rows = rows[position:end]
            span_owner = owner[position:end]
            block = reader.span(span_rows)
            library = np.maximum(block.sum(axis=1), 1.0)
            cp10k = block * (TARGET_SUM / library)[:, None]
            # Rows are sorted by file position; group them for reduceat.
            by_group = np.argsort(span_owner, kind="stable")
            grouped = span_owner[by_group]
            starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
            present = grouped[starts]
            values = cp10k[by_group]
            sums[present] += np.add.reduceat(values, starts, axis=0)
            sumsq[present] += np.add.reduceat(values * values, starts, axis=0)
            done += span_rows.size
            position = end
            blocks += 1
            del block, cp10k, values
            elapsed = time.time() - started
            if position == rows.size or blocks % LOG_EVERY_BLOCKS == 0:
                rate = done / max(elapsed, 1e-9)
                log.info("  %s: %d of %d cells read (%.0f cells/s, %.0f s left)",
                         label or context.name, done, rows.size, rate,
                         (rows.size - done) / max(rate, 1e-9))
    finally:
        reader.close()

    # In place: for every target of a genome-wide screen these are gigabytes.
    denom = np.maximum(counts, 1)[:, None].astype(np.float64)
    sums /= denom
    sumsq /= denom
    return GroupMoments(n=counts, mean=sums, meansq=sumsq)


# --------------------------------------------------------------------------- #
# changes and their sampling error, from moments
# --------------------------------------------------------------------------- #
def log2_change(mean_t: np.ndarray, mean_c: np.ndarray, floor: float) -> np.ndarray:
    """log2 of the ratio of mean CP10K, both floored: what the lookup and the
    count generator use."""
    return np.log2(np.maximum(mean_t, floor) / np.maximum(mean_c, floor))


def per_cell_variance(n, mean: np.ndarray, meansq: np.ndarray) -> np.ndarray:
    """Unbiased per-cell variance of CP10K from its first two moments."""
    n = np.asarray(n, dtype=np.float64)
    if n.ndim:
        n = n[:, None]
    raw = np.maximum(meansq - mean * mean, 0.0)
    return raw * n / np.maximum(n - 1.0, 1.0)


def residual_z(n_t, mean_t: np.ndarray, meansq_t: np.ndarray, n_c, mean_c: np.ndarray,
               meansq_c: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """z of each target's deviation from the screen's mean response, per gene.

    The residual a lookup carries is a target's change minus the screen's mean
    change `shift` (log2). Under "no target-specific effect" the target's mean
    CP10K is the control's times 2^shift; this tests the observed mean against
    that, on the linear scale where the test is well behaved:

        z = (m_t − m_c 2^s) / sqrt(max(v_t, v_c 2^s) / n_t + v_c 4^s / n_c)

    The target's variance is never taken below the null's (the controls',
    scaled to the null mean as Poisson-like counts scale). That matters where a
    target's cells happen to show a gene not at all: its own variance is then
    0, its log change sits at the floor (−3 to −6), and a log-scale sd would
    call that a strong effect. Here a low-expressed gene absent from 80 cells
    reads z ≈ −1, as it should.
    """
    n_arr = np.asarray(n_t, dtype=np.float64)
    n_col = n_arr[:, None] if n_arr.ndim else n_arr
    factor = np.exp2(np.asarray(shift, dtype=np.float64))
    var_t = per_cell_variance(n_arr, mean_t, meansq_t)
    var_c = per_cell_variance(n_c, mean_c, meansq_c)
    null_mean = mean_c * factor
    null_var = var_c * factor
    denom = np.sqrt(np.maximum(var_t, null_var) / np.maximum(n_col, 1.0)
                    + var_c * factor * factor / max(float(n_c), 1.0))
    diff = mean_t - null_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(denom > 0, diff / np.where(denom > 0, denom, 1.0),
                     np.where(diff == 0, 0.0, np.sign(diff) * np.inf))
    return z


def null_log2_sd(n_t, n_c, mean_c: np.ndarray, meansq_c: np.ndarray,
                 floor: float) -> np.ndarray:
    """The sampling sd of a log2 change with no effect, at `n_t` target cells:
    sqrt(v_c (1/n_t + 1/n_c)) / (m_c ln 2). A gene's noise scale at that
    depth, defined even where a target shows the gene not at all; the
    low-rank denoiser whitens by it."""
    n_t = np.asarray(n_t, dtype=np.float64)
    n_t = n_t[:, None] if n_t.ndim else n_t
    var_c = per_cell_variance(n_c, mean_c, meansq_c)
    return np.sqrt(var_c * (1.0 / np.maximum(n_t, 1.0) + 1.0 / max(float(n_c), 1.0))) / (
        np.maximum(mean_c, floor) * np.log(2.0)
    )


def meansq_cp10k(counts: np.ndarray) -> np.ndarray:
    """Per-gene mean of squared per-cell CP10K: `mean_cp10k`'s second moment,
    with the same library size (the counts' row sum, at least 1)."""
    counts = np.asarray(counts)
    library = np.maximum(counts.sum(axis=1), 1).astype(np.float64)
    cp10k = counts.astype(np.float64) * (TARGET_SUM / library)[:, None]
    return np.mean(cp10k * cp10k, axis=0)
