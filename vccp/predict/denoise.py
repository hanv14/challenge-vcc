"""Denoisers for the lookup's residuals (D127, D128).

A covered target's lookup residual is its change in a Replogle screen minus
that screen's mean change. On the server it is about 85% sampling noise
(median split-half reliability 0.152, D126), and the raw residual still won
+0.032 on the leaderboard. A denoiser that improves the residual's
*direction*, entry by entry, is the largest lever left. The candidates:

* ``gene`` — the per-gene James–Stein factor (D124). Pools each gene across
  targets, so it zeroes the sparse effects that tell targets apart; refuted
  on the leaderboard (D127). Kept as a reference arm.
* ``snr`` — per entry, z tests the target's deviation from the screen's mean
  response (`screen_stats.residual_z`), and the residual is kept in
  proportion z² / (z² + tau): an entry well above its noise survives whole,
  one at its noise level is halved at tau = 1, one below it is nearly gone.
  tau is in z² units, so a value chosen at one depth carries to another: a
  real effect's z grows with the square root of the cells.
* ``snr_hard`` — keeps an entry whole when z² > tau, else zero.
* ``lowrank`` — the projection onto the top `rank` components of every
  target's residual in the screen (all ~9,800 in K562 genome-wide), after
  dividing each gene by its sampling sd with no effect
  (`screen_stats.null_log2_sd`), so that noisy genes do not choose the
  components.
* ``lowrank_snr`` — the projection plus the SNR-shrunk remainder, so sparse
  effects outside the components can survive.

Everything here is a pure function of arrays; the lookup applies them when a
change is asked for, and `scripts/measure/lookup_proxy.py` scores them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NONE, GENE, SNR, SNR_HARD, LOWRANK, LOWRANK_SNR = (
    "none", "gene", "snr", "snr_hard", "lowrank", "lowrank_snr"
)
ALL = (NONE, GENE, SNR, SNR_HARD, LOWRANK, LOWRANK_SNR)
#: Denoisers that need each entry's z.
NEEDS_Z = frozenset({SNR, SNR_HARD, LOWRANK_SNR})
#: Denoisers that need a screen's low-rank basis.
NEEDS_BASIS = frozenset({LOWRANK, LOWRANK_SNR})
#: Below this many targets a basis is not worth fitting.
MIN_BASIS_TARGETS = 3


# --------------------------------------------------------------------------- #
# per gene (the reference arm)
# --------------------------------------------------------------------------- #
def gene_factor(full: np.ndarray, mean: np.ndarray, half_a: np.ndarray,
                half_b: np.ndarray) -> np.ndarray:
    """James–Stein factor per gene, max(0, 1 − noise / power), in [0, 1].

    `full`, `half_a`, `half_b`: (targets, genes) changes at full depth and in
    two halves of each target's cells; `mean` is `full`'s mean over targets.
    The noise of a full-depth change is var(A − B) / 4.
    """
    noise = np.mean((half_a - half_b) ** 2, axis=0) / 4
    power = np.mean((full - mean) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(power > 0, 1.0 - noise / power, 0.0)
    return np.clip(factor, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# per entry
# --------------------------------------------------------------------------- #
def _z_squared(z: np.ndarray) -> np.ndarray:
    """z², with an undefined z (0/0) read as no evidence."""
    return np.nan_to_num(np.asarray(z, dtype=np.float64) ** 2, nan=0.0, posinf=np.inf)


def snr_soft(residual: np.ndarray, z: np.ndarray, tau: float) -> np.ndarray:
    """`residual × z² / (z² + tau)`; tau = 0 is the raw residual."""
    if tau <= 0:
        return np.asarray(residual, dtype=np.float32).copy()
    z2 = _z_squared(z)
    with np.errstate(invalid="ignore"):
        keep = np.where(np.isinf(z2), 1.0, z2 / (z2 + tau))
    return (np.asarray(residual, dtype=np.float64) * keep).astype(np.float32)


def snr_hard(residual: np.ndarray, z: np.ndarray, tau: float) -> np.ndarray:
    """The residual where z² > tau, else 0."""
    z2 = _z_squared(z)
    return np.where(z2 > tau, residual, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# low rank
# --------------------------------------------------------------------------- #
@dataclass
class LowRankBasis:
    """Orthonormal components of noise-whitened residuals, on one gene axis."""

    components: np.ndarray       # (k_max, genes) float32; zero outside used genes
    scale: np.ndarray            # (genes,) float32 noise sd per gene; 0 = unused
    singular_values: np.ndarray  # (k_max,) float32
    n_targets: int

    @property
    def k_max(self) -> int:
        return int(self.components.shape[0])

    def project(self, residual: np.ndarray, rank: int) -> np.ndarray:
        """The residual's part in the top `rank` components (rows or one row)."""
        if rank < 1 or rank > self.k_max:
            raise ValueError(f"rank {rank} is outside 1..{self.k_max} of this basis")
        used = self.scale > 0
        residual = np.asarray(residual, dtype=np.float64)
        whitened = np.where(used, residual / np.where(used, self.scale, 1.0), 0.0)
        V = self.components[:rank].astype(np.float64)
        projected = (whitened @ V.T) @ V
        return (projected * self.scale).astype(np.float32)


def fit_basis(residual: np.ndarray, noise: np.ndarray, k_max: int, seed: int) -> LowRankBasis:
    """Top `k_max` components of `residual` (targets × genes), each gene first
    divided by its median noise sd over targets (`noise`, same shape: the
    sampling sd of a change with no effect).

    Genes that never move or have no sd are left out (scale 0). Components are
    sign-fixed (largest entry positive) and the decomposition is seeded, so the
    same residuals give the same basis.
    """
    residual = np.asarray(residual, dtype=np.float64)
    noise = np.broadcast_to(np.asarray(noise, dtype=np.float64), residual.shape)
    n_targets, n_genes = residual.shape
    moves = np.any(residual != 0, axis=0)
    scale = np.zeros(n_genes, dtype=np.float64)
    if moves.any():
        with np.errstate(all="ignore"):
            scale[moves] = np.nan_to_num(np.median(noise[:, moves], axis=0), nan=0.0)
        scale[~np.isfinite(scale)] = 0.0
    used = scale > 0
    k = int(min(k_max, n_targets - 1, int(used.sum())))
    if n_targets < MIN_BASIS_TARGETS:
        k = 0
    k = max(k, 0)
    components = np.zeros((k, n_genes), dtype=np.float32)
    values = np.zeros(k, dtype=np.float32)
    if k >= 1:
        W = residual[:, used] / scale[used]
        W = W - W.mean(axis=0, keepdims=True)
        if min(W.shape) <= 2000:
            _, s, vt = np.linalg.svd(W, full_matrices=False)
            s, vt = s[:k], vt[:k]
        else:
            from sklearn.utils.extmath import randomized_svd

            _, s, vt = randomized_svd(W, n_components=k, n_iter=7, random_state=seed)
        largest = np.argmax(np.abs(vt), axis=1)
        signs = np.sign(vt[np.arange(k), largest])
        signs[signs == 0] = 1.0
        vt = vt * signs[:, None]
        components[:, used] = vt.astype(np.float32)
        values[:] = s.astype(np.float32)
    return LowRankBasis(components=components, scale=scale.astype(np.float32),
                        singular_values=values, n_targets=int(n_targets))


def lowrank(residual: np.ndarray, basis: LowRankBasis, rank: int) -> np.ndarray:
    return basis.project(residual, rank)


def lowrank_snr(residual: np.ndarray, z: np.ndarray, basis: LowRankBasis, rank: int,
                tau: float) -> np.ndarray:
    """The low-rank part plus the SNR-shrunk remainder.

    The remainder's z is the residual's, rescaled by the same per-entry sd
    (|residual| / |z|), so it measures the remainder against the same noise.
    """
    residual = np.asarray(residual, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    part = basis.project(residual, rank).astype(np.float64)
    rest = residual - part
    abs_z = np.abs(z)
    finite = np.isfinite(abs_z) & (abs_z > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # sd 0 where z is infinite (keep whatever remains), inf where z is 0.
        sd = np.where(finite, np.abs(residual) / np.where(finite, abs_z, 1.0),
                      np.where(np.isinf(abs_z), 0.0, np.inf))
        usable = np.isfinite(sd) & (sd > 0)
        z_rest = np.where(usable, rest / np.where(usable, sd, 1.0),
                          np.where(sd == 0, np.sign(rest) * np.inf, 0.0))
    return (part + snr_soft(rest, z_rest, tau)).astype(np.float32)


# --------------------------------------------------------------------------- #
# one entry point
# --------------------------------------------------------------------------- #
def apply(kind: str, residual: np.ndarray, *, z: np.ndarray | None = None,
          basis: LowRankBasis | None = None, gene: np.ndarray | None = None,
          tau: float = 1.0, rank: int = 1) -> np.ndarray:
    """Denoise `residual` (one row or rows) by `kind`."""
    if kind == NONE:
        return np.asarray(residual, dtype=np.float32)
    if kind == GENE:
        if gene is None:
            raise ValueError("the 'gene' denoiser needs the per-gene factors")
        return (np.asarray(residual, dtype=np.float32) * gene).astype(np.float32)
    if kind in NEEDS_Z and z is None:
        raise ValueError(f"the {kind!r} denoiser needs each entry's z")
    if kind in NEEDS_BASIS and basis is None:
        raise ValueError(f"the {kind!r} denoiser needs the screen's low-rank basis")
    if kind == SNR:
        return snr_soft(residual, z, tau)
    if kind == SNR_HARD:
        return snr_hard(residual, z, tau)
    if kind == LOWRANK:
        return lowrank(residual, basis, rank)
    if kind == LOWRANK_SNR:
        return lowrank_snr(residual, z, basis, rank, tau)
    raise ValueError(f"unknown denoiser {kind!r}; known: {ALL}")
