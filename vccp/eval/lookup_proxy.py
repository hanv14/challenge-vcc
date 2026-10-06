"""Leaderboard-like scores inside one Replogle screen, for the lookup's denoisers.

`scripts/measure/lookup_proxy.py` splits each target's cells in two, denoises
the half-A residual and scores it against half B. These are its scores, per
target, each a stand-in for one leaderboard member (docs/metrics.md):

* `pds_like` — the rank of a target's own half-B residual among every
  panel target's, by cosine distance to its denoised half-A residual, with
  mid-rank ties and `1 − rank / (P − 1)`, as `pds_cosine` ranks (§ pds).
  Scale-free, like the member.
* `nmae_like` — mean |predicted − half-B log2 change| over the genes half B
  calls significant, divided by mean |half-B change| there: 1.0 for no
  change, as `de_wilcoxon_lfc_nmae`.
* `reach_like` — the genes half B calls significant, ranked by our
  confidence; the deepest depth whose signs are at least 90% right, over
  the number of those genes, as `de_wilcoxon_direction_reach_raw`.
* `pearson_like` — a diagnostic: the correlation of the denoised half-A
  residual with the half-B residual over the genes both could move.

Significance in half B is a z-test of its change against its own controls
(the delta-method sd), BH-corrected per target at `alpha`; a target with
fewer than `min_gate` significant genes scores NaN on the two DE-like
members, as the scorer omits it.

They are proxies. They know nothing of the challenge's large shared shift or
of our count generator; they score how well a denoiser recovers a screen's
own target-specific change, entry by entry. The leaderboard decides whether
that recovery transfers.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

REACH_PURITY = 0.9
MIN_GATE = 10


def two_sided_p(z: np.ndarray) -> np.ndarray:
    return erfc(np.abs(z) / np.sqrt(2.0))


def bh_significant(p: np.ndarray, valid: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg per row, over the `valid` entries only."""
    p = np.where(valid, p, np.inf)
    out = np.zeros(p.shape, dtype=bool)
    m = valid.sum(axis=1)
    order = np.argsort(p, axis=1, kind="stable")
    ranked = np.take_along_axis(p, order, axis=1)
    k = np.arange(1, p.shape[1] + 1)[None, :]
    passes = ranked <= alpha * k / np.maximum(m[:, None], 1)
    # The largest rank that passes; everything ranked before it is significant.
    last = np.where(passes.any(axis=1), p.shape[1] - 1 - np.argmax(passes[:, ::-1], axis=1), -1)
    keep = k <= (last[:, None] + 1)
    np.put_along_axis(out, order, keep, axis=1)
    return out & valid


def pds_like(pred: np.ndarray, truth: np.ndarray, own: np.ndarray,
             exclude: np.ndarray | None = None, own_gene: np.ndarray | None = None) -> np.ndarray:
    """Per query target, `1 − rank / (P − 1)` of its own truth among all.

    `pred` (queries × genes), `truth` (panel × genes), `own[i]` the panel row
    of query i. `exclude` (genes,) drops features for every comparison (the
    scorer's panel-wide target-gene exclusion); `own_gene[i]` (−1 = none)
    drops query i's own gene from its comparisons only.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if exclude is not None:
        pred = pred[:, ~exclude]
        truth = truth[:, ~exclude]
        if own_gene is not None:
            remap = np.cumsum(~exclude) - 1
            own_gene = np.where(own_gene >= 0, np.where(exclude[np.maximum(own_gene, 0)], -1,
                                                        remap[np.maximum(own_gene, 0)]), -1)
    dots = pred @ truth.T
    pred_sq = np.sum(pred * pred, axis=1)
    truth_sq = np.broadcast_to(np.sum(truth * truth, axis=1)[None, :], dots.shape).copy()
    pred_sq = np.broadcast_to(pred_sq[:, None], dots.shape).copy()
    if own_gene is not None:
        rows = np.flatnonzero(own_gene >= 0)
        g = own_gene[rows]
        p_g = pred[rows, g]
        t_g = truth[:, g].T                  # (rows, panel)
        dots[rows] -= p_g[:, None] * t_g
        truth_sq[rows] -= t_g * t_g
        pred_sq[rows] -= (p_g * p_g)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = dots / np.sqrt(np.maximum(pred_sq, 0) * np.maximum(truth_sq, 0))
    distance = np.where(np.isfinite(cosine), 1.0 - cosine, 1.0)
    true_distance = distance[np.arange(len(own)), own][:, None]
    closer = np.sum(distance < true_distance, axis=1)
    tied = np.sum(distance == true_distance, axis=1)
    rank = closer + (tied - 1) / 2.0
    panel = truth.shape[0]
    return 1.0 - rank / max(panel - 1, 1)


def nmae_like(pred: np.ndarray, truth: np.ndarray, significant: np.ndarray,
              min_gate: int = MIN_GATE) -> np.ndarray:
    """Per target, mean |pred − truth| / mean |truth| over its significant genes."""
    n = significant.sum(axis=1)
    err = np.where(significant, np.abs(pred - truth), 0.0).sum(axis=1)
    size = np.where(significant, np.abs(truth), 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = err / size
    return np.where((n >= min_gate) & (size > 0), out, np.nan)


def reach_like(pred: np.ndarray, confidence: np.ndarray, truth: np.ndarray,
               significant: np.ndarray, purity: float = REACH_PURITY,
               min_gate: int = MIN_GATE) -> np.ndarray:
    """Per target, the deepest prefix of its significant genes, ranked by our
    confidence, whose signs are at least `purity` right, over their number."""
    out = np.full(pred.shape[0], np.nan)
    for i in range(pred.shape[0]):
        genes = np.flatnonzero(significant[i] & (truth[i] != 0))
        if genes.size < min_gate:
            continue
        order = np.lexsort((genes, -np.abs(pred[i, genes]), -confidence[i, genes]))
        genes = genes[order]
        match = np.sign(pred[i, genes]) == np.sign(truth[i, genes])
        purity_at = np.cumsum(match) / np.arange(1, genes.size + 1)
        good = np.flatnonzero(purity_at >= purity)
        out[i] = (good[-1] + 1) / genes.size if good.size else 0.0
    return out


def pearson_like(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per target, Pearson correlation over its `valid` genes (NaN if flat)."""
    out = np.full(pred.shape[0], np.nan)
    for i in range(pred.shape[0]):
        a, b = pred[i, valid[i]], truth[i, valid[i]]
        if a.size < 3:
            continue
        a, b = a - a.mean(), b - b.mean()
        denom = np.sqrt((a @ a) * (b @ b))
        if denom > 0:
            out[i] = float(a @ b) / denom
    return out


def paired_bootstrap(diff: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    """Mean and bootstrap sd of a per-target difference, NaNs dropped."""
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan"), float("nan")
    if n <= 0 or diff.size < 2:
        return float(diff.mean()), float("nan")
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n, diff.size))].mean(axis=1)
    return float(diff.mean()), float(means.std(ddof=1))
