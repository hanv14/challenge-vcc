#!/usr/bin/env python
"""What differs between two runs' submitted predictions? (DECISIONS.md D116)

    python scripts/compare_predictions.py runs/server runs/server_cal_s05

Reads each run's `predictions/model/<context>/fold_changes.npz` — the per-gene log2
fold changes the cells were built from, after the generator's threshold and
scale — and splits every context's (targets x genes) matrix into a **shared**
part (the mean over targets: what every target is predicted to do) and a
**specific** part (each target's deviation from it). Prints, per run and
context, how large each part is, how many genes move, and how alike the
targets are, then how the two runs' shared parts and per-target predictions
relate.

The question it answers: the leaderboard's 0 is a mean-response baseline, so
a submission whose predictions are mostly one shared profile is close to that
baseline by construction. Needs no GPU and trains nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load(run: Path) -> dict[str, tuple[list[str], np.ndarray]]:
    out = {}
    for path in sorted((run / "predictions").glob("**/fold_changes.npz")):
        data = np.load(path, allow_pickle=True)
        out[path.parent.name] = (
            [str(t) for t in data["targets"]],
            np.asarray(data["log2_fold_change"], dtype=np.float64),
        )
    return out


def median_pairwise_correlation(matrix: np.ndarray, limit: int = 200) -> float:
    rows = matrix[:limit]
    centered = rows - rows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    keep = norms > 0
    if keep.sum() < 2:
        return float("nan")
    unit = centered[keep] / norms[keep, None]
    corr = unit @ unit.T
    return float(np.median(corr[np.triu_indices(len(unit), 1)]))


def describe(matrix: np.ndarray) -> dict[str, float]:
    shared = matrix.mean(axis=0)
    specific = matrix - shared
    total = float((matrix**2).sum())
    return {
        "rms_all": float(np.sqrt((matrix**2).mean())),
        "rms_shared": float(np.sqrt((shared**2).mean())),
        "rms_specific": float(np.sqrt((specific**2).mean())),
        "shared_share": float((shared**2).sum() * matrix.shape[0] / total) if total else 0.0,
        "shared_genes>0.1": int((np.abs(shared) > 0.1).sum()),
        "moved/target": float((np.abs(matrix) > 0).sum(axis=1).mean()),
        "target_corr": median_pairwise_correlation(matrix),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    args = parser.parse_args(argv)

    runs = [(str(args.run_a), load(args.run_a)), (str(args.run_b), load(args.run_b))]
    for name, contexts in runs:
        if not contexts:
            print(f"{name}: no predictions/**/fold_changes.npz", file=sys.stderr)
            return 1

    keys = list(describe(np.zeros((2, 2))))
    print(f"{'run':>22} {'ctx':>4} " + " ".join(f"{k:>16}" for k in keys))
    for name, contexts in runs:
        for ctx, (_, matrix) in contexts.items():
            d = describe(matrix)
            print(f"{name:>22} {ctx:>4} " + " ".join(f"{d[k]:>16.4g}" for k in keys))

    (a_name, a), (b_name, b) = runs
    print(f"\n{a_name} vs {b_name}, per context:")
    for ctx in sorted(set(a) & set(b)):
        targets_a, ma = a[ctx]
        targets_b, mb = b[ctx]
        common = [t for t in targets_a if t in set(targets_b)]
        ia = [targets_a.index(t) for t in common]
        ib = [targets_b.index(t) for t in common]
        ma, mb = ma[ia], mb[ib]
        shared_corr = corr(ma.mean(axis=0), mb.mean(axis=0))
        per_target = [corr(x, y) for x, y in zip(ma, mb)]
        specific = [
            corr(x - ma.mean(axis=0), y - mb.mean(axis=0)) for x, y in zip(ma, mb)
        ]
        print(
            f"  {ctx}: {len(common)} targets | shared parts correlate {shared_corr:+.3f} | "
            f"per-target median {np.nanmedian(per_target):+.3f} | "
            f"target-specific parts median {np.nanmedian(specific):+.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
