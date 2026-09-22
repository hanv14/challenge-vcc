"""The leaderboard's 0–1 scale, built with `cell-eval2`'s own tools (§6).

The leaderboard does not report the six raw metrics. It rescales each so that
**0 is the organizers' mean-response baseline** and **1 is a split-half
replicate of the real data**, then averages them. Those two ends are
properties of a *dataset*, not of a submission, and `cell-eval2` builds both
from the real data alone:

* the 0 end — the generic-response prediction, scored as an ordinary
  submission (`cell-eval2 baseline`);
* the 1 end — the replicate anchor, the mean over disjoint split-halves of
  the real cells (`run --anchor`).

Here they are built together as a **real bundle**, which is the packaging
`cell-eval2` gives the pair, and a scored run is then placed on that scale
with `score_metrics(..., real_bundle=...)`.

Two rules from §6, both enforced by this module doing nothing else:

* these are **reference points of the scorer**. Never predictions, never
  submitted, never used to train or to choose the model — the calibration
  maximizes the no-change objective (`eval/nochange.py`), not this;
* where the pair **cannot** be built, the six raw values are reported alone
  and the reason is recorded. On small data the baseline leg is often
  degenerate (a metric whose 0 and 1 ends coincide), and `cell-eval2` refuses
  such a bundle rather than publishing a ranking that rests on it — that
  refusal is the answer, so it is caught and reported, not worked around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger

#: What the scaled frame calls the two columns and the average.
FROM_BASELINE = "from_baseline"
FROM_REPLICATE = "from_replicate"
AVERAGE = "avg_score"


@dataclass
class ScaleReference:
    """The two ends of one dataset's scale, or why they could not be built."""

    built: bool
    reason: str | None = None
    bundle_dir: str | None = None
    bundle_id: str | None = None
    n_splits: int = 0
    base_seed: int = 0
    manifest: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "built": self.built,
            "reason": self.reason,
            "definition": "0 = the organizers' mean-response baseline, 1 = a "
            "split-half replicate of the real data (§6)",
            "bundle": self.bundle_dir,
            "bundle_id": self.bundle_id,
            "n_splits": self.n_splits,
            "base_seed": self.base_seed,
            "cell_eval2_version": self.manifest.get("cell_eval2_version"),
            "anchor_digest": self.manifest.get("anchor_digest"),
        }


def competition_rule() -> tuple[int, int]:
    """`(n_splits, base_seed)` of the competition's own anchor rule."""
    from cell_eval2 import competition

    return int(competition.N_SPLITS), int(competition.BASE_SEED)


def build(
    cfg,
    real,
    *,
    pert_col: str,
    control_label: str,
    outdir: Path,
    bundle_id: str,
) -> ScaleReference:
    """Build both ends on this dataset, or say why they could not be built."""
    from . import official

    log = get_logger()
    if not official.available():
        return ScaleReference(False, "cell-eval2 is not installed")

    try:
        from cell_eval2.baseline import build_baseline_prediction, generic_response_profile
        from cell_eval2.real_bundle import build_real_bundle

        n_splits, base_seed = competition_rule()
        config = official.build_config(cfg, pert_col, control_label)

        # The 0 end: every perturbation predicted as the mean response.
        profile = generic_response_profile(real, pert_col=pert_col, control=control_label)
        baseline_prediction = build_baseline_prediction(
            profile, real, pert_col=pert_col, control=control_label,
            emit="dispersed", seed=base_seed,
        )
        outdir.mkdir(parents=True, exist_ok=True)
        manifest = build_real_bundle(
            real,
            baseline_prediction,
            config=config,
            outdir=str(outdir),
            bundle_id=bundle_id,
            base_seed=base_seed,
            n_splits=n_splits,
            force=True,
        )
    except Exception as exc:  # noqa: BLE001 — a refusal is the answer, not a crash
        reason = f"{type(exc).__name__}: {exc}"
        log.warning("  the leaderboard scale could not be built: %s", reason)
        return ScaleReference(False, reason)

    log.info(
        "  leaderboard scale built on %s: baseline + %d-split replicate anchor",
        bundle_id, n_splits,
    )
    return ScaleReference(
        True, None, str(outdir), bundle_id, n_splits, base_seed, dict(manifest)
    )


def rescale(score, reference: ScaleReference) -> dict[str, Any]:
    """Place one scored run on the scale, as `cell-eval2` computes it."""
    if not reference.built:
        return {"available": False, "reason": reference.reason}
    if score.aggregate_wide is None or score.run_meta is None:
        return {
            "available": False,
            "reason": "the scored run kept no aggregate or run metadata to enrol",
        }

    try:
        from cell_eval2.score import score_metrics

        frame = score_metrics(
            score.aggregate_wide,
            real_bundle=reference.bundle_dir,
            user_meta=score.run_meta,
        )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    rows = {str(row["metric"]): row for row in frame.iter_rows(named=True)}
    scaled = {
        metric: float(row[FROM_BASELINE])
        for metric, row in rows.items()
        if row.get(FROM_BASELINE) is not None and metric != AVERAGE
    }
    average = rows.get(AVERAGE, {}).get(FROM_BASELINE)
    replicate = {
        metric: float(row[FROM_REPLICATE])
        for metric, row in rows.items()
        if row.get(FROM_REPLICATE) is not None and metric != AVERAGE
    }
    return {
        "available": True,
        "from_baseline": scaled,
        "from_replicate": replicate,
        "avg_score": float(average) if average is not None else None,
        "bundle_id": reference.bundle_id,
    }
