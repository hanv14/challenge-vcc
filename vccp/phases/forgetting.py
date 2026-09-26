"""What each phase costs the ones before it (CLAUDE.md §4.3, item 7).

Phase 1's held-out-target validation is re-scored after Phase 1, after
Phase 2 and after Phase 3, always through Phase 1's own adapter and Phase 1's
own data. The core and the gene vocabulary are shared, so a later phase can
move them; this is what says by how much.

It also settles the one design choice the specification leaves open. §4.3's
default freezes the core in every later phase, which would leave the guard
with almost nothing to act on; the M0 review flipped Phase 2's default to
training the core, on the argument that Phase 2 has the data for it
(DECISIONS.md D3). Phase 2 therefore trains a second arm with the core frozen,
and both arms are scored here — so the default rests on a measured difference
rather than on the argument.

Reading it: `phase1_after_phase1` is the reference. A later number that is
worse means the later phase cost Phase 1 something; the replay and L2-SP
settings (`train.replay_fraction`, `train.l2sp_weight`) are what bound it.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..config import Config
from ..logging_utils import get_logger
from ..paths import RunPaths
from . import phase1 as phase1_module

#: Checkpoints to score, in the order they are produced. A missing one is
#: skipped rather than fabricated — the stage is useful after Phase 2 and
#: again after Phase 3.
CHECKPOINTS = (
    ("phase1", "phase1", "Phase 1 itself: the reference"),
    ("phase2", "phase2", "after Phase 2, main arm"),
    ("phase2_core_frozen", "phase2", "after Phase 2, core-frozen ablation arm"),
    ("phase2_core_unfrozen", "phase2", "after Phase 2, core-unfrozen ablation arm"),
    ("phase3", "phase3", "after Phase 3"),
)

#: The metric the comparison is read on. Higher is better.
HEADLINE = "sig_pearson"


def score_checkpoint(cfg: Config, run_paths: RunPaths, name: str, device: str, tensors) -> dict:
    """Phase 1 validation under one saved core."""
    from ..models.checkpoint import load_core
    from ..models.core import build_model
    from ..priors.build import PriorFeatures

    features = PriorFeatures.load(run_paths.prior_features)
    model = build_model(cfg, features).to(device)
    for adapter in ("phase1", "phase2", "phase3"):
        model.add_adapter(adapter)

    load_core(
        run_paths.core_checkpoint(name),
        model,
        layout_hash=features.layout_hash(),
        strict=False,
    )
    # Always through Phase 1's adapter: the question is what Phase 1's own
    # configuration can still do, not what a later phase's can.
    model.use_adapters([phase1_module.ADAPTER])
    return phase1_module.evaluate(model, tensors, cfg)


def run_forgetting(cfg: Config) -> dict[str, Any]:
    """The `forgetting` stage."""
    from ..runtime import resolve_device

    log = get_logger()
    run_paths = RunPaths(cfg)
    run_paths.ensure()
    device = resolve_device(cfg.device)

    tensors = phase1_module.load_phase1_tensors(cfg, device)

    scores: dict[str, dict] = {}
    for name, phase, description in CHECKPOINTS:
        path = run_paths.core_checkpoint(name)
        if not path.is_file():
            continue
        scores[name] = {
            "phase": phase,
            "description": description,
            "metrics": score_checkpoint(cfg, run_paths, name, device, tensors),
        }

    if not scores:
        raise RuntimeError(
            f"no core checkpoint under {run_paths.checkpoints}; run the phases first"
        )

    for name, entry in scores.items():
        entry["measurable"] = _measurable(entry["metrics"])
        if not entry["measurable"]:
            entry["not_measurable_because"] = _why_unmeasurable(cfg, name)

    reference = scores.get("phase1", {}).get("metrics", {})
    for name, entry in scores.items():
        entry["change_from_phase1"] = {
            key: round(float(value - reference[key]), 6)
            for key, value in entry["metrics"].items()
            if key in reference and np.isfinite(value) and np.isfinite(reference[key])
        }

    report = {
        "description": "Phase 1 held-out-target validation, re-scored under each "
        "phase's saved core through Phase 1's own adapter (CLAUDE.md §4.3)",
        "headline_metric": HEADLINE,
        "guard": {
            "replay_fraction": cfg.train.replay_fraction,
            "l2sp_weight": cfg.train.l2sp_weight,
        },
        "checkpoints": scores,
        "core_freeze_ablation": _ablation(cfg, scores),
        "note": (
            "on mini_data these numbers are noisy and not indicative; the "
            "comparison is the point, not the level"
            if cfg.on_mini_data
            else "the comparison between cores is the point, not the level"
        ),
    }

    run_paths.forgetting.write_text(json.dumps(report, indent=2, default=str))

    log.info("forgetting — Phase 1 validation under each saved core:")
    for name, entry in scores.items():
        headline = entry["metrics"].get(HEADLINE)
        change = entry["change_from_phase1"].get(HEADLINE)
        if not entry["measurable"]:
            # Printing `+nan` invited reading it as a number; the server run
            # then drew a recommendation out of one (DECISIONS.md D93).
            log.info("  %-24s not measurable — %s", name, entry["not_measurable_because"])
            continue
        log.info(
            "  %-24s %s %s",
            name,
            f"{HEADLINE} {headline:+.4f}" if headline is not None else "(no score)",
            f"({change:+.4f} vs phase 1)" if change is not None else "",
        )

    ablation = report["core_freeze_ablation"]
    if ablation.get("available"):
        log.info(
            "  core-freeze ablation: %s is better on %s by %+.4f  ->  %s",
            ablation["better_arm"],
            HEADLINE,
            ablation["difference"],
            ablation["reading"],
        )
    else:
        log.info("  core-freeze ablation: %s", ablation["reason"])

    return report


def _measurable(metrics: dict[str, Any]) -> bool:
    """Whether this core produced a Phase 1 score that is a number.

    A core that never carried Phase 1's weights has an untrained `sig` head,
    which is zero-initialized and so predicts a constant — and the Pearson
    of a constant is NaN, not a bad score.
    """
    value = metrics.get(HEADLINE)
    return value is not None and bool(np.isfinite(value))


def _why_unmeasurable(cfg: Config, name: str) -> str:
    if name.startswith("phase2") or name.startswith("phase3"):
        if cfg.phase2.warm_start == "none":
            return (
                "`phase2.warm_start: none`, so this core never carried Phase 1's "
                "weights: its Phase 1 head is untrained and predicts a constant. "
                "There is nothing of Phase 1 in it to forget, which is the D91 "
                "deviation showing up here rather than a failure of the guard"
            )
    return (
        f"{HEADLINE} came back non-finite under this core — a head that predicts a "
        "constant has no correlation to report"
    )


def _ablation(cfg: Config, scores: dict[str, dict]) -> dict[str, Any]:
    """Compare the two Phase 2 arms on Phase 1 validation."""
    configured = "core_unfrozen" if cfg.phase2.unfreeze_core else "core_frozen"
    other = "core_frozen" if cfg.phase2.unfreeze_core else "core_unfrozen"

    main = scores.get("phase2", {}).get("metrics", {})
    ablation = scores.get(f"phase2_{other}", {}).get("metrics", {})

    if HEADLINE not in main or HEADLINE not in ablation:
        return {
            "available": False,
            "reason": "the ablation arm was not trained "
            "(train.core_freeze_ablation is off, or Phase 2 has not run)",
            "configured_arm": configured,
        }

    # A comparison against NaN answers every question the same way, and the
    # server run turned one into "consider flipping phase2.unfreeze_core".
    unmeasurable = [
        name
        for name, metrics in ((configured, main), (other, ablation))
        if not _measurable(metrics)
    ]
    if unmeasurable:
        return {
            "available": False,
            "reason": (
                f"{', '.join(unmeasurable)} scored non-finite on {HEADLINE}: "
                + _why_unmeasurable(cfg, "phase2")
            ),
            "configured_arm": configured,
        }

    main_score = float(main[HEADLINE])
    other_score = float(ablation[HEADLINE])
    better = configured if main_score >= other_score else other

    return {
        "available": True,
        "metric": HEADLINE,
        "configured_arm": configured,
        "arms": {configured: main_score, other: other_score},
        "better_arm": better,
        "difference": round(abs(main_score - other_score), 6),
        "reading": (
            f"the configured default ({configured}) retains more of Phase 1"
            if better == configured
            else f"{other} retains more of Phase 1 than the configured default "
            f"({configured}); consider flipping phase2.unfreeze_core"
        ),
        "caveat": "this compares what each arm cost Phase 1, not what it gained on "
        "Phase 2 — read it beside phase2/metrics.json. And read the difference "
        "against run-to-run variation before acting on it: on mini_data, where "
        "Phase 1 has ~130 training rows, the verdict flips between seeds, so a "
        "difference of this size is not evidence either way. The comparison is "
        "built for the server, where it is not.",
    }
