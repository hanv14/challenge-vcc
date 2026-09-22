"""The `rehearsal` stage: run the variants, calibrate, write the policy.

Produces `rehearsal/report.json`, `rehearsal/summary.txt` and
`phase3_policy.json` (CLAUDE.md §4.6, checklist item 10). The policy is what
Phase 3 obeys: which modules it may adapt, and the two generator settings the
calibration chose.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..config import Config
from ..data import reference
from ..eval import nochange, official
from ..logging_utils import get_logger
from ..paths import DataPaths, RunPaths
from ..phases import phase2
from ..predict.generator import GeneratorSettings
from ..priors.build import PriorFeatures
from ..runtime import release_gpu_memory, resolve_device
from . import calibrate as calibration_module
from . import common, variants

POLICY_VERSION = 1


def run_rehearsal(cfg: Config) -> dict[str, Any]:
    log = get_logger()
    run_paths = RunPaths(cfg)
    run_paths.ensure()
    paths = DataPaths(cfg)
    device = resolve_device(cfg.device)

    features = PriorFeatures.load(run_paths.prior_features)
    axis = reference.load_gene_names(paths.gene_names)
    gene_index = {symbol: i for i, symbol in enumerate(axis)}
    contexts = phase2.load_contexts(cfg, device, gene_index)

    if not official.available():
        log.warning(
            "cell-eval2 is not installed: the rehearsal cannot score with the official "
            "metrics, so the generator keeps its default settings and the policy says so"
        )

    log.info("rehearsal: running the three variants (CLAUDE.md §4.6)")
    results: list[variants.VariantResult] = []

    results += variants.run_controls_only(cfg, features, contexts, gene_index, device, run_paths)
    release_gpu_memory()

    cross = variants.run_cross_context(
        cfg, features, contexts, gene_index, device, run_paths, paths
    )
    results += cross
    release_gpu_memory()

    results += variants.run_unseen_genes(
        cfg, features, contexts, gene_index, device, run_paths, axis
    )
    release_gpu_memory()

    policy_calibration = _calibrate(cfg, cross, contexts, gene_index, device, log)

    report = {
        "description": "Phase 3's procedure run on Replogle data whose answers are "
        "hidden, each variant reported against an upper bound and a no-change floor "
        "(CLAUDE.md §4.6)",
        "scorer": {
            "available": official.available(),
            "version": official.scorer_version(),
            "scored_metrics": list(official.SCORED),
        },
        "keys": {
            common.END_TO_END: "variant 2 only: whole predicted cells, end-to-end",
            common.PANEL_ASSISTED: "variants 1 and 3: the panel values fed in are the "
            "truth, so these are NOT end-to-end performance",
        },
        "variants": [result.as_dict() for result in results],
        "calibration": policy_calibration.as_dict(),
        "scale": _leaderboard_scale(cfg, results),
        "note": "on mini_data these numbers are not indicative of real performance "
        "(CLAUDE.md §5)",
    }

    run_paths.rehearsal_dir.mkdir(parents=True, exist_ok=True)
    run_paths.rehearsal_report.write_text(json.dumps(report, indent=2, default=str))
    run_paths.rehearsal_summary.write_text(summarize(report))

    policy = build_policy(cfg, policy_calibration, results)
    run_paths.phase3_policy.write_text(json.dumps(policy, indent=2, default=str))

    log.info("\n%s", summarize(report))
    log.info("rehearsal written -> %s", run_paths.rehearsal_report)
    log.info("phase 3 policy   -> %s", run_paths.phase3_policy)
    return report


def _calibrate(cfg, cross, contexts, gene_index, device, log):
    """Fit the generator on variant 2, which is the one scored end to end."""
    usable = [
        result
        for result in cross
        if common.METHOD in result.arms
        and "error" not in result.arms[common.METHOD].get(common.END_TO_END, {})
    ]
    if not usable:
        return calibration_module.default_calibration(
            "the cross-context variant produced no scorable arm, so the generator keeps "
            "its default settings"
        )

    # The context with the most targets gives the steadiest objective.
    chosen = max(usable, key=lambda r: r.detail.get("n_targets", 0))
    tensors = contexts[chosen.context]
    log.info("calibrating the generator on cross_context/%s", chosen.context)

    rng = np.random.default_rng(cfg.seed)
    gene_names, ctrl_mean, ctrl_std = variants.gene_axis(tensors)
    control_counts = variants.control_counts_for(tensors, cfg, rng)

    targets = list(chosen.detail.get("predictions", {}))
    stored = chosen.detail.get("predictions", {})
    if not stored:
        return calibration_module.default_calibration(
            "the cross-context variant stored no per-target predictions to calibrate on "
            "(raise rehearsal.n_saved_predictions)"
        )

    real_counts, real_labels, cells_per_target = variants.real_cells_for(
        tensors, targets, cfg, rng
    )
    predictions = {
        target: common.TargetPrediction(
            target, np.asarray(values, dtype=np.float32), {"arm": common.METHOD}
        )
        for target, values in stored.items()
        if target in cells_per_target
    }
    if len(predictions) < 2:
        return calibration_module.default_calibration(
            "fewer than two cross-context targets have both a stored prediction and "
            "real cells to score against"
        )

    return calibration_module.calibrate(
        cfg, predictions, control_counts, real_counts, real_labels, cells_per_target,
        gene_names, "cross_context", "non-targeting",
    )


def _leaderboard_scale(cfg, results) -> dict[str, Any]:
    """What §6's 0 = baseline / 1 = replicate scale came to, per dataset.

    It is attempted on every dataset that produces whole cells — the
    cross-context variant — and where `cell-eval2` refuses the pair (a
    degenerate end on data this small is the usual reason) the refusal itself
    is the reported reason and the six raw values stand alone.
    """
    attempts = {
        f"{result.variant}/{result.context}": result.detail["leaderboard_scale"]
        for result in results
        if "leaderboard_scale" in result.detail
    }
    return {
        "requested": cfg.eval.build_scale,
        "attempted_on": sorted(attempts),
        "built_on": sorted(k for k, v in attempts.items() if v.get("built")),
        "datasets": attempts,
        "note": "the no-change normalization in `calibration` is the common scale "
        "used for choosing the generator settings; this one is for reading the "
        "result the way the leaderboard would.",
    }


def build_policy(cfg, calibration, results) -> dict[str, Any]:
    """`phase3_policy.json`: what Phase 3 may adapt, and how it generates."""
    return {
        "version": POLICY_VERSION,
        "description": "what Phase 3 may adapt, and the generator calibration chosen "
        "from the rehearsal (CLAUDE.md §4.6, §4.7)",
        "adapt": {
            # Exactly what §4.7 permits: context adapters and delta for genes
            # first seen here, on the challenge controls.
            "context_adapters": True,
            "delta": True,
            "core": False,
            "perturbation_module": False,
            "heads": False,
            "reason": "a challenge context arrives as control cells and nothing else, "
            "so the mapping is the only thing it can support. What a knockdown does "
            "was learned in Phases 1 and 2 from data this context does not have.",
        },
        "generator": {
            "name": cfg.predict.generator,
            **calibration.settings.as_dict(),
        },
        "calibration": calibration.as_dict(),
        "evidence": [
            {
                "variant": result.variant,
                "context": result.context,
                "n_targets": result.detail.get("n_targets"),
            }
            for result in results
        ],
    }


def load_policy(path) -> dict[str, Any]:
    policy = json.loads(path.read_text())
    if policy.get("version") != POLICY_VERSION:
        raise ValueError(
            f"{path} is a version {policy.get('version')} policy but this build writes "
            f"version {POLICY_VERSION}; rerun the rehearsal stage"
        )
    return policy


def policy_settings(policy: dict[str, Any]) -> GeneratorSettings:
    return GeneratorSettings.from_dict(policy["generator"])


def summarize(report: dict[str, Any]) -> str:
    """The readable summary §4.6 asks for beside the JSON."""
    lines = ["Rehearsal — Phase 3's procedure where the answers are known", ""]
    if not report["scorer"]["available"]:
        lines.append("cell-eval2 is not installed; the official metrics were not computed.")
        lines.append("")

    for entry in report["variants"]:
        header = f"{entry['variant']} / {entry['context']}"
        lines.append(header)
        lines.append("-" * len(header))
        lines.append(f"  targets scored: {entry.get('n_targets', '?')}")

        for partial in ("rest_genes", "hidden_genes"):
            if partial in entry["arms"].get(common.METHOD, {}):
                lines.append(f"  {partial} (per-gene, on the genes this variant predicts):")
                for arm in (common.METHOD, common.UPPER_BOUND, common.FLOOR):
                    scores = entry["arms"].get(arm, {}).get(partial)
                    if scores:
                        lines.append(
                            f"    {arm:<12} pearson {scores['pearson']:+.3f}  "
                            f"mse/no-change {scores['mse_ratio_to_no_change']:.3f}"
                        )

        key = common.END_TO_END if common.END_TO_END in entry["arms"].get(
            common.METHOD, {}
        ) else common.PANEL_ASSISTED
        label = "official (end-to-end)" if key == common.END_TO_END else (
            "official (PANEL-ASSISTED — the panel fed in is the truth)"
        )
        scored_any = any(
            "scored" in entry["arms"].get(arm, {}).get(key, {})
            for arm in (common.METHOD, common.UPPER_BOUND, common.FLOOR)
        )
        if scored_any:
            lines.append(f"  {label}:")
            for metric in official.SCORED:
                row = [f"    {metric:<42}"]
                for arm in (common.METHOD, common.UPPER_BOUND, common.FLOOR):
                    value = entry["arms"].get(arm, {}).get(key, {}).get("scored", {}).get(metric)
                    row.append(f"{arm}={value:.4f}" if value is not None else f"{arm}=--")
                lines.append("  ".join(row))
            for arm in (common.METHOD, common.UPPER_BOUND, common.FLOOR):
                normalized = entry["arms"].get(arm, {}).get(key, {}).get("normalized", {})
                if normalized.get("objective") is not None:
                    lines.append(
                        f"    {arm:<12} objective vs no change: "
                        f"{normalized['objective']:+.4f}"
                    )
            for arm in (common.METHOD, common.UPPER_BOUND, common.FLOOR):
                board = entry["arms"].get(arm, {}).get(key, {}).get("leaderboard", {})
                if board.get("available") and board.get("avg_score") is not None:
                    lines.append(
                        f"    {arm:<12} leaderboard scale (0 = baseline, "
                        f"1 = replicate): {board['avg_score']:+.4f}"
                    )
        lines.append("")

    calibration = report["calibration"]
    lines.append("Generator calibration (fitted on the cross-context variant)")
    lines.append("-" * 58)
    lines.append(f"  confidence_threshold : {calibration['settings']['confidence_threshold']}")
    lines.append(f"  effect_scale         : {calibration['settings']['effect_scale']}")
    if calibration.get("fallback_reason"):
        lines.append(f"  fallback: {calibration['fallback_reason']}")
    lines.append("")

    scale = report["scale"]
    lines.append("Leaderboard scale (0 = mean-response baseline, 1 = split-half replicate)")
    lines.append("-" * 70)
    if scale["built_on"]:
        lines.append(f"  built on: {', '.join(scale['built_on'])}")
    for dataset, entry in sorted(scale["datasets"].items()):
        if not entry.get("built"):
            lines.append(f"  {dataset}: not built — {entry.get('reason')}")
    if not scale["datasets"]:
        lines.append("  not attempted: no variant produced whole cells to enrol")
    lines.append("")
    lines.append(report["note"])
    return "\n".join(lines)
