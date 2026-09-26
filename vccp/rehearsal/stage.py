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

#: Bumped to 2 when `adapt` stopped being a fixed block and became a reading
#: off rehearsal variant 1 (DECISIONS.md D94). A version 1 policy says
#: `context_adapters: true` because it was written that way, not because it
#: was measured, so Phase 3 refuses it and asks for the rehearsal again
#: rather than adapting on the strength of a claim nobody checked.
POLICY_VERSION = 2


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

    # Cross-assay: LINCS to Replogle, the boundary every submission crosses
    # and the only one the other three variants never test (§7.7). Evidence
    # for the write-up rather than score, so it is opt-in.
    if cfg.rehearsal.cross_assay:
        log.info("rehearsal: the cross-assay variant (LINCS -> Replogle)")
        results += variants.run_cross_assay(
            cfg, features, contexts, gene_index, device, run_paths, paths
        )
        release_gpu_memory()

    # The A/B that decides `phase2.perturbation_mode` (PLAN_PERCELL.md §12).
    # Off unless asked for: it retrains one Phase 2 arm per configuration, so
    # it is a deliberate measurement rather than something every run pays for.
    sweep: list[variants.VariantResult] = []
    if cfg.rehearsal.mode_sweep:
        log.info("rehearsal: the mode sweep (perturbation_mode, ot_epsilon)")
        sweep = variants.run_mode_sweep(
            cfg, features, contexts, gene_index, device, run_paths, paths
        )
        results += sweep
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
            common.END_TO_END: "variants 2 and cross_assay: whole predicted cells, "
            "end-to-end",
            common.PANEL_ASSISTED: "variants 1 and 3: the panel values fed in are the "
            "truth, so these are NOT end-to-end performance",
        },
        "variants": [result.as_dict() for result in results],
        "mode_sweep": _mode_sweep_table(cfg, sweep),
        "calibration": policy_calibration.as_dict(),
        "scale": _leaderboard_scale(cfg, results),
        "note": (
            "on mini_data these numbers are not indicative of real performance "
            "(CLAUDE.md §5)"
            if cfg.on_mini_data
            else f"measured on the real data under {cfg.data_root}"
        ),
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

    calibration = calibration_module.calibrate(
        cfg, predictions, control_counts, real_counts, real_labels, cells_per_target,
        gene_names, "cross_context", "non-targeting",
    )
    # Where the settings came from and where they are going. The modalities
    # usually match, since Replogle and the challenge are both CRISPR
    # interference — which is why `carried_across_assays` is recorded
    # unconditionally beside them: a matching modality is not the same assay,
    # and these settings still cross a change of lab, protocol and sequencing
    # depth that nothing in the model represents (PLAN_PERCELL.md §7.5).
    calibration.fitted_assay = cfg.phase2.pert_type
    calibration.applied_assay = cfg.phase3.pert_type
    calibration.per_dataset = _per_dataset_calibration(
        cfg, usable, chosen, contexts, device, log
    )
    calibration.per_dataset["spread"] = _settings_spread(
        chosen.context, calibration.settings, calibration.per_dataset
    )
    return calibration


def _per_dataset_calibration(cfg, usable, chosen, contexts, device, log) -> dict[str, Any]:
    """The same fit on the other screens, when it was asked for.

    One global setting fitted on one screen and applied to the challenge is
    an assumption; fitting it on every screen says whether the assumption
    holds. Screens that agree are evidence the transfer is safe, screens that
    disagree are evidence it is not, and either reading is worth having before
    a submission rests on it.

    Off by default because each screen costs a full grid of official
    scorings, which is the slow part of the rehearsal.
    """
    others = [r for r in usable if r.context != chosen.context]
    if not cfg.rehearsal.calibrate_per_dataset:
        return {
            "ran": False,
            "reason": "rehearsal.calibrate_per_dataset is off",
            "would_have_fitted": [r.context for r in others],
        }
    if not others:
        return {"ran": False, "reason": "only one screen produced a scorable arm"}

    fits = {}
    for result in others:
        tensors = contexts[result.context]
        stored = result.detail.get("predictions", {})
        if not stored:
            fits[result.context] = {"error": "no stored predictions"}
            continue
        rng = np.random.default_rng(cfg.seed)
        gene_names, _, _ = variants.gene_axis(tensors)
        control_counts = variants.control_counts_for(tensors, cfg, rng)
        targets = list(stored)
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
            fits[result.context] = {"error": "fewer than two scorable targets"}
            continue
        log.info("  calibrating again on cross_context/%s, for the spread", result.context)
        fit = calibration_module.calibrate(
            cfg, predictions, control_counts, real_counts, real_labels,
            cells_per_target, gene_names, "cross_context", "non-targeting",
        )
        fits[result.context] = fit.settings.as_dict()

    return {
        "ran": True,
        "fitted_on": chosen.context,
        "others": fits,
        "reading": (
            "settings that agree across screens are evidence the transfer to the "
            "challenge is safe; settings that disagree are evidence it is not"
        ),
    }


def _settings_spread(chosen_context: str, chosen, per_dataset: dict[str, Any]) -> dict[str, Any]:
    """How far the fitted settings move between screens.

    The number that says whether one screen's calibration can be trusted on
    another assay. `spread` is max minus min over the screens that fitted;
    `agree` is whether both settings stayed inside one step of the grid.
    """
    fits = {chosen_context: chosen.as_dict()}
    for context, entry in (per_dataset.get("others") or {}).items():
        if "error" not in entry:
            fits[context] = entry
    if len(fits) < 2:
        return {"measured": False, "reason": "only one screen fitted", "fits": fits}

    spread = {}
    for key in ("confidence_threshold", "effect_scale"):
        values = [float(f[key]) for f in fits.values() if key in f]
        spread[key] = {"min": min(values), "max": max(values), "range": max(values) - min(values)}
    return {
        "measured": True,
        "fits": fits,
        "spread": spread,
        "reading": (
            "a wide range here means the generator's settings are a property of "
            "the screen they were fitted on rather than of the method, and "
            "carrying them to the challenge — a different lab and protocol — is "
            "the weakest link in the submission path"
        ),
    }


def _mode_sweep_table(cfg, sweep) -> dict[str, Any]:
    """The A/B as a table: one row per configuration, leaderboard scale where
    it could be built, the six raw metrics either way.

    `overall` is the mean of the six on the leaderboard's 0–1 scale, which is
    what the leaderboard reports and what disagreed with the calibration
    objective the last time the two were compared. The floor row is predicting
    that nothing happened, and a configuration that does not beat it has not
    earned a submission.
    """
    if not cfg.rehearsal.mode_sweep:
        return {"ran": False, "reason": "rehearsal.mode_sweep is off"}
    if not sweep:
        return {"ran": False, "reason": "the sweep produced no scorable screen"}

    datasets = {}
    for result in sweep:
        rows = []
        for arm in sorted(result.arms):
            scored = result.arms[arm].get(common.END_TO_END, {})
            if "error" in scored:
                rows.append({"arm": arm, "error": scored["error"]})
                continue
            scale = scored.get("leaderboard_scale") or {}
            rows.append({
                "arm": arm,
                "overall": scale.get("overall"),
                "scaled": scale.get("metrics"),
                "raw": {k: scored.get(k) for k in official.SCORED if k in scored},
            })
        scorable = [r for r in rows if r.get("overall") is not None]
        datasets[result.context] = {
            "n_targets": result.detail.get("n_targets"),
            "learned_from": result.detail.get("learned_from"),
            "rows": rows,
            "best": (
                max(scorable, key=lambda r: r["overall"])["arm"] if scorable else None
            ),
            "scale_built": bool(scorable),
        }
    return {
        "ran": True,
        "epsilons": list(cfg.rehearsal.mode_sweep_epsilons),
        "datasets": datasets,
        "note": "one Phase 2 arm per configuration, everything else held fixed; "
                "`overall` is the mean of the six official metrics on the "
                "leaderboard's 0 = baseline / 1 = replicate scale",
    }


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


#: How much worse than predicting no change the controls-only mapping may be
#: before Phase 3 is told not to adapt at all. Slightly above 1.0 so that a
#: mapping which merely ties with the baseline is still allowed to try.
CONTROLS_ONLY_TOLERANCE = 1.02


def _adapt_policy(results) -> dict[str, Any]:
    """Whether Phase 3 may adapt, read off rehearsal variant 1.

    §4.6 says the policy states what Phase 3 may adapt *chosen from these
    results*, and variant 1 is the measurement that bears on it: it fits the
    mapping on a held-out context's controls alone — exactly Phase 3's
    procedure — and scores the rest genes it then predicts. If that comes
    out worse than predicting no change, adapting on the challenge controls
    is not a neutral step that might help, it is a measured harm, and the
    policy has to say so instead of permitting it because §4.7 describes it.

    The first full server run measured 1.484, 1.509 and 1.468 against a
    floor of 1.0, with the same mapping reaching 0.989 when it was allowed
    the perturbed cells too (DECISIONS.md D94).
    """
    ratios = {}
    for result in results:
        if result.variant != "controls_only":
            continue
        scores = result.arms.get(common.METHOD, {}).get("rest_genes")
        if scores and np.isfinite(scores.get("mse_ratio_to_no_change", float("nan"))):
            ratios[result.context] = float(scores["mse_ratio_to_no_change"])

    permitted = {
        "core": False,
        "perturbation_module": False,
        "heads": False,
        "measured_on": "rehearsal variant 1 (controls_only), rest genes, "
        "mse against predicting no change",
        "controls_only_ratio_per_context": ratios,
    }
    if not ratios:
        return {
            **permitted,
            "context_adapters": True,
            "delta": True,
            "measured": False,
            "reason": "variant 1 did not run, so §4.7's design stands as written: a "
            "challenge context arrives as control cells and nothing else, so the "
            "mapping is the only thing it can support.",
        }

    worst = max(ratios.values())
    median = float(np.median(list(ratios.values())))
    helps = median <= CONTROLS_ONLY_TOLERANCE
    return {
        **permitted,
        "context_adapters": helps,
        "delta": helps,
        "measured": True,
        "controls_only_ratio_median": round(median, 4),
        "controls_only_ratio_worst": round(worst, 4),
        "reason": (
            "variant 1 fits the mapping on a held-out context's controls alone — "
            f"Phase 3's own procedure — and reaches {median:.3f} against predicting "
            "no change (1.0). Adapting is at worst harmless, so it is permitted."
            if helps
            else
            "variant 1 fits the mapping on a held-out context's controls alone — "
            f"Phase 3's own procedure — and reaches {median:.3f} against predicting "
            f"no change (1.0), worst {worst:.3f}. Adapting on controls is therefore "
            "measured to make the rest genes worse than leaving them alone, so "
            "Phase 3 keeps Phase 2's mapping unchanged. This is a departure from "
            "§4.7's described procedure, taken on §4.6's instruction that the "
            "policy be chosen from the rehearsal."
        ),
    }


def build_policy(cfg, calibration, results) -> dict[str, Any]:
    """`phase3_policy.json`: what Phase 3 may adapt, and how it generates."""
    return {
        "version": POLICY_VERSION,
        "description": "what Phase 3 may adapt, and the generator calibration chosen "
        "from the rehearsal (CLAUDE.md §4.6, §4.7)",
        "adapt": _adapt_policy(results),
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


def _wrap(message: str, indent: int, width: int = 96) -> str:
    """One long scorer message, folded so a terminal can show it."""
    import textwrap

    text = " ".join(str(message).split())
    lines = textwrap.wrap(text, width=width) or [""]
    pad = " " * indent
    return ("\n" + pad).join(lines)


def _summarize_mode_sweep(sweep: dict[str, Any]) -> list[str]:
    """The A/B table, first in the summary because it decides the default."""
    if not sweep.get("ran"):
        return []

    lines = ["Mode sweep — which perturbation_mode, and at which ot_epsilon",
             "=" * 62]
    for context, entry in sorted(sweep.get("datasets", {}).items()):
        lines.append(
            f"  {context}: {entry.get('n_targets', '?')} targets, learned from "
            f"{', '.join(entry.get('learned_from') or [])}"
        )
        rows = entry.get("rows", [])
        # When every row fails the same way the failure is about the data,
        # not about the configurations — saying it once is the readable form,
        # and repeating a 500-character scorer message per row is not.
        failed = [row for row in rows if "error" in row]
        errors = {row["error"] for row in failed}
        shared_error = (
            next(iter(errors)) if rows and len(errors) == 1 and len(failed) == len(rows) else None
        )
        if shared_error:
            lines.append(
                f"    all {len(rows)} configurations failed to score, identically:"
            )
            lines.append(f"      {_wrap(shared_error, 8)}")
            lines.append(
                "    that is a property of this dataset, not of the configurations: "
                "they all ran."
            )
            lines.append("")
            continue

        if not entry.get("scale_built"):
            lines.append(
                "    the leaderboard scale could not be built on this dataset, so "
                "only the six raw metrics are available (see report.json)"
            )
        lines.append(f"    {'arm':<18}{'overall':>10}   the six, scaled")
        for row in rows:
            if "error" in row:
                lines.append(f"    {row['arm']:<18}{'—':>10}   {_wrap(row['error'], 24)}")
                continue
            overall = row.get("overall")
            scaled = row.get("scaled") or {}
            detail = "  ".join(
                f"{official.short_name(k)} {v:+.2f}"
                for k in official.SCORED
                if (v := scaled.get(k)) is not None
            ) if scaled else "(raw only)"
            lines.append(
                f"    {row['arm']:<18}"
                + (f"{overall:>+10.4f}" if overall is not None else f"{'—':>10}")
                + f"   {detail}"
            )
        best = entry.get("best")
        if best:
            lines.append(
                f"    best: {best}. Read it against the `floor` row — a configuration "
                "that does not beat predicting no change has not earned a submission."
            )
        lines.append("")
    lines.append(
        "  One Phase 2 arm per row, everything else held fixed: same screen, same "
        "targets,"
    )
    lines.append(
        "  same control draw, same seed. `overall` is the mean of the six official "
        "metrics"
    )
    lines.append("  on the leaderboard's 0 = baseline / 1 = replicate scale.")
    lines.append("")
    return lines


def summarize(report: dict[str, Any]) -> str:
    """The readable summary §4.6 asks for beside the JSON."""
    lines = ["Rehearsal — Phase 3's procedure where the answers are known", ""]
    if not report["scorer"]["available"]:
        lines.append("cell-eval2 is not installed; the official metrics were not computed.")
        lines.append("")

    lines += _summarize_mode_sweep(report.get("mode_sweep") or {})

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
        # The three standard arms, plus whatever else this variant produced —
        # the cross-assay variant adds a second method arm asked as the
        # modality the weights were trained on, and an arm that is not
        # printed is an arm that was measured for nothing.
        standard = (common.METHOD, common.UPPER_BOUND, common.FLOOR)
        arm_order = list(standard) + [a for a in sorted(entry["arms"]) if a not in standard]
        scored_any = any(
            "scored" in entry["arms"].get(arm, {}).get(key, {}) for arm in arm_order
        )
        if scored_any:
            lines.append(f"  {label}:")
            for metric in official.SCORED:
                row = [f"    {official.short_name(metric):<8}"]
                for arm in arm_order:
                    value = entry["arms"].get(arm, {}).get(key, {}).get("scored", {}).get(metric)
                    row.append(f"{arm}={value:.4f}" if value is not None else f"{arm}=--")
                lines.append("  ".join(row))
            for arm in arm_order:
                normalized = entry["arms"].get(arm, {}).get(key, {}).get("normalized", {})
                if normalized.get("objective") is not None:
                    lines.append(
                        f"    {arm:<24} objective vs no change: "
                        f"{normalized['objective']:+.4f}"
                    )
            for arm in arm_order:
                board = entry["arms"].get(arm, {}).get(key, {}).get("leaderboard", {})
                if board.get("available") and board.get("avg_score") is not None:
                    lines.append(
                        f"    {arm:<24} leaderboard scale (0 = baseline, "
                        f"1 = replicate): {board['avg_score']:+.4f}"
                    )
            if entry.get("modalities"):
                lines.append(
                    "    the two method arms are the same weights asked as "
                    f"{entry['modalities'].get(common.METHOD)} and as "
                    f"{entry['modalities'].get(variants.SOURCE_MODALITY)}; the gap "
                    "is what the knockout-specific type offset is worth here"
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
