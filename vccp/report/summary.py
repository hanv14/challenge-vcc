"""`reports/summary.md` — the results summary of CLAUDE.md §5.

It re-runs nothing. Every number here was written by an earlier stage; this
stage collects them into something that can be read on a projector and
stands on its own in a proposal defense.

When the run used `mini_data` the summary says so at the top, in the first
line, because the numbers there are not indicative of real performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..logging_utils import get_logger
from ..paths import DataPaths, RunPaths
from . import figures as figure_mod

MINI_WARNING = (
    "> **These numbers come from `mini_data` and are not indicative of real "
    "performance.** It is a deliberately tiny example of the real inputs — fewer "
    "genes, cells, targets and cell lines — so every number below says that the "
    "pipeline runs end to end, and none of them says how well the method works "
    "(CLAUDE.md §5)."
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path) if path.is_file() else None


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_no rows_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append(
            "| " + " | ".join(
                "" if v is None else str(v).replace("|", "\\|") for v in row
            ) + " |"
        )
    return "\n".join(out) + "\n"


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:,.{digits}f}"


def build_summary(cfg: Config) -> dict[str, Any]:
    """The `report` stage (checklist item 15)."""
    from .. import checklist as checklist_mod

    log = get_logger()
    run_paths = RunPaths(cfg)
    run_paths.ensure()
    paths = DataPaths(cfg)
    run_paths.figures.mkdir(parents=True, exist_ok=True)

    check_data = _read_json(run_paths.check_data_report)
    coverage = _read_csv(run_paths.prior_coverage)
    prior_checks = _read_json(run_paths.prior_checks)
    phase1 = _read_json(run_paths.phase_metrics("phase1"))
    phase2 = _read_json(run_paths.phase_metrics("phase2"))
    phase3 = _read_json(run_paths.phase_metrics("phase3"))
    rehearsal = _read_json(run_paths.rehearsal_report)
    forgetting = _read_json(run_paths.forgetting)
    sanity = _read_json(run_paths.sanity_report)
    validate = _read_json(run_paths.validate_report)
    predictions = _read_json(run_paths.prediction_index)
    policy = _read_json(run_paths.phase3_policy)

    drawn, skipped = _draw(cfg, run_paths, rehearsal, forgetting, phase3, sanity)
    # Built, not written, yet: item 15's own artifact is this summary, so the
    # checklist is only true once the summary exists.
    checklist = checklist_mod.build(cfg, run_paths)

    on_mini = cfg.on_mini_data
    text = _compose(
        cfg, run_paths, paths, on_mini,
        check_data=check_data, coverage=coverage, prior_checks=prior_checks,
        phase1=phase1, phase2=phase2, phase3=phase3, rehearsal=rehearsal,
        forgetting=forgetting, sanity=sanity, validate=validate,
        predictions=predictions, policy=policy, checklist=checklist,
        drawn=drawn, skipped=skipped,
    )
    run_paths.summary_md.write_text(text)
    checklist = checklist_mod.write(cfg, run_paths)
    log.info("summary written -> %s (%d figures)", run_paths.summary_md, len(drawn))
    log.info("\n%s", checklist_mod.summarize(checklist))
    return {
        "summary": str(run_paths.summary_md),
        "figures": {name: str(path) for name, path in drawn.items()},
        "figures_skipped": skipped,
        "checklist": checklist,
    }


def _draw(cfg, run_paths, rehearsal, forgetting, phase3, sanity):
    """Every figure §5 asks for, and the reason for each one not drawn."""
    log = get_logger()
    drawn: dict[str, Path] = {}
    skipped: dict[str, str] = {}

    curves = {
        phase: _read_csv(run_paths.phase_curves(phase)) for phase in ("phase1", "phase2")
    }
    sources = _knockdown_sources(run_paths)
    jobs = [
        ("rehearsal_official", lambda p: figure_mod.rehearsal_official(cfg, rehearsal, p)),
        ("rehearsal_per_gene", lambda p: figure_mod.rehearsal_per_gene(cfg, rehearsal, p)),
        ("rehearsal_scatter", lambda p: figure_mod.rehearsal_scatter(cfg, rehearsal, p)),
        ("forgetting", lambda p: figure_mod.forgetting(cfg, forgetting, p)),
        ("phase3_adaptation", lambda p: figure_mod.phase3_adaptation(cfg, phase3, p)),
        ("knockdown", lambda p: figure_mod.knockdown(
            cfg, sanity.get("knockdown", []), sources, p)),
        ("training_curves", lambda p: figure_mod.training_curves(cfg, curves, p)),
    ]
    for name, job in jobs:
        try:
            path, reason = job(run_paths.figure(name))
        except Exception as exc:  # noqa: BLE001 — a figure must never fail a run
            path, reason = None, f"{type(exc).__name__}: {exc}"
            log.warning("  figure %s could not be drawn: %s", name, reason)
        if path is None:
            skipped[name] = reason or "not drawn"
        else:
            drawn[name] = path
    return drawn, skipped


def _knockdown_sources(run_paths) -> dict[tuple, str]:
    """(context, target) -> where its `fold_expr` came from."""
    import pandas as pd

    out = {}
    directory = run_paths.phase_dir("phase3")
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("knockdown_*.csv")):
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            out[(str(row["context"]), str(row["target_gene"]))] = str(
                row.get("fold_expr_source", "pooled")
            )
    return out


def _compose(cfg, run_paths, paths, on_mini, **parts) -> str:
    from ..rehearsal import common

    checklist = parts["checklist"]
    sections: list[str] = []
    sections.append(f"# Virtual Cell Challenge 2026 — prototype results\n")
    stamp = checklist.get("provenance", {})
    git = stamp.get("git", {})
    sections.append(
        f"Run `{cfg.run_name}` · seed {cfg.seed} · data `{cfg.data_root}` · "
        f"config `{cfg.source}`\n"
    )
    sections.append(
        f"Built from commit `{git.get('short', 'unknown')}`"
        + (" **with uncommitted changes**" if git.get("dirty") else "")
        + f" on `{git.get('branch', 'unknown')}` · "
        + " · ".join(
            f"{name} {version}"
            for name, version in sorted(stamp.get("packages", {}).items())
            if name in ("torch", "cell-eval2")
        )
        + f" · {stamp.get('started_at', '')}\n"
    )
    if on_mini:
        sections.append(MINI_WARNING + "\n")

    # ---- data ------------------------------------------------------------
    sections.append("## 1. Data used\n")
    sections.append(_data_section(parts["check_data"], parts["coverage"]))

    # ---- method ----------------------------------------------------------
    sections.append("## 2. The method, as built\n")
    sections.append(_method_section(cfg, parts["policy"], parts["prior_checks"]))
    sections.append("### Element checklist\n")
    sections.append(
        _table(
            ["#", "element", "§", "status", "note"],
            [
                [item["item"], item["element"], item["section"], item["status"],
                 item["reason"] or ""]
                for item in checklist["items"]
            ],
        )
    )

    # ---- phases ----------------------------------------------------------
    sections.append("## 3. Phase 1 and Phase 2\n")
    sections.append(_phase_section(parts["phase1"], parts["phase2"]))
    sections.append(_figure_link(run_paths, parts, "training_curves",
                                 "Training curves for both phases."))

    # ---- rehearsal -------------------------------------------------------
    sections.append("## 4. Rehearsal — Phase 3's procedure where the answers are known\n")
    sections.append(_rehearsal_section(parts["rehearsal"], common))
    sections.append(_figure_link(run_paths, parts, "rehearsal_official",
                                 "The end-to-end variant, each arm against the no-change point."))
    sections.append(_figure_link(run_paths, parts, "rehearsal_per_gene",
                                 "The two partial variants, on the genes each predicts."))
    sections.append(_figure_link(run_paths, parts, "rehearsal_scatter",
                                 "Predicted against measured change for held-out targets."))

    # ---- forgetting ------------------------------------------------------
    sections.append("## 5. Forgetting\n")
    sections.append(_forgetting_section(parts["forgetting"]))
    sections.append(_figure_link(run_paths, parts, "forgetting",
                                 "Phase 1's validation under each phase's core."))

    # ---- phase 3 ---------------------------------------------------------
    sections.append("## 6. Phase 3 and the prediction\n")
    sections.append(_phase3_section(parts["phase3"], parts["predictions"]))
    sections.append(_figure_link(run_paths, parts, "phase3_adaptation",
                                 "Adaptation loss before and after, per context."))
    sections.append(_figure_link(run_paths, parts, "knockdown",
                                 "Each target gene's own predicted level against its control "
                                 "level, split by where its `fold_expr` came from."))

    # ---- sanity and submission ------------------------------------------
    sections.append("## 7. Sanity checks\n")
    sections.append(_sanity_section(parts["sanity"]))
    sections.append("## 8. Submission\n")
    sections.append(_submission_section(run_paths, parts["validate"], parts["predictions"]))

    if parts["skipped"]:
        sections.append("## 9. Figures not drawn\n")
        sections.append(
            _table(["figure", "reason"], [[k, v] for k, v in sorted(parts["skipped"].items())])
        )
    return "\n".join(sections)


def _figure_link(run_paths, parts, name: str, caption: str) -> str:
    if name not in parts["drawn"]:
        return f"_{caption} — not drawn: {parts['skipped'].get(name, 'unavailable')}._\n"
    return f"![{caption}](figures/{name}.png)\n\n_{caption}_\n"


def _data_section(check_data: dict[str, Any], coverage) -> str:
    if not check_data:
        return "_`check-data` has not run in this run directory._\n"

    sizes = check_data.get("sizes") or {}
    text = ""
    headline = [
        [key, _fmt(value)] for key, value in sizes.items()
        if not isinstance(value, (dict, list))
    ]
    contexts = sizes.get("contexts")
    if contexts:
        headline.append(["contexts", ", ".join(map(str, contexts))])
    if headline:
        text += _table(["quantity", "value"], headline) + "\n"

    rows = []
    for group in ("phase1", "phase2", "phase3"):
        entry = sizes.get(group)
        if isinstance(entry, dict) and entry:
            nested = all(isinstance(v, dict) for v in entry.values())
            items = entry.items() if nested else [(group, entry)]
            for name, values in items:
                rows.append([
                    group, name,
                    _fmt(values.get("n_cells")), _fmt(values.get("n_genes")),
                    _fmt(values.get("n_targets") or values.get("n_rows")),
                ])
    if rows:
        text += "### What each source contributes\n\n"
        text += _table(["phase", "source", "cells", "genes", "targets / rows"], rows) + "\n"
    if coverage is not None and len(coverage):
        built = coverage[coverage["built"]] if "built" in coverage else coverage
        text += "### Prior blocks\n\n"
        text += _table(
            ["block", "roles", "genes covered", "share of genes", "width"],
            [
                [row["block"], row.get("roles", ""), _fmt(row["n_covered"]),
                 f"{100 * float(row['fraction_covered']):.1f}%", _fmt(row["width"])]
                for _, row in built.iterrows()
            ],
        )
    return text or "_no data summary available_\n"


def _method_section(cfg, policy: dict[str, Any], prior_checks: dict[str, Any]) -> str:
    text = (
        "One shared gene-token model, updated by three phases. Genes are tokens, so each "
        "phase uses a different gene set with the same network:\n\n"
        "1. **Phase 1 — LINCS.** Control profile + perturbation token → signature "
        "(`sig`, `gmt`) → perturbed profile, held out by target gene.\n"
        "2. **Phase 2 — Replogle.** The panel→rest mapping on single cells, trained on "
        "control *and* perturbed cells with the same weights, plus the perturbation "
        "module adapted to Replogle.\n"
        "3. **Phase 3 — the challenge.** A context arrives as control cells and nothing "
        "else, so only the mapping is adapted; the perturbation module stays frozen.\n\n"
    )
    if policy:
        adapt = policy.get("adapt", {})
        text += "**What Phase 3 was allowed to adapt** (`phase3_policy.json`): "
        text += ", ".join(
            f"{key}={'yes' if value else 'no'}"
            for key, value in adapt.items() if isinstance(value, bool)
        )
        generator = policy.get("generator", {})
        text += (
            f". Generator `{generator.get('name')}` with confidence threshold "
            f"{_fmt(generator.get('confidence_threshold'), 3)} and effect scale "
            f"{_fmt(generator.get('effect_scale'), 3)}, both fitted on the rehearsal.\n\n"
        )
    text += _prior_check_section(prior_checks)
    return text


def _prior_check_section(prior_checks: dict[str, Any]) -> str:
    """The two §4.1 checks, as numbers rather than as a claim."""
    if not prior_checks:
        return ""
    text = "### Prior checks (CLAUDE.md §4.1)\n\n"

    families = [
        f for f in (prior_checks.get("neighbour_check") or {}).get("families", [])
        if f.get("status") == "ok"
    ]
    if families:
        text += (
            "Do complex members find each other? Each family's share of same-family "
            "nearest neighbours, against what chance would give:\n\n"
        )
        text += _table(
            ["gene family", "members", "same-family neighbours", "by chance", "enrichment"],
            [
                [f["family"], _fmt(f["n_members"]),
                 f"{100 * f['observed_same_family_fraction']:.1f}%",
                 f"{100 * f['expected_by_chance']:.2f}%",
                 f"{f['enrichment']:.1f}×"]
                for f in families
            ],
        ) + "\n"

    contexts = [
        c for c in (prior_checks.get("response_correlation_check") or {}).get("contexts", [])
        if c.get("status") == "ok"
    ]
    if contexts:
        text += (
            "Do genes with similar target-role embeddings have correlated knockdown "
            "responses?\n\n"
        )
        text += _table(
            ["screen", "targets", "pairs", "spearman", "p"],
            [[c["context"], _fmt(c["n_targets"]), _fmt(c["n_pairs"]),
              _fmt(c["spearman"], 3), f"{c['p_value']:.1e}"] for c in contexts],
        ) + "\n"

    magnitude = prior_checks.get("magnitude_check") or {}
    if magnitude.get("note"):
        text += f"_{magnitude['note']}_\n\n"
    return text


#: Phase 2 reports the mapping on control and on perturbed cells separately,
#: which is what checklist item 9 asks for.
PHASE2_COLUMNS = (
    ("map_control_mse", "mapping: control cells"),
    ("map_perturbed_mse", "mapping: perturbed cells"),
    # The two columns above are dominated by the baseline expression level.
    # This one is the *change* between them, which is the only quantity the
    # six official metrics score, reported beside the floor a perfect
    # predictor could reach at these cell counts (PLAN_PERCELL.md §5).
    ("map_delta_ratio_to_no_change", "mapping change ÷ no change"),
    ("map_delta_noise_floor_ratio", "…its noise floor"),
    ("pert_mse_ratio_to_no_change", "perturbation ÷ no change"),
    ("pert_pearson", "perturbation pearson"),
)


def _phase_section(phase1: dict[str, Any], phase2: dict[str, Any]) -> str:
    text = ""
    if phase1:
        held = phase1.get("validation", {})
        text += "### Phase 1 — LINCS, held-out target genes\n\n"
        text += _table(["metric", "value"],
                       [[k, _fmt(v)] for k, v in sorted(held.items())
                        if isinstance(v, (int, float))])
        per_line = phase1.get("validation_per_cell_line") or {}
        if per_line:
            columns = ["sig_pearson", "gmt_pearson", "pert_pearson", "n_rows"]
            text += "\nPer cell line (held-out targets only):\n\n" + _table(
                ["cell line", *columns],
                [[name, *[_fmt(values.get(c)) for c in columns]]
                 for name, values in sorted(per_line.items())],
            )
        text += "\n"

    if phase2:
        text += "### Phase 2 — Replogle, control and perturbed cells reported separately\n\n"
        rows = []
        for arm, entry in sorted((phase2.get("arms") or {}).items()):
            validation = entry.get("validation", {})
            label = arm + (" (used)" if arm == phase2.get("main_arm") else "")
            rows.append([label, *[_fmt(validation.get(key)) for key, _ in PHASE2_COLUMNS]])
        if rows:
            text += _table(["arm", *[label for _, label in PHASE2_COLUMNS]], rows)

        per_context = phase2.get("validation_per_context") or {}
        if per_context:
            text += "\nPer screen, held-out targets:\n\n" + _table(
                ["screen", *[label for _, label in PHASE2_COLUMNS]],
                [[name, *[_fmt(values.get(key)) for key, _ in PHASE2_COLUMNS]]
                 for name, values in sorted(per_context.items())],
            )
        text += "\n" + _phase1_contribution_text(phase2.get("phase1_contribution") or {})
    return text or "_no phase metrics available_\n"


def _phase1_contribution_text(contribution: dict[str, Any]) -> str:
    """What warm-starting from Phase 1 bought, as a table.

    Until this was measured the three-phase design rested on an assumption:
    both Phase 2 arms were warm-started, so nothing in the pipeline said
    whether the first phase contributed anything at all.
    """
    if not contribution.get("measured"):
        return (
            "\n**Phase 1's contribution:** not measured — "
            f"{contribution.get('reason', 'the core_scratch arm did not run')}.\n\n"
        )

    steps = contribution.get("n_phase2_steps", {})
    text = (
        "\n#### What Phase 1 contributed\n\n"
        f"`core_scratch` trains with no Phase 1 at all — no warm start and no "
        f"replayed objective — against `{contribution.get('warm_arm')}`, which has both. "
        "Every metric is a ratio against predicting no change, so a **positive** "
        "difference means Phase 1 helped by that much.\n\n"
    )
    text += _table(
        ["metric", "scratch − warm"],
        [[key, _fmt(value, 4)] for key, value in sorted(contribution["improvement"].items())],
    )
    text += (
        f"\nThe two arms did not spend equal effort on Phase 2's own objective: "
        f"the warm arm gave some of its steps to replaying Phase 1 "
        f"({steps.get('warm')} against {steps.get('scratch')}). That difference "
        "favours `core_scratch`, so a small positive number here is a stronger "
        "result than it looks, and a small negative one is weaker.\n\n"
    )
    return text


def _rehearsal_section(rehearsal: dict[str, Any], common) -> str:
    if not rehearsal:
        return "_the rehearsal has not run in this run directory._\n"

    text = ""
    rows = []
    for entry in rehearsal.get("variants", []):
        for key, label in ((common.END_TO_END, "end-to-end"),
                           (common.PANEL_ASSISTED, "PANEL-ASSISTED")):
            if key not in entry["arms"].get(common.METHOD, {}):
                continue
            for metric in ("pds_cosine", "expr_mse_unbiased_capped_norm",
                           "de_wilcoxon_lfc_nmae",
                           "de_wilcoxon_direction_fidelity_yield_raw",
                           "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard"):
                rows.append([
                    f"{entry['variant']}/{entry['context']}", label, metric,
                    *[
                        _fmt(entry["arms"].get(arm, {}).get(key, {})
                             .get("scored", {}).get(metric))
                        for arm in figure_mod.ARMS
                    ],
                ])
    if rows:
        text += "### The six official metrics\n\n"
        text += _table(["dataset", "reading", "metric", "method", "upper bound", "floor"], rows)
        text += (
            "\n`PANEL-ASSISTED` rows are fed the **true** panel values, so they are not "
            "end-to-end performance and must not be read as such (CLAUDE.md §4.6).\n\n"
        )

    # A dash is not an explanation: where the scorer refused a dataset, say so
    # and quote what it said.
    refusals = {}
    for entry in rehearsal.get("variants", []):
        for key in (common.END_TO_END, common.PANEL_ASSISTED):
            error = entry["arms"].get(common.METHOD, {}).get(key, {}).get("error")
            if error:
                refusals[f"{entry['variant']}/{entry['context']}"] = error
    if refusals:
        text += "**Datasets the scorer refused.**\n\n"
        for dataset, error in sorted(refusals.items()):
            text += f"- `{dataset}`: {error}\n"
        text += "\n"

    per_gene = []
    for entry in rehearsal.get("variants", []):
        for key in ("rest_genes", "hidden_genes"):
            if key not in entry["arms"].get(common.METHOD, {}):
                continue
            per_gene.append([
                f"{entry['variant']}/{entry['context']}", key,
                *[
                    _fmt(entry["arms"].get(arm, {}).get(key, {}).get("pearson"), 3)
                    + " / "
                    + _fmt(entry["arms"].get(arm, {}).get(key, {})
                           .get("mse_ratio_to_no_change"), 3)
                    for arm in figure_mod.ARMS
                ],
            ])
    if per_gene:
        text += "### Per gene, on the genes each partial variant predicts (pearson / error ÷ no change)\n\n"
        text += _table(["dataset", "genes", "method", "upper bound", "floor"], per_gene)
        text += "\n"

    scale = rehearsal.get("scale", {})
    if scale:
        text += "### The leaderboard's 0–1 scale\n\n"
        if scale.get("built_on"):
            text += f"Built on: {', '.join(scale['built_on'])}.\n\n"
            board_rows = []
            for entry in rehearsal.get("variants", []):
                for arm in figure_mod.ARMS:
                    board = entry["arms"].get(arm, {}).get(common.END_TO_END, {}).get(
                        "leaderboard", {})
                    if board.get("available"):
                        board_rows.append([
                            f"{entry['variant']}/{entry['context']}", arm,
                            _fmt(board.get("avg_score")),
                        ])
            if board_rows:
                text += _table(["dataset", "arm", "avg_score (0 = baseline, 1 = replicate)"],
                               board_rows) + "\n"
        for dataset, entry in sorted((scale.get("datasets") or {}).items()):
            if not entry.get("built"):
                text += f"- **{dataset}**: not built — {entry.get('reason')}\n"
        text += "\n"
    return text


def _forgetting_section(forgetting: dict[str, Any]) -> str:
    if not forgetting:
        return "_the forgetting report has not been written._\n"
    metric = forgetting.get("headline_metric", "sig_pearson")
    rows = []
    for name, entry in (forgetting.get("checkpoints") or {}).items():
        rows.append([
            name,
            str(entry.get("description", "")),
            _fmt((entry.get("metrics") or {}).get(metric)),
            _fmt((entry.get("change_from_phase1") or {}).get(metric)),
        ])
    text = _table(["core", "what it is", f"Phase 1 {metric}", "change vs Phase 1"], rows)
    guard = forgetting.get("guard") or {}
    if guard:
        text += "\n**Guard.** " + ", ".join(
            f"{key}: {value}" for key, value in guard.items()
        ) + "\n"
    ablation = forgetting.get("core_freeze_ablation") or {}
    if ablation.get("available"):
        text += (
            f"\n**Core-freeze ablation.** {ablation.get('reading', '')} "
            f"(difference {_fmt(ablation.get('difference'))} on {ablation.get('metric')}).\n"
        )
    return text


def _phase3_section(phase3: dict[str, Any], predictions: dict[str, Any]) -> str:
    text = ""
    if phase3:
        rows = [
            [name, _fmt(entry.get("loss_before")), _fmt(entry.get("loss_after")),
             _fmt(entry.get("loss_improvement"))]
            for name, entry in sorted((phase3.get("contexts") or {}).items())
        ]
        text += _table(["context", "mapping error before", "after", "improvement"], rows) + "\n"
    if predictions:
        rows = [
            [name, _fmt(entry.get("n_targets")), _fmt(entry.get("n_cells")),
             _fmt(entry.get("stored_entries")), _fmt(entry.get("knockdown_applied")),
             ", ".join(entry.get("targets_without_token") or []) or "—"]
            for name, entry in sorted((predictions.get("contexts") or {}).items())
        ]
        text += "### What was predicted\n\n"
        text += _table(
            ["context", "targets", "cells", "stored entries", "knockdown applied",
             "targets with no embedding"],
            rows,
        )
        prior = predictions.get("knockdown_prior", {})
        if prior:
            text += (
                f"\nKnockdown prior: pooled median `fold_expr` "
                f"{_fmt(prior.get('pooled_median_fold_expr'), 3)}, per screen "
                f"{ {k: round(v, 3) for k, v in (prior.get('median_per_screen') or {}).items()} }, "
                f"{prior.get('n_targets_with_own_value')} targets with their own value.\n"
            )
    return text or "_Phase 3 has not run in this run directory._\n"


def _sanity_section(sanity: dict[str, Any]) -> str:
    if not sanity:
        return "_the sanity checks have not run in this run directory._\n"
    rows = [
        [check["check"], check["name"], check["status"],
         _fmt(check["value"]) if not isinstance(check["value"], (dict, list)) else
         str(check["value"]), check["message"]]
        for check in sanity.get("checks", [])
    ]
    text = _table(["#", "check", "status", "value", "message"], rows)
    text += (
        f"\n{sanity.get('n_pass', 0)} passed, {sanity.get('n_warn', 0)} warned, "
        f"{sanity.get('n_fail', 0)} failed.\n"
    )
    if sanity.get("on_mini_data"):
        text += (
            "\nOn `mini_data` these statistics are noisy and warnings are expected; the "
            "checks still run and still report (CLAUDE.md §6.1).\n"
        )
    return text


def _submission_section(run_paths, validate: dict[str, Any], predictions: dict[str, Any]) -> str:
    if not validate:
        return "_the submission has not been validated in this run directory._\n"
    totals = validate.get("totals", {})
    requires = validate.get("requires", {})
    text = _table(
        ["quantity", "value"],
        [
            ["file", validate.get("source")],
            ["rows", _fmt(totals.get("n_rows"))],
            ["genes", _fmt(requires.get("n_genes"))],
            ["contexts", ", ".join(requires.get("contexts", []))],
            ["perturbations", _fmt(requires.get("n_perturbations"))],
            ["cells per perturbation", requires.get("cells_per_perturbation")],
            ["stored entries", _fmt(totals.get("stored_entries"))],
            ["largest cell total", _fmt(totals.get("max_cell_total"), 0)],
            ["file size (MB)", _fmt((totals.get("file_size_bytes") or 0) / 1024**2, 1)],
            ["all §8 rules pass", "yes" if validate.get("ok") else "NO"],
        ],
    )
    tool = validate.get("vcc_prep", {})
    if tool:
        if tool.get("ran"):
            text += (
                f"\n`vcc prep --dry-run`: {'accepted' if tool.get('ok') else 'refused'} "
                f"(exit {tool.get('returncode')}).\n"
            )
        else:
            text += f"\n`vcc prep` was not run: {tool.get('skipped_reason')}\n"
    return text
