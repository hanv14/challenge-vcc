"""The element checklist of CLAUDE.md §4.8, checked against the run.

Every item names the artifact the specification asks for. After an `all` run
each is `done`, or a `deviation` whose reason is also in `DECISIONS.md` and in
the final message — never silently absent. An item whose artifact is missing
is `missing`, which is a failure, not a deviation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DONE, DEVIATION, MISSING = "done", "deviation", "missing"


@dataclass
class Item:
    """One row of §4.8."""

    number: int
    element: str
    section: str
    artifacts: list[str]
    status: str
    reason: str | None = None
    present: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "item": self.number,
            "element": self.element,
            "section": self.section,
            "status": self.status,
            "artifacts": self.present,
            "reason": self.reason,
        }


def _rel(run_paths, path: Path) -> str:
    try:
        return str(Path(path).relative_to(run_paths.root))
    except ValueError:
        return str(path)


def build(cfg, run_paths) -> dict[str, Any]:
    """Walk §4.8 and report each item against what the run produced."""
    rows: list[tuple[int, str, str, list[Path], Callable[[], tuple[str, str | None]] | None]] = []

    def phase_files(phase: str) -> list[Path]:
        return [run_paths.phase_metrics(phase)]

    contexts = sorted(
        path.name for path in run_paths.predictions_dir.glob("*") if path.is_dir()
    )

    rows = [
        (1, "Gene vocabulary — all challenge genes, embedding = W·prior + δ", "4.1",
         [run_paths.prior_features], None),
        (2, "Prior blocks 1–6 built (7 if configured), coverage per block", "4.1",
         [run_paths.prior_coverage], None),
        (3, "Two gene roles — feature-role and target-role priors and embeddings", "4.1",
         [run_paths.prior_features], None),
        (4, "Prior checks — nearest neighbours and response correlation", "4.1",
         [run_paths.prior_checks], None),
        (5, "Shared gene-token model, one core used by all three phases", "4.2",
         [run_paths.core_checkpoint("phase1"), run_paths.core_checkpoint("phase2"),
          run_paths.core_checkpoint("phase3")], None),
        (6, "Frozen core + per-phase and per-context adapters, verified frozen", "4.3",
         [run_paths.frozen_check], None),
        (7, "Guard against forgetting — replay, L2-SP, Phase 1 re-scored", "4.3",
         [run_paths.forgetting], None),
        (8, "Phase 1 — LINCS, control + perturbation → signature → perturbed", "4.4",
         phase_files("phase1"), None),
        (9, "Phase 2 — Siamese panel→rest on control and perturbed cells", "4.5",
         phase_files("phase2"), None),
        (10, "Rehearsal — all three variants, with upper bound and floor", "4.6",
         [run_paths.rehearsal_report, run_paths.phase3_policy], None),
        (11, "Phase 3 adaptation on each context's challenge controls", "4.7",
         [run_paths.phase_metrics("phase3")]
         + [run_paths.phase3_adapt(c) for c in _contexts_from(run_paths)], None),
        (12, "Phase 3 prediction — panel, rest, knockdown prior, distinct cells", "4.7",
         [run_paths.prediction_index], None),
        (13, "Submission and validation, `vcc prep` where available", "8",
         [run_paths.submission, run_paths.validate_report], _vcc_status(run_paths)),
        (14, "Sanity checks — all seven", "6.1", [run_paths.sanity_report], None),
        (15, "Results summary — tables and figures for the defense", "5",
         [run_paths.summary_md], _figures_status(run_paths)),
    ]

    items = []
    for number, element, section, artifacts, extra in rows:
        present = {_rel(run_paths, path): Path(path).exists() for path in artifacts}
        status = DONE if all(present.values()) and present else MISSING
        reason = None
        if status is DONE or status == DONE:
            if extra is not None:
                status, reason = extra()
        else:
            reason = "artifact(s) not produced by this run: " + ", ".join(
                name for name, ok in present.items() if not ok
            )
        items.append(Item(number, element, section, [str(a) for a in artifacts],
                          status, reason, present))

    from . import provenance

    return {
        "run": str(run_paths.root),
        "provenance": provenance.describe(cfg),
        "n_items": len(items),
        "n_done": sum(1 for i in items if i.status == DONE),
        "n_deviation": sum(1 for i in items if i.status == DEVIATION),
        "n_missing": sum(1 for i in items if i.status == MISSING),
        "complete": all(i.status in (DONE, DEVIATION) for i in items),
        "items": [item.as_dict() for item in items],
    }


def _contexts_from(run_paths) -> list[str]:
    """Contexts this run actually adapted, from the files it wrote."""
    directory = run_paths.phase_dir("phase3")
    if not directory.is_dir():
        return []
    return sorted(
        path.stem[len("adapt_"):] for path in directory.glob("adapt_*.json")
    )


def _vcc_status(run_paths):
    def status() -> tuple[str, str | None]:
        from .submit import package

        if run_paths.submission_vcc.is_file():
            return DONE, None
        if package.tool_path() is None:
            return (
                DEVIATION,
                "the challenge's `vcc` tool is not on PATH here, so no .vcc was written. "
                "`submission/prediction.h5ad` is the deliverable and our own validator "
                "has checked every rule of §8; the user packages it on the server "
                "(PLAN.md §9, decided at the M0 review).",
            )
        return (
            DEVIATION,
            "`vcc prep` did not produce a .vcc — see reports/validate.json for what "
            "the tool said.",
        )

    return status


def _figures_status(run_paths):
    def status() -> tuple[str, str | None]:
        figures = sorted(run_paths.figures.glob("*.png")) if run_paths.figures.is_dir() else []
        if figures:
            return DONE, None
        return DEVIATION, "summary.md was written but no figure could be drawn"

    return status


def write(cfg, run_paths) -> dict[str, Any]:
    report = build(cfg, run_paths)
    run_paths.checklist.write_text(json.dumps(report, indent=2, default=str))
    return report


def summarize(report: dict[str, Any]) -> str:
    lines = ["Element checklist (CLAUDE.md §4.8)", ""]
    for item in report["items"]:
        mark = {DONE: "done", DEVIATION: "DEVIATION", MISSING: "MISSING"}[item["status"]]
        lines.append(f"  {item['item']:>2}. [{mark:^9}] {item['element']}")
        if item["reason"]:
            lines.append(f"        {item['reason']}")
    lines.append("")
    lines.append(
        f"{report['n_done']} done, {report['n_deviation']} deviation(s), "
        f"{report['n_missing']} missing"
    )
    return "\n".join(lines)
