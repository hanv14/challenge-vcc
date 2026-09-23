"""The `validate` and `package` stages (CLAUDE.md §8).

`validate` assembles the one submission file from the prediction blocks and
then checks every rule of §8 against the file itself — not against the
in-memory objects that produced it. It refuses to run at all if the sanity
stage found a **fail**, and refuses the submission if the sanity stage only
warned, unless it is run with `--allow-warnings` (§6.1).

`package` runs the challenge's own `vcc prep` where that tool is on `PATH`.
It is not on `PATH` in the cloud, and the user packages the file themselves
on the server (PLAN.md §9), so its absence is reported, never fatal.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import Config
from ..logging_utils import get_logger
from ..paths import DataPaths, RunPaths
from . import package as package_mod
from . import validator
from .spec import load_spec
from .writer import write_submission


def _sanity_gate(cfg: Config, run_paths: RunPaths, allow_warnings: bool) -> dict[str, Any]:
    """§6.1: `validate` refuses a submission the sanity checks flagged."""
    if not run_paths.sanity_report.is_file():
        raise RuntimeError(
            f"{run_paths.sanity_report} is missing: run the `sanity` stage before "
            "`validate`, so that a broken prediction is caught before a submission "
            "is written"
        )
    report = json.loads(run_paths.sanity_report.read_text())
    if not report.get("ok"):
        raise RuntimeError(
            "the sanity checks failed, so no submission is written "
            f"(see {run_paths.sanity_summary})"
        )
    accepted = allow_warnings or cfg.sanity.accept_warnings
    if not report.get("clean") and not accepted:
        warned = [c["name"] for c in report.get("checks", []) if c["status"] == "warn"]
        raise RuntimeError(
            f"the sanity checks warned on {warned}; `validate` refuses the submission. "
            "Read sanity/summary.txt, then rerun with --allow-warnings to accept them "
            "(or set sanity.accept_warnings in the config, as mini.yaml does — on "
            "mini_data warnings are expected, CLAUDE.md §6.1)."
        )
    return report


def run_validate(cfg: Config, options=None) -> dict[str, Any]:
    """Assemble the submission, then check it against §8 (item 13)."""
    log = get_logger()
    allow_warnings = bool(getattr(options, "allow_warnings", False))
    run_paths = RunPaths(cfg)
    run_paths.ensure()
    paths = DataPaths(cfg)

    sanity = _sanity_gate(cfg, run_paths, allow_warnings)
    spec = load_spec(cfg)

    write = write_submission(cfg, spec, run_paths)
    run_paths.submission_report.write_text(json.dumps(write.as_dict(), indent=2))

    result = validator.validate_file(cfg, spec, run_paths.submission)
    on_mini = cfg.on_mini_data
    dry = package_mod.dry_run(cfg, paths, run_paths.submission, warning_only=on_mini)

    report = {
        **result.as_dict(),
        "written": write.as_dict(),
        "vcc_prep": dry.as_dict(),
        "sanity": {
            "ok": sanity.get("ok"),
            "clean": sanity.get("clean"),
            "n_warn": sanity.get("n_warn"),
            "accepted_with_warnings": (
                (allow_warnings or cfg.sanity.accept_warnings) and not sanity.get("clean")
            ),
            "accepted_by": (
                "--allow-warnings" if allow_warnings
                else "sanity.accept_warnings in the config" if cfg.sanity.accept_warnings
                else None
            ),
        },
        "note": "our own validator is the gate; `vcc prep` is the challenge's own "
        "check and runs only where the tool is installed. On mini_data the tool may "
        "refuse a file that is deliberately smaller than the real round, which is a "
        "warning there and never a pass condition (§8).",
    }
    run_paths.validate_report.write_text(json.dumps(report, indent=2, default=str))
    log.info("\n%s", result.summary())
    if dry.ran and not dry.ok and not on_mini:
        log.error("`vcc prep --dry-run` refused the file; see reports/validate.json")
    if not result.ok:
        raise RuntimeError(
            f"the submission violates {len(result.failures)} rule(s) of §8; "
            f"see {run_paths.validate_report}"
        )
    log.info("submission validated -> %s", run_paths.validate_report)
    return report


def run_package(cfg: Config, options=None) -> dict[str, Any]:
    """`vcc prep`, where the challenge's tool exists (item 13)."""
    log = get_logger()
    run_paths = RunPaths(cfg)
    paths = DataPaths(cfg)
    if not run_paths.submission.is_file():
        raise RuntimeError(
            f"{run_paths.submission} does not exist: run the `validate` stage first"
        )

    on_mini = cfg.on_mini_data
    result = package_mod.package(
        cfg, paths, run_paths.submission, run_paths.submission_vcc, warning_only=on_mini
    )
    if not result.ran:
        log.info("package: %s", result.skipped_reason)
    elif result.ok:
        log.info("packaged -> %s", run_paths.submission_vcc)
    else:
        (log.warning if on_mini else log.error)(
            "`vcc prep` did not write a .vcc; prediction.h5ad remains the deliverable"
        )

    report = {
        **result.as_dict(),
        "submission": str(run_paths.submission),
        "vcc": str(run_paths.submission_vcc) if run_paths.submission_vcc.is_file() else None,
    }
    (run_paths.submission_dir / "package.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    return report
