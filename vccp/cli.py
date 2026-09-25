"""The command line: `python -m vccp <stage> --config configs/<name>.yaml`.

`all` runs every stage in order and is resumable — a stage that finished
under the same config is skipped unless `--force`.

Stages are registered here as they are built. A name that is not registered
is rejected with the list of the ones that are, rather than accepted and then
found to do nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import Config, ConfigError, config_to_dict, load_config
from .logging_utils import banner, get_logger, setup_logging
from .paths import RunPaths
from . import provenance
from .runtime import (
    apply_resources,
    effective_thread_env,
    release_gpu_memory,
    resolve_device,
    seed_everything,
    set_determinism,
)

@dataclass(frozen=True)
class StageOptions:
    """Command-line choices a stage may need.

    They are deliberately *not* part of the config: `--allow-warnings` says
    what this invocation accepts, not what the run is, and it must not change
    the config hash that decides which stages are already done.
    """

    allow_warnings: bool = False


StageFn = Callable[..., Any]

CHECK_DATA = "check-data"
PRIORS = "priors"
PHASE1 = "phase1"
PHASE2 = "phase2"
REHEARSAL = "rehearsal"
PHASE3 = "phase3"
FORGETTING = "forgetting"
PREDICT = "predict"
SANITY = "sanity"
VALIDATE = "validate"
PACKAGE = "package"
REPORT = "report"


def _stage_check_data(cfg: Config) -> Any:
    from .data.checks import check_data

    return check_data(cfg)


def _stage_priors(cfg: Config) -> Any:
    from .priors.stage import run_priors

    return run_priors(cfg)


def _stage_phase1(cfg: Config) -> Any:
    from .phases.phase1 import run_phase1

    return run_phase1(cfg)


def _stage_phase2(cfg: Config) -> Any:
    from .phases.phase2 import run_phase2

    return run_phase2(cfg)


def _stage_rehearsal(cfg: Config) -> Any:
    from .rehearsal.stage import run_rehearsal

    return run_rehearsal(cfg)


def _stage_phase3(cfg: Config) -> Any:
    from .phases.phase3 import run_phase3

    return run_phase3(cfg)


def _stage_predict(cfg: Config) -> Any:
    from .predict.run import run_predict

    return run_predict(cfg)


def _stage_forgetting(cfg: Config) -> Any:
    from .phases.forgetting import run_forgetting

    return run_forgetting(cfg)


def _stage_sanity(cfg: Config) -> Any:
    from .sanity.stage import run_sanity

    return run_sanity(cfg)


def _stage_validate(cfg: Config, options: StageOptions) -> Any:
    from .submit.stage import run_validate

    return run_validate(cfg, options)


def _stage_package(cfg: Config, options: StageOptions) -> Any:
    from .submit.stage import run_package

    return run_package(cfg, options)


def _stage_report(cfg: Config) -> Any:
    from .report.summary import build_summary

    return build_summary(cfg)


#: Stage name -> implementation, in the order `all` runs them.
STAGES: dict[str, StageFn] = {
    CHECK_DATA: _stage_check_data,
    PRIORS: _stage_priors,
    PHASE1: _stage_phase1,
    PHASE2: _stage_phase2,
    REHEARSAL: _stage_rehearsal,
    PHASE3: _stage_phase3,
    # After Phase 3, so it scores Phase 1's validation under all three cores
    # — which is what checklist item 7 asks for.
    FORGETTING: _stage_forgetting,
    PREDICT: _stage_predict,
    # §6.1: sanity runs on the predictions *before* the submission is written,
    # so a violation stops the pipeline instead of producing a bad file.
    SANITY: _stage_sanity,
    VALIDATE: _stage_validate,
    PACKAGE: _stage_package,
    REPORT: _stage_report,
}


def stage_names() -> list[str]:
    return list(STAGES)


def config_hash(cfg: Config) -> str:
    """A stable hash of the config, so `all` can tell a finished stage from
    one whose settings have since changed."""
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def newest_mtime(paths) -> float | None:
    """The most recent mtime among `paths`, or None if none exist."""
    stamps = [Path(p).stat().st_mtime for p in paths if Path(p).exists()]
    return max(stamps) if stamps else None


def _prediction_blocks(run_paths: RunPaths) -> float | None:
    return newest_mtime(run_paths.predictions_dir.glob("*/*.npz"))


def _submission(run_paths: RunPaths) -> float | None:
    return newest_mtime([run_paths.submission])


#: Stages whose work is invalidated by something *other* than the config.
#: `sanity` and `validate` read the prediction blocks, and `package` reads the
#: assembled submission; if those have been rewritten since the stage ran, the
#: stage is not done however its marker reads. Without this a `predict` rerun
#: leaves `validate` and `package` silently skipped, and the `.vcc` on disk
#: describes predictions that no longer exist (DECISIONS.md D50).
STAGE_INPUTS: dict[str, Any] = {
    SANITY: _prediction_blocks,
    VALIDATE: _prediction_blocks,
    PACKAGE: _submission,
}


def _input_mtime(run_paths: RunPaths, stage: str) -> float | None:
    probe = STAGE_INPUTS.get(stage)
    return probe(run_paths) if probe else None


def _is_done(run_paths: RunPaths, stage: str, digest: str) -> bool:
    marker = run_paths.stage_marker(stage)
    if not marker.is_file():
        return False
    try:
        recorded = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if recorded.get("config_hash") != digest:
        return False

    current = _input_mtime(run_paths, stage)
    if current is None:
        return True
    previous = recorded.get("input_mtime")
    if previous is None or current > previous:
        get_logger().info(
            "%s ran before its inputs were last written; rerunning it rather than "
            "leaving stale output in place", stage,
        )
        return False
    return True


def _mark_done(run_paths: RunPaths, stage: str, digest: str) -> None:
    run_paths.stage_marker(stage).write_text(
        json.dumps(
            {
                "stage": stage,
                "config_hash": digest,
                "input_mtime": _input_mtime(run_paths, stage),
            },
            indent=2,
        )
    )


def write_run_config(
    cfg: Config, run_paths: RunPaths, *, checks_skipped: bool,
    stamp: dict[str, Any] | None = None,
) -> None:
    """Record exactly what this run was given: the settings, whether the data
    checks were skipped, and the code that produced it."""
    from . import provenance

    payload = config_to_dict(cfg)
    payload["checks_skipped"] = checks_skipped
    payload["config_hash"] = config_hash(cfg)
    payload["provenance"] = stamp or provenance.describe(cfg)
    run_paths.config.write_text(yaml.safe_dump(payload, sort_keys=False))


def _call(stage: str, cfg: Config, options: StageOptions) -> Any:
    """Stages take the config; the few that need the invocation's own choices
    take the options too."""
    import inspect

    function = STAGES[stage]
    if len(inspect.signature(function).parameters) > 1:
        return function(cfg, options)
    return function(cfg)


def run_stage(
    stage: str, cfg: Config, run_paths: RunPaths, *, force: bool,
    options: StageOptions | None = None,
) -> Any:
    log = get_logger()
    digest = config_hash(cfg)
    if not force and _is_done(run_paths, stage, digest):
        log.info("skipping %s (already done under this config; --force to rerun)", stage)
        return None

    banner(f"stage: {stage}")
    result = _call(stage, cfg, options or StageOptions())
    _mark_done(run_paths, stage, digest)
    release_gpu_memory()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vccp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage",
        choices=[*stage_names(), "all"],
        help="stage to run, or 'all' for every stage in order",
    )
    parser.add_argument("--config", required=True, type=Path, help="path to a YAML config")
    parser.add_argument("--run-name", default=None, help="override run_name from the config")
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="override seed from the config. Repeats of an ablation differ only in "
        "this: within a run every arm starts from the same weights, so the seed is "
        "what one repeat is a draw from (DECISIONS.md D84).",
    )
    parser.add_argument(
        "--force", action="store_true", help="rerun stages that have already finished"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="skip check-data in 'all' (NOT recommended: it is what catches a bad "
        "input layout before hours of training)",
    )
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="accept a submission the sanity checks warned about (§6.1). Warnings are "
        "expected on mini_data; on the real data, read sanity/summary.txt first.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.run_name:
        cfg = dataclasses.replace(cfg, run_name=args.run_name)
        cfg.validate()

    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
        cfg.validate()

    run_paths = RunPaths(cfg)
    run_paths.ensure()
    log = setup_logging(run_paths.log, verbose=args.verbose)

    log.info("vccp — run %s", cfg.run_name)
    log.info("config        : %s", cfg.source)
    log.info("data_root     : %s", cfg.data_root)
    log.info("vcc_root      : %s", cfg.vcc_root)
    log.info("output        : %s", run_paths.root)
    log.info("seed          : %d", cfg.seed)

    device = resolve_device(cfg.device)
    apply_resources(device, cfg.resources.cpu_threads, cfg.resources.gpu_memory_fraction)
    seed_everything(cfg.seed)
    determinism = set_determinism(cfg.train.deterministic, device)

    stamp = provenance.describe(cfg)
    log.info("code          : %s", provenance.one_line(stamp))

    log.info("device        : %s (config: %s)", device, cfg.device)
    log.info(
        "resources     : cpu_threads=%d num_workers=%d gpu_memory_fraction=%.2f max_ram_gb=%.0f",
        cfg.resources.cpu_threads,
        cfg.resources.num_workers,
        cfg.resources.gpu_memory_fraction,
        cfg.resources.max_ram_gb,
    )
    log.info("thread env    : %s", _format_env(effective_thread_env()))
    log.info(
        "determinism   : %s",
        "on (tf32 off, deterministic kernels)" if determinism["deterministic"]
        else "OFF — two runs of the same checkpoint may differ (train.deterministic)",
    )

    stages = stage_names() if args.stage == "all" else [args.stage]

    if args.skip_check and CHECK_DATA in stages:
        stages = [s for s in stages if s != CHECK_DATA]
        log.warning("")
        log.warning("!" * 72)
        log.warning("--skip-check: the input data checks were SKIPPED for this run.")
        log.warning("Nothing has verified that the files, columns and gene orders are")
        log.warning("what the pipeline expects. Results from this run are not trustworthy")
        log.warning("until `python -m vccp check-data` has passed on this data.")
        log.warning("!" * 72)
        log.warning("")

    write_run_config(cfg, run_paths, checks_skipped=args.skip_check, stamp=stamp)

    options = StageOptions(allow_warnings=args.allow_warnings)
    try:
        for stage in stages:
            run_stage(stage, cfg, run_paths, force=args.force, options=options)
    except Exception as exc:  # noqa: BLE001 — the CLI is the boundary
        log.error("%s", exc)
        return 1

    log.info("")
    log.info("done: %s", ", ".join(stages) if stages else "(nothing to do)")
    return 0


def _format_env(env: dict[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in env.items())
