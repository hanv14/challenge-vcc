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
from pathlib import Path
from typing import Any

import yaml

from .config import Config, ConfigError, config_to_dict, load_config
from .logging_utils import banner, get_logger, setup_logging
from .paths import RunPaths
from .runtime import (
    apply_resources,
    effective_thread_env,
    release_gpu_memory,
    resolve_device,
    seed_everything,
)

StageFn = Callable[[Config], Any]

CHECK_DATA = "check-data"
PRIORS = "priors"
PHASE1 = "phase1"
PHASE2 = "phase2"
FORGETTING = "forgetting"


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


def _stage_forgetting(cfg: Config) -> Any:
    from .phases.forgetting import run_forgetting

    return run_forgetting(cfg)


#: Stage name -> implementation, in the order `all` runs them.
STAGES: dict[str, StageFn] = {
    CHECK_DATA: _stage_check_data,
    PRIORS: _stage_priors,
    PHASE1: _stage_phase1,
    PHASE2: _stage_phase2,
    # Scores whatever cores exist, so it is meaningful after Phase 2 and
    # again once Phase 3 lands.
    FORGETTING: _stage_forgetting,
}


def stage_names() -> list[str]:
    return list(STAGES)


def config_hash(cfg: Config) -> str:
    """A stable hash of the config, so `all` can tell a finished stage from
    one whose settings have since changed."""
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _is_done(run_paths: RunPaths, stage: str, digest: str) -> bool:
    marker = run_paths.stage_marker(stage)
    if not marker.is_file():
        return False
    try:
        recorded = json.loads(marker.read_text()).get("config_hash")
    except (json.JSONDecodeError, OSError):
        return False
    return recorded == digest


def _mark_done(run_paths: RunPaths, stage: str, digest: str) -> None:
    run_paths.stage_marker(stage).write_text(
        json.dumps({"stage": stage, "config_hash": digest}, indent=2)
    )


def write_run_config(cfg: Config, run_paths: RunPaths, *, checks_skipped: bool) -> None:
    """Record exactly what this run was given, including whether the data
    checks were skipped — that fact has to survive into the run directory."""
    payload = config_to_dict(cfg)
    payload["checks_skipped"] = checks_skipped
    payload["config_hash"] = config_hash(cfg)
    run_paths.config.write_text(yaml.safe_dump(payload, sort_keys=False))


def run_stage(stage: str, cfg: Config, run_paths: RunPaths, *, force: bool) -> Any:
    log = get_logger()
    digest = config_hash(cfg)
    if not force and _is_done(run_paths, stage, digest):
        log.info("skipping %s (already done under this config; --force to rerun)", stage)
        return None

    banner(f"stage: {stage}")
    result = STAGES[stage](cfg)
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
        "--force", action="store_true", help="rerun stages that have already finished"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="skip check-data in 'all' (NOT recommended: it is what catches a bad "
        "input layout before hours of training)",
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

    log.info("device        : %s (config: %s)", device, cfg.device)
    log.info(
        "resources     : cpu_threads=%d num_workers=%d gpu_memory_fraction=%.2f max_ram_gb=%.0f",
        cfg.resources.cpu_threads,
        cfg.resources.num_workers,
        cfg.resources.gpu_memory_fraction,
        cfg.resources.max_ram_gb,
    )
    log.info("thread env    : %s", _format_env(effective_thread_env()))

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

    write_run_config(cfg, run_paths, checks_skipped=args.skip_check)

    try:
        for stage in stages:
            run_stage(stage, cfg, run_paths, force=args.force)
    except Exception as exc:  # noqa: BLE001 — the CLI is the boundary
        log.error("%s", exc)
        return 1

    log.info("")
    log.info("done: %s", ", ".join(stages) if stages else "(nothing to do)")
    return 0


def _format_env(env: dict[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in env.items())
