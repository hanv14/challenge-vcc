#!/usr/bin/env python
"""Can the perturbation module overfit a handful of targets?

    python scripts/capacity_check.py --config configs/server.yaml

Phase 2 reports the module at 1.036 on its *training* targets — no better
than predicting no change, on data it has already seen. This asks whether it
can fit targets it is allowed to memorize: a few targets from one screen, no
validation, no mapping loss, every parameter trainable.

If it cannot fit even these, the problem is capacity, conditioning or
optimisation, and redesigning what it is supervised against would inherit it.
Needs a finished `priors` stage in the run directory; warm-starts from the
Phase 1 core when one is there. Writes `reports/capacity_check.json`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vccp.runtime import set_thread_env  # noqa: E402

set_thread_env()

from vccp.config import ConfigError, load_config  # noqa: E402
from vccp.diagnostics.capacity import run_capacity_check  # noqa: E402
from vccp.logging_utils import setup_logging  # noqa: E402
from vccp.paths import RunPaths  # noqa: E402
from vccp.runtime import seed_everything, set_determinism, resolve_device  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--screen", default=None, help="which Replogle screen (default: the first)")
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.run_name:
        import dataclasses

        cfg = dataclasses.replace(cfg, run_name=args.run_name)

    run_paths = RunPaths(cfg)
    run_paths.ensure()
    setup_logging(run_paths.log)
    seed_everything(cfg.seed)
    set_determinism(cfg.train.deterministic, resolve_device(cfg.device))

    report = run_capacity_check(
        cfg, screen=args.screen, n_targets=args.targets, steps=args.steps,
        log_every=args.log_every, lr=args.lr,
    )
    return 0 if report["verdict"] == "can-fit" else 1


if __name__ == "__main__":
    sys.exit(main())
