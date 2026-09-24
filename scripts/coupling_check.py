#!/usr/bin/env python
"""Does the OT coupling carry any information on real cells?

    python scripts/coupling_check.py --config configs/server.yaml

The premise of the per-cell redesign is that a control cell can be matched to
the perturbed cells it most resembles. In high dimension that premise can
fail outright: pairwise distances concentrate, every control cell is nearly
equidistant from every perturbed cell, and the coupling comes out uniform
whatever `epsilon` says — at which point OT is the pooled objective under
another name, and looks exactly like a correctly configured pooled run.

This asks the question before any training is built on the answer. It reads
cells, computes couplings across the `epsilon` sweep and reports, per arm,
how much the barycentric targets actually vary between control cells and
whether the pairing tracks sequencing depth. A control-against-control arm
gives the null.

Needs `processed/phases/phase2/`; trains nothing and writes no checkpoint.
Writes `reports/coupling_check.json`. Exits non-zero on a verdict of
`degenerate` or `pairs-on-noise`.
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
from vccp.diagnostics.coupling import run_coupling_check  # noqa: E402
from vccp.logging_utils import setup_logging  # noqa: E402
from vccp.paths import RunPaths  # noqa: E402
from vccp.runtime import resolve_device, seed_everything, set_determinism  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--screen", default=None, help="which Replogle screen (default: the first)")
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--cells", type=int, default=256, help="cells per batch")
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

    report = run_coupling_check(
        cfg, screen=args.screen, n_targets=args.targets, n_cells=args.cells
    )
    return 0 if report["verdict"] == "informative" else 1


if __name__ == "__main__":
    sys.exit(main())
