#!/usr/bin/env python
"""What could the panel→rest mapping reach, and is the target even there?

    python scripts/mapping_ceiling.py --config configs/server.yaml

Phase 2's mapping sat at `map_control_pearson` 0.016 to 0.061 across the three
screens. (Its `map_control_mse` of 0.965 to 1.144 is not a ratio: the no-change
error of the 32 cells it is scored on varies with the draw, so read
`map_control_ratio_to_no_change` instead, D105.) This bounds the problem before anyone
tries to improve it (PLAN_MAPPING.md §B1), with two numbers and no GPU:

* a **ridge regression** from the panel to the rest genes on control cells, in
  the same control-SD units the model uses. A linear map is a weak model; if
  it does better, the model is the problem, and if it does not, the panel does
  not predict single-cell rest genes at this depth.
* the **target's own reliability**, from a binomial split of each cell's
  counts. Its square root bounds any predictor's correlation (D101).

A third fit on library size alone separates "the panel carries gene-specific
information" from "the mapping is a depth correction".

Reads control cells a block at a time, so the 2-million-cell K562 file is
never held. Trains nothing, writes no checkpoint. Needs
`processed/phases/phase2/`; reads `phase2/metrics.json` from the run directory
when it is there, to put the model's number beside its ceiling. Writes
`reports/mapping_ceiling.json` and exits non-zero on any verdict but
`model-underperforms`, which is the only one with a cheap fix.
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
from vccp.diagnostics.mapping import run_mapping_ceiling  # noqa: E402
from vccp.logging_utils import setup_logging  # noqa: E402
from vccp.paths import RunPaths  # noqa: E402
from vccp.runtime import seed_everything  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--screen", default=None, help="one Replogle screen (default: every one found)"
    )
    parser.add_argument(
        "--fit-cells", type=int, default=20000, help="control cells the ridge is fitted on"
    )
    parser.add_argument(
        "--eval-cells", type=int, default=2048, help="held-out control cells it is scored on"
    )
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

    report = run_mapping_ceiling(
        cfg, screen=args.screen, n_fit=args.fit_cells, n_eval=args.eval_cells
    )
    return 0 if report["verdict"] == "model-underperforms" else 1


if __name__ == "__main__":
    sys.exit(main())
