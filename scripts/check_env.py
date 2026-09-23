#!/usr/bin/env python
"""Is this machine set up to run the pipeline?

    python scripts/check_env.py --config configs/server.yaml

Run it once on the server before the first `python -m vccp all`. It checks
the packages, `cell-eval2` and its DE backend, the challenge's `vcc` tool,
the device, the thread limits of CLAUDE.md §1.2, the configured data roots
and the free disk under `output_root`. It writes nothing.

Exits 0 if the pipeline can run (warnings are printed but do not fail), and
1 if something would stop it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vccp.runtime import set_thread_env  # noqa: E402

set_thread_env()

from vccp.config import ConfigError, load_config  # noqa: E402
from vccp.envcheck import check_environment  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    report = check_environment(cfg)
    print(json.dumps(report, indent=2, default=str) if args.json else report["summary"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
