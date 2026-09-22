"""Entry point: `python -m vccp <stage> --config configs/<name>.yaml`.

Thread limits are set here, before anything imports numpy, torch, polars or
scanpy — those libraries fix their pool sizes at import, so a cap applied
afterwards has no effect. Variables already set by the environment (e.g. by
`scripts/server_env.sh`) are never overridden (CLAUDE.md §1.2).
"""

from __future__ import annotations

import sys

from .runtime import set_thread_env

set_thread_env()

from .cli import main  # noqa: E402 — must follow set_thread_env()

if __name__ == "__main__":
    sys.exit(main())
