#!/usr/bin/env python
"""Read every `.h5ad` the pipeline reads, and say which one is unreadable.

    python scripts/probe_data.py --config configs/server.yaml

`check-data` verifies files, columns and gene orders — it reads metadata. This
reads the *matrices*: blocks of each `X` and each layer, spread across the
file. It is what to run when a stage fails hours in with an HDF5 error such as

    Can't synchronously read data (filter returned failure during read)

which names no file. This does, with the element and the row range.

Small files are read in full; large ones are sampled (`--samples`,
`--block-rows`), so a pass means "nothing broken was found", not "every byte
is sound". Writes nothing. Exits 1 if any file failed.
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
from vccp.data.h5probe import explain, probe_h5ad  # noqa: E402
from vccp.paths import DataPaths  # noqa: E402


def candidate_files(cfg) -> list[Path]:
    """Every `.h5ad` a run opens, in the order the stages open them."""
    paths = DataPaths(cfg)
    files: list[Path] = [paths.phase1_lincs]

    for _, directory in sorted(paths.discover_phase2_contexts().items()):
        files.append(directory / "pseudobulk.h5ad")
    files.extend(path for _, path in sorted(paths.discover_replogle_singlecell().items()))
    files.extend(path for _, path in sorted(paths.discover_replogle_bulk().items()))
    files.extend(path for _, path in sorted(paths.discover_phase3_controls().items()))
    files.extend(path for _, path in sorted(paths.discover_context_files().items()))
    return [path for path in files if path is not None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=12,
                        help="row blocks sampled from a file too large to read in full")
    parser.add_argument("--block-rows", type=int, default=512)
    parser.add_argument("--full", action="store_true",
                        help="read every file in full, however large (slow)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--only", type=str, default=None,
                        help="probe only files whose path contains this substring")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.full:
        from vccp.data import h5probe

        h5probe.FULL_READ_BYTES = float("inf")

    files = candidate_files(cfg)
    if args.only:
        files = [path for path in files if args.only in str(path)]
    if not files:
        print("no .h5ad files found — is `data_root` right?", file=sys.stderr)
        return 2

    reports = []
    failed = []
    for path in files:
        report = probe_h5ad(path, samples=args.samples, block_rows=args.block_rows)
        reports.append(report)
        if not args.json:
            print(report.summary(), flush=True)
        if not report.ok:
            failed.append(report)

    if args.json:
        print(json.dumps([r.as_dict() for r in reports], indent=2, default=str))
    elif failed:
        print()
        for report in failed:
            print(f"{report.path}\n  {explain(report)}")
    else:
        print(f"\n{len(reports)} file(s) readable.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
