"""Which code produced this run.

A submission is a file on a server; six weeks later the only question that
matters about it is "what made this, exactly?". The config records the
settings, `phase3_policy.json` the generator, `predictions/model/index.json`
what the blocks were built with — and none of them record the **code**.

So every run stamps the git commit, whether the tree was dirty, and the
versions of the packages whose numbers end up in the output. It goes into
`config.yaml`, `checklist.json` and the results summary, so any `.vcc` can be
traced back to the commit that produced it.

Nothing here can fail a run: a repository that is not a git checkout, or a
machine without `git`, reports what it can and says the rest is unknown.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

UNKNOWN = "unknown"
#: Packages whose version changes the numbers, not just the plumbing.
STAMPED_PACKAGES = ("torch", "numpy", "scipy", "anndata", "cell-eval2", "pdex")

#: Seconds before a `git` call is abandoned. Long enough for a cold cache on
#: a network filesystem, short enough that a hung git never holds up a run.
GIT_TIMEOUT_SECONDS = 30  # not-a-size: a timeout in seconds, not a count of cells
#: How many modified file names to record when the tree is dirty. Enough to
#: see what differed; not a transcript of a large refactor.
MAX_MODIFIED_LISTED = 50  # not-a-size: a list cap, not a count of cells


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(  # noqa: S603 — fixed command, no user input
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_state(repo_root: Path | None) -> dict[str, Any]:
    """The commit this run was made from, and whether it was modified."""
    root = Path(repo_root or Path(__file__).resolve().parent.parent)
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": UNKNOWN, "reason": "not a git checkout, or git is unavailable"}

    status = _git(root, "status", "--porcelain")
    dirty = bool(status)
    return {
        "commit": commit,
        "short": commit[:12],
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD") or UNKNOWN,
        "describe": _git(root, "describe", "--always", "--dirty") or UNKNOWN,
        "dirty": dirty,
        # Named, not just counted: a submission built from uncommitted work
        # cannot be reproduced from the commit alone, and the files that
        # differed are what a later reader needs to know.
        "modified_files": sorted(
            line[3:] for line in (status or "").splitlines()
        )[:MAX_MODIFIED_LISTED],
    }


def package_versions() -> dict[str, str]:
    import importlib.metadata as metadata

    versions = {}
    for name in STAMPED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:  # noqa: BLE001 — a missing package is not a failure
            versions[name] = UNKNOWN
    return versions


def describe(cfg) -> dict[str, Any]:
    """Everything needed to tie an artifact to the code that made it."""
    import datetime

    return {
        "git": git_state(cfg.repo_root),
        "packages": package_versions(),
        "python": platform.python_version(),
        "host": platform.node(),
        "user": os.environ.get("USER", UNKNOWN),
        "started_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def one_line(stamp: dict[str, Any]) -> str:
    """The stamp as a log line."""
    git = stamp.get("git", {})
    commit = git.get("short", UNKNOWN)
    dirty = " +uncommitted changes" if git.get("dirty") else ""
    torch_version = stamp.get("packages", {}).get("torch", UNKNOWN)
    return (
        f"{commit}{dirty} on {git.get('branch', UNKNOWN)}, "
        f"python {stamp.get('python')}, torch {torch_version}"
    )
