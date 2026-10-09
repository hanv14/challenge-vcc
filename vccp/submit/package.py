"""The challenge's own `vcc` tool, when it is on `PATH` (CLAUDE.md §8).

`vcc prep` validates the gene set, context labels, perturbation labels, cell
counts and the raw-counts requirement, and writes the `.vcc` the upload
takes. The user installs it on the server; it does not exist in the cloud, so
everything here is conditional on `shutil.which` and skips with a clear
message rather than failing.

On `mini_data` the tool may refuse the file because the example is
deliberately smaller than the real round. That is a **warning** there, never
a pass condition (§8).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger

TOOL = "vcc"
#: The config key that bounds one call of the tool (D133).
TIMEOUT_KEY = "submission.vcc_timeout_minutes"


@dataclass
class ToolResult:
    """What the challenge's tool said, or why it was not run."""

    ran: bool
    ok: bool = False
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    skipped_reason: str | None = None
    treated_as_warning: bool = False
    seconds: float | None = None
    timed_out: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "ok": self.ok,
            "seconds": self.seconds,
            "timed_out": self.timed_out,
            "command": " ".join(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout[-4000:],
            "stderr": self.stderr[-4000:],
            "skipped_reason": self.skipped_reason,
            "treated_as_warning": self.treated_as_warning,
        }


def tool_path() -> str | None:
    return shutil.which(TOOL)


def timeout_seconds(cfg) -> float | None:
    """`submission.vcc_timeout_minutes` in seconds; None when it is 0 (no limit)."""
    minutes = cfg.submission.vcc_timeout_minutes
    return None if minutes <= 0 else float(minutes) * 60.0


def _run(command: list[str], *, warning_only: bool, timeout: float | None) -> ToolResult:
    """One call of the tool. Its stdin is closed: a question it might ask can
    never wait on a terminal nobody is watching, as its stdout is captured."""
    log = get_logger()
    log.info("  running: %s", " ".join(command))
    started = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603 — the command is built here
            command, capture_output=True, text=True, timeout=timeout, check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        seconds = time.monotonic() - started
        message = (
            f"{TOOL} did not finish within {seconds / 60:.0f} min and was stopped; "
            f"raise {TIMEOUT_KEY} (0 = no limit) or run the command by hand"
        )
        (log.warning if warning_only else log.error)("  %s", message)
        return ToolResult(
            ran=True, ok=False, command=command, stderr=message,
            treated_as_warning=warning_only, seconds=seconds, timed_out=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ToolResult(
            ran=True, ok=False, command=command, stderr=f"{type(exc).__name__}: {exc}",
            treated_as_warning=warning_only, seconds=time.monotonic() - started,
        )

    result = ToolResult(
        ran=True,
        ok=completed.returncode == 0,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        treated_as_warning=warning_only,
        seconds=time.monotonic() - started,
    )
    log.info("  %s %s in %.1f min", TOOL, "finished" if result.ok else "failed",
             result.seconds / 60)
    if not result.ok:
        (log.warning if warning_only else log.error)(
            "  %s exited %d: %s", TOOL, completed.returncode,
            (completed.stderr or completed.stdout).strip().splitlines()[-1:] or "",
        )
    return result


def dry_run(cfg, paths, submission: Path, *, warning_only: bool) -> ToolResult:
    """`vcc prep --dry-run`: the tool's own checks, writing nothing."""
    executable = tool_path()
    if executable is None:
        return ToolResult(
            ran=False,
            skipped_reason=f"the challenge's {TOOL!r} tool is not on PATH; our own "
            "`validate` stage has already checked every rule of §8. Install it on the "
            "server and rerun `validate` to add the tool's own check.",
        )
    return _run(
        [
            executable, "prep", str(submission),
            "-g", str(paths.gene_names),
            "--perts", str(paths.pert_counts),
            "--dry-run",
        ],
        warning_only=warning_only,
        timeout=timeout_seconds(cfg),
    )


def package(cfg, paths, submission: Path, output: Path, *, warning_only: bool) -> ToolResult:
    """`vcc prep`: write the `.vcc` the upload takes."""
    executable = tool_path()
    if executable is None:
        return ToolResult(
            ran=False,
            skipped_reason=f"the challenge's {TOOL!r} tool is not on PATH, so no .vcc was "
            "written. `submission/prediction.h5ad` is the deliverable; the README gives "
            "the exact `vcc prep` command to run on the server.",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        # `vcc prep` refuses to overwrite, so a second run of this stage in a
        # run directory that already has a .vcc fails with "Output already
        # exists" and leaves the *previous* run's file sitting there — which
        # the checklist then read as item 13 done, for a file that does not
        # match the prediction.h5ad beside it (DECISIONS.md D97). Removing it
        # first means the file on disk is always this run's, or absent.
        log = get_logger()
        log.info("  replacing the .vcc left by an earlier run: %s", output)
        output.unlink()
    result = _run(
        [
            executable, "prep", str(submission),
            "-g", str(paths.gene_names),
            "--perts", str(paths.pert_counts),
            "-o", str(output),
        ],
        warning_only=warning_only,
        timeout=timeout_seconds(cfg),
    )
    if not result.ok and output.exists():
        # A tool stopped mid-write, or one that failed after starting the
        # file, leaves a .vcc that is not this submission. The file on disk is
        # this run's, whole, or absent (D97, D133).
        get_logger().info("  removing the incomplete .vcc %s left by the failed call", output)
        output.unlink()
    return result
