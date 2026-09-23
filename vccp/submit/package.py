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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger

TOOL = "vcc"
#: Long enough for the real file, which `vcc prep` reads end to end.
TIMEOUT_SECONDS = 3600


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "ok": self.ok,
            "command": " ".join(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout[-4000:],
            "stderr": self.stderr[-4000:],
            "skipped_reason": self.skipped_reason,
            "treated_as_warning": self.treated_as_warning,
        }


def tool_path() -> str | None:
    return shutil.which(TOOL)


def _run(command: list[str], *, warning_only: bool) -> ToolResult:
    log = get_logger()
    log.info("  running: %s", " ".join(command))
    try:
        completed = subprocess.run(  # noqa: S603 — the command is built here
            command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ToolResult(
            ran=True, ok=False, command=command, stderr=f"{type(exc).__name__}: {exc}",
            treated_as_warning=warning_only,
        )

    result = ToolResult(
        ran=True,
        ok=completed.returncode == 0,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        treated_as_warning=warning_only,
    )
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
    return _run(
        [
            executable, "prep", str(submission),
            "-g", str(paths.gene_names),
            "--perts", str(paths.pert_counts),
            "-o", str(output),
        ],
        warning_only=warning_only,
    )
