"""Calling the challenge's `vcc prep` (D97, D133).

A fake `vcc` on PATH stands in for the real tool, which exists only on the
server: one that finishes, one that never does, one that fails after starting
its file. The file on disk afterwards must be this run's, whole, or absent.
"""

from __future__ import annotations

import dataclasses
import os
from types import SimpleNamespace

import pytest

from vccp.config import ConfigError
from vccp.submit import package as package_mod


def fake_tool(tmp_path, monkeypatch, body: str):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = bin_dir / "vcc"
    tool.write_text("#!/bin/sh\n" + body + "\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return tool


def with_timeout(cfg, minutes):
    return dataclasses.replace(
        cfg, submission=dataclasses.replace(cfg.submission, vcc_timeout_minutes=minutes))


@pytest.fixture
def files(tmp_path):
    submission = tmp_path / "prediction.h5ad"
    submission.write_text("not read by the fake")
    paths = SimpleNamespace(gene_names=tmp_path / "genes.csv", pert_counts=tmp_path / "perts.csv")
    return paths, submission, tmp_path / "out" / "prediction.vcc"


# The fake finds its -o argument: the last one.
LAST_ARG = 'for a in "$@"; do out="$a"; done'


def test_a_tool_that_finishes_writes_the_file_and_reports_its_time(mini_cfg, files, tmp_path,
                                                                    monkeypatch):
    paths, submission, output = files
    fake_tool(tmp_path, monkeypatch, f'{LAST_ARG}\necho packed > "$out"')
    result = package_mod.package(mini_cfg, paths, submission, output, warning_only=False)
    assert result.ok and output.read_text() == "packed\n"
    assert result.seconds is not None and not result.timed_out


def test_a_tool_that_never_finishes_is_stopped_and_its_partial_file_removed(
        mini_cfg, files, tmp_path, monkeypatch):
    paths, submission, output = files
    fake_tool(tmp_path, monkeypatch, f'{LAST_ARG}\necho partial > "$out"\nsleep 30')
    cfg = with_timeout(mini_cfg, 0.02)            # 1.2 s
    result = package_mod.package(cfg, paths, submission, output, warning_only=False)
    assert result.timed_out and not result.ok
    assert package_mod.TIMEOUT_KEY in result.stderr
    assert not output.exists(), "a half-written .vcc was left beside the submission"


def test_a_tool_that_fails_after_starting_its_file_leaves_nothing(mini_cfg, files, tmp_path,
                                                                  monkeypatch):
    paths, submission, output = files
    fake_tool(tmp_path, monkeypatch, f'{LAST_ARG}\necho partial > "$out"\nexit 2')
    result = package_mod.package(mini_cfg, paths, submission, output, warning_only=False)
    assert result.returncode == 2 and not output.exists()


def test_the_tool_cannot_wait_on_a_terminal(mini_cfg, files, tmp_path, monkeypatch):
    """Its output is captured, so a question it asked would be invisible."""
    paths, submission, _ = files
    fake_tool(tmp_path, monkeypatch,
              'if read -r answer; then exit 9; fi\n'
              '[ "$(readlink /proc/self/fd/0)" = /dev/null ] || exit 8\nexit 0')
    result = package_mod.dry_run(mini_cfg, paths, submission, warning_only=False)
    assert result.ok, result.returncode


def test_the_timeout_is_a_config_key(mini_cfg):
    assert package_mod.timeout_seconds(mini_cfg) == mini_cfg.submission.vcc_timeout_minutes * 60
    assert package_mod.timeout_seconds(with_timeout(mini_cfg, 0)) is None
    with pytest.raises(ConfigError):
        with_timeout(mini_cfg, -1).submission.validate()
