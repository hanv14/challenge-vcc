"""The README and the environment check.

The README is what the user follows on a machine this build never saw, so
these keep it honest: every stage it documents exists, every stage that
exists is documented, and the commands it prints are the ones the code
actually accepts.
"""

from __future__ import annotations

import re

import pytest

from vccp import cli
from vccp.envcheck import FAIL, OK, WARN, check_environment


@pytest.fixture(scope="module")
def readme(repo_root):
    path = repo_root / "README.md"
    assert path.is_file(), "the hand-off needs a README (CLAUDE.md §9)"
    return path.read_text()


def test_every_stage_is_documented(readme):
    for stage in cli.stage_names():
        assert f"`{stage}`" in readme, f"the README does not mention the {stage!r} stage"


def test_no_stage_is_invented(readme):
    """A documented command that does not exist is worse than none."""
    documented = set(re.findall(r"python -m vccp ([a-z-]+) --config", readme))
    known = set(cli.stage_names()) | {"all"}
    assert documented <= known, documented - known


def test_the_readme_covers_the_hand_off(readme):
    """CLAUDE.md §9's definition of done, item by item."""
    for phrase in (
        "source scripts/server_env.sh",
        "scripts/check_env.py",
        "cell-eval2",
        "vcc prep",
        "--dry-run",
        "nice -n 10 ionice -c3 python -m vccp all --config configs/server.yaml",
        "sanity/summary.txt",
        "rehearsal/summary.txt",
        "checklist.json",
        "vcc_root",
        "pert_counts.csv",
    ):
        assert phrase in readme, f"the README does not cover {phrase!r}"


def test_the_readme_does_not_promise_measured_server_timings(readme):
    """The full data was never available here; the README must say so where
    it gives server numbers."""
    assert "estimates" in readme.lower()
    assert "not measurements" in readme.lower()


# --------------------------------------------------------------------------- #
# the environment check
# --------------------------------------------------------------------------- #
def test_the_environment_check_passes_here(mini_cfg):
    report = check_environment(mini_cfg)
    assert report["ok"], [f for f in report["findings"] if f["status"] == FAIL]
    names = {finding["name"] for finding in report["findings"]}
    assert {"cell-eval2", "vcc", "device", "threads", "disk", "round"} <= names
    assert "package:torch" in names


def test_the_environment_check_reports_a_missing_data_root(mini_cfg, tmp_path):
    import dataclasses

    broken = dataclasses.replace(mini_cfg, data_root=tmp_path / "nowhere")
    report = check_environment(broken)
    assert not report["ok"]
    failed = {f["name"] for f in report["findings"] if f["status"] == FAIL}
    assert "path:data_root" in failed


def test_a_missing_challenge_tool_is_a_warning_not_a_failure(mini_cfg):
    """`vcc` does not exist in the cloud, and the run must still complete."""
    report = check_environment(mini_cfg)
    tool = next(f for f in report["findings"] if f["name"] == "vcc")
    assert tool["status"] in (OK, WARN)


def test_the_check_writes_nothing(mini_cfg, tmp_path):
    import dataclasses

    cfg = dataclasses.replace(mini_cfg, output_root=tmp_path / "untouched")
    check_environment(cfg)
    assert not (tmp_path / "untouched").exists()


# --------------------------------------------------------------------------- #
# provenance and determinism
# --------------------------------------------------------------------------- #
def test_the_stamp_names_the_commit_and_whether_it_was_clean(mini_cfg):
    from vccp import provenance

    stamp = provenance.describe(mini_cfg)
    git = stamp["git"]
    assert git["commit"] != provenance.UNKNOWN, "this repository is a git checkout"
    assert isinstance(git["dirty"], bool)
    assert git["short"] == git["commit"][:12]
    assert stamp["packages"]["torch"] != provenance.UNKNOWN
    assert provenance.one_line(stamp).startswith(git["short"])


def test_the_stamp_survives_a_directory_that_is_not_a_checkout(mini_cfg, tmp_path):
    """It must never fail a run — a copied tree, or a machine without git."""
    import dataclasses

    from vccp import provenance

    detached = dataclasses.replace(mini_cfg, repo_root=tmp_path)
    stamp = provenance.describe(detached)
    assert stamp["git"]["commit"] in (provenance.UNKNOWN,) or stamp["git"]["commit"]
    assert stamp["python"]


def test_determinism_is_on_by_default_and_can_be_turned_off(mini_cfg):
    from vccp.runtime import set_determinism

    assert mini_cfg.train.deterministic is True

    from vccp.runtime import CUBLAS_WORKSPACE, CUBLAS_WORKSPACE_VALUE, set_thread_env

    set_thread_env()  # what `python -m vccp` does before importing torch
    on = set_determinism(True, "cpu")
    assert on["deterministic"] is True
    assert on["tf32"] is False
    assert on[CUBLAS_WORKSPACE] == CUBLAS_WORKSPACE_VALUE

    assert set_determinism(False, "cpu") == {"deterministic": False}


def test_determinism_warns_when_the_cublas_workspace_never_got_set(monkeypatch):
    """Claiming determinism while cuBLAS is free to reduce as it likes would
    be worse than claiming nothing."""
    from vccp.runtime import CUBLAS_WORKSPACE, set_determinism

    monkeypatch.delenv(CUBLAS_WORKSPACE, raising=False)
    assert "warning" not in set_determinism(True, "cpu"), "only matters on CUDA"
    assert CUBLAS_WORKSPACE in set_determinism(True, "cuda")["warning"]


def test_the_cublas_workspace_is_set_before_torch_would_read_it():
    """It is read once, when CUDA initializes, so it belongs with the thread
    variables in `__main__` — not with the config."""
    import os

    from vccp.runtime import CUBLAS_WORKSPACE, set_thread_env

    set_thread_env()
    assert os.environ[CUBLAS_WORKSPACE]
