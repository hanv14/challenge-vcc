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
