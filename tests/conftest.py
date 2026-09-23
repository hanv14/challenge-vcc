"""Shared fixtures. Every test runs on `mini_data`, which is read-only input.

Tests that need to break an input file work on a *mirror*: a tmp directory of
symlinks pointing at `mini_data`. Removing or overwriting a link changes only
the mirror, so `mini_data/` itself is never modified (CLAUDE.md §1.1).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from vccp.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
MINI_CONFIG = REPO_ROOT / "configs" / "mini.yaml"
SERVER_CONFIG = REPO_ROOT / "configs" / "server.yaml"


def mirror_tree(src: Path, dst: Path) -> None:
    """Recreate `src`'s directory structure in `dst`, symlinking the files."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        target = dst / entry.name
        if entry.is_dir():
            mirror_tree(entry, target)
        else:
            target.symlink_to(entry.resolve())


def _assert_is_mirror_link(mirror_path: Path) -> None:
    """Guard: only ever touch the mirror's own symlink.

    `Path.resolve()` on a mirror path follows the link into `mini_data`, so a
    test that resolves before unlinking would delete the real input. This
    refuses anything that is not a link in the mirror.
    """
    if not mirror_path.is_symlink():
        raise AssertionError(
            f"{mirror_path} is not a symlink in the mirror — refusing to touch it, "
            "because it may be the real mini_data file"
        )


def unlink_in_mirror(mirror_path: Path) -> None:
    """Remove a file from the mirror, leaving `mini_data` untouched."""
    _assert_is_mirror_link(mirror_path)
    mirror_path.unlink()


def replace_in_mirror(mirror_path: Path, write) -> None:
    """Swap a symlink in the mirror for a real file written by `write(path)`."""
    _assert_is_mirror_link(mirror_path)
    mirror_path.unlink()
    write(mirror_path)


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="session", autouse=True)
def mini_data_is_read_only() -> None:
    """Fail the session if anything wrote to `mini_data` (CLAUDE.md §1.1)."""
    data_root = load_config(MINI_CONFIG).data_root
    before = _snapshot(data_root)
    yield
    after = _snapshot(data_root)
    if before != after:
        changed = sorted(
            set(before) ^ set(after)
            | {k for k in set(before) & set(after) if before[k] != after[k]}
        )
        raise AssertionError(f"mini_data was modified by the tests: {changed}")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def mini_cfg(tmp_path: Path) -> Config:
    """The mini config, writing its outputs into a tmp run directory."""
    cfg = load_config(MINI_CONFIG)
    return dataclasses.replace(cfg, output_root=tmp_path / "runs")


@pytest.fixture(scope="session")
def session_cfg(tmp_path_factory) -> Config:
    """A mini config whose run directory survives the whole session.

    Building the priors takes a few seconds, so the stages that need them
    share one build rather than repeating it per test.
    """
    cfg = load_config(MINI_CONFIG)
    return dataclasses.replace(cfg, output_root=tmp_path_factory.mktemp("runs"))


@pytest.fixture(scope="session")
def built_priors(session_cfg):
    """The full-scope priors, built once."""
    from vccp.priors.build import build_priors

    return build_priors(session_cfg)


@pytest.fixture(scope="session")
def pipeline_cfg(tmp_path_factory) -> Config:
    """The mini config at budgets that exercise the wiring, not the training.

    Every stage runs for real; only the step counts are cut. One run is shared
    by every test that needs a finished pipeline, because running the stages
    is the slowest thing in the suite by far.
    """
    cfg = load_config(MINI_CONFIG)
    return dataclasses.replace(
        cfg,
        output_root=tmp_path_factory.mktemp("pipeline"),
        phase1=dataclasses.replace(cfg.phase1, steps=1, batch_size=2),
        phase2=dataclasses.replace(cfg.phase2, steps=1, batch_size=2, cells_per_draw=2),
        train=dataclasses.replace(
            cfg.train, output_genes_per_step=8, log_every=1, core_freeze_ablation=False
        ),
        rehearsal=dataclasses.replace(
            cfg.rehearsal, max_targets=2, adapt_steps=1, phase2_steps=1,
            calibration_thresholds=(0.0, 0.25), calibration_scales=(0.5, 1.0),
        ),
        phase3=dataclasses.replace(cfg.phase3, adapt_steps=1, batch_size=2),
        # The leaderboard scale runs the scorer many more times; it is tested
        # on its own in test_official.py.
        eval=dataclasses.replace(cfg.eval, build_scale=False),
    )


@pytest.fixture(scope="session")
def pipeline_run(pipeline_cfg):
    """Every stage, run once, in order — the run the M4/M5 tests read."""
    from vccp import cli
    from vccp.paths import RunPaths

    options = cli.StageOptions(allow_warnings=True)
    for stage in cli.stage_names():
        cli._call(stage, pipeline_cfg, options)
    return RunPaths(pipeline_cfg)


@pytest.fixture
def mirrored_cfg(tmp_path: Path) -> Config:
    """Like `mini_cfg`, but reading a symlink mirror the test may break."""
    cfg = load_config(MINI_CONFIG)
    data_root = tmp_path / "data"
    mirror_tree(cfg.data_root, data_root)
    vcc_root = data_root / cfg.vcc_root.relative_to(cfg.data_root)
    return dataclasses.replace(
        cfg,
        data_root=data_root,
        vcc_root=vcc_root,
        output_root=tmp_path / "runs",
    )
