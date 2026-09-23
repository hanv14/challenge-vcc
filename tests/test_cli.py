"""The command line: stage selection, resume, `--force`, `--skip-check`."""

from __future__ import annotations

import json

import pytest
import yaml

from vccp import cli
from vccp.paths import RunPaths

from tests.conftest import MINI_CONFIG


@pytest.fixture
def argv(tmp_path):
    """Base arguments pointing the run directory at a tmp path."""
    return ["--config", str(MINI_CONFIG), "--run-name", "t"], tmp_path


def tiny(cfg, tmp_path):
    """The mini config with training budgets cut to the bone.

    These tests are about the CLI's plumbing — stage selection, resume,
    `--force`, `--skip-check` — not about training. Running `all` at the mini
    config's real budgets made them the slowest thing in the suite by an
    order of magnitude, and told us nothing the phase tests do not.
    """
    import dataclasses

    return dataclasses.replace(
        cfg,
        output_root=tmp_path / "runs",
        phase1=dataclasses.replace(cfg.phase1, steps=1, batch_size=2),
        phase2=dataclasses.replace(cfg.phase2, steps=1, batch_size=2, cells_per_draw=2),
        train=dataclasses.replace(
            cfg.train, output_genes_per_step=8, log_every=1, core_freeze_ablation=False
        ),
        rehearsal=dataclasses.replace(
            cfg.rehearsal, max_targets=2, adapt_steps=1, phase2_steps=1,
            calibration_thresholds=(0.0,), calibration_scales=(1.0,),
        ),
        phase3=dataclasses.replace(cfg.phase3, adapt_steps=1, batch_size=2),
        # The leaderboard scale runs the scorer six more times per dataset;
        # what it does is tested where it belongs, not in the CLI's plumbing.
        eval=dataclasses.replace(cfg.eval, build_scale=False),
    )


def run(stage, extra=(), *, tmp_path):
    """Run the CLI with `output_root` redirected, via the run-name override."""
    from vccp.config import load_config

    cfg = tiny(load_config(MINI_CONFIG), tmp_path)
    run_paths = RunPaths(cfg)
    run_paths.ensure()

    # The CLI reads the config file, so redirect by monkeypatching the loader
    # rather than by writing a second config file.
    original = cli.load_config
    cli.load_config = lambda path: cfg  # noqa: ARG005
    try:
        code = cli.main([stage, "--config", str(MINI_CONFIG), *extra])
    finally:
        cli.load_config = original
    return code, run_paths


def test_check_data_stage_succeeds(tmp_path):
    code, run_paths = run("check-data", tmp_path=tmp_path)
    assert code == 0
    assert run_paths.check_data_report.is_file()
    assert run_paths.stage_marker("check-data").is_file()


def test_unknown_stage_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["not-a-stage", "--config", str(MINI_CONFIG)])


def test_all_runs_every_registered_stage(tmp_path):
    """CLAUDE.md §4.8: `all` on mini_data, every artifact present, every
    checklist item `done` or an explicitly reported `deviation`.

    `--allow-warnings` because mini_data's statistics are noisy and §6.1 says
    warnings are expected there; a **fail** would still stop the run.
    """
    code, run_paths = run("all", ["--allow-warnings"], tmp_path=tmp_path)
    assert code == 0
    for stage in cli.stage_names():
        assert run_paths.stage_marker(stage).is_file(), stage

    checklist = json.loads(run_paths.checklist.read_text())
    assert checklist["n_items"] == 15
    assert checklist["n_missing"] == 0, [
        item for item in checklist["items"] if item["status"] == "missing"
    ]
    for item in checklist["items"]:
        assert item["status"] in ("done", "deviation")
        if item["status"] == "deviation":
            assert item["reason"]

    for artifact in (run_paths.submission, run_paths.validate_report,
                     run_paths.sanity_report, run_paths.summary_md,
                     run_paths.rehearsal_report, run_paths.phase3_policy):
        assert artifact.is_file(), artifact


def test_a_finished_stage_is_skipped_on_rerun(tmp_path):
    _, run_paths = run("check-data", tmp_path=tmp_path)
    report = run_paths.check_data_report
    first = report.stat().st_mtime_ns

    code, _ = run("check-data", tmp_path=tmp_path)
    assert code == 0
    assert report.stat().st_mtime_ns == first, "the stage should not have rerun"


def test_force_reruns_a_finished_stage(tmp_path):
    _, run_paths = run("check-data", tmp_path=tmp_path)
    marker = run_paths.stage_marker("check-data")
    marker.write_text(json.dumps({"stage": "check-data", "config_hash": "stale"}))

    code, _ = run("check-data", ["--force"], tmp_path=tmp_path)
    assert code == 0
    assert json.loads(marker.read_text())["config_hash"] != "stale"


def test_a_changed_config_reruns_the_stage(tmp_path):
    """Resume keys on the config hash, so a settings change is not skipped."""
    import dataclasses

    from vccp.config import load_config

    cfg = dataclasses.replace(load_config(MINI_CONFIG), output_root=tmp_path / "runs")
    digest_a = cli.config_hash(cfg)
    digest_b = cli.config_hash(dataclasses.replace(cfg, seed=cfg.seed + 1))
    assert digest_a != digest_b


def test_failure_returns_nonzero_without_a_traceback(tmp_path, mirrored_cfg):
    from vccp.paths import DataPaths

    from tests.conftest import unlink_in_mirror

    unlink_in_mirror(DataPaths(mirrored_cfg).phase1_lincs)

    original = cli.load_config
    cli.load_config = lambda path: mirrored_cfg  # noqa: ARG005
    try:
        code = cli.main(["check-data", "--config", str(MINI_CONFIG)])
    finally:
        cli.load_config = original
    assert code == 1


def test_bad_config_returns_two(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"data_root": "x"}))
    assert cli.main(["check-data", "--config", str(bad)]) == 2


def test_skip_check_warns_and_is_recorded_in_the_run_config(tmp_path, caplog):
    """`--skip-check` must never pass quietly: the run directory has to say
    that nothing verified the inputs (PLAN.md §8.11)."""
    code, run_paths = run("all", ["--skip-check", "--allow-warnings"], tmp_path=tmp_path)
    assert code == 0

    recorded = yaml.safe_load(run_paths.config.read_text())
    assert recorded["checks_skipped"] is True
    assert not run_paths.stage_marker("check-data").is_file()

    log = run_paths.log.read_text()
    assert "SKIPPED" in log
    assert "not trustworthy" in log


def test_without_skip_check_the_run_config_says_so(tmp_path):
    _, run_paths = run("all", ["--allow-warnings"], tmp_path=tmp_path)
    assert yaml.safe_load(run_paths.config.read_text())["checks_skipped"] is False


def test_no_shipped_command_uses_skip_check(repo_root):
    """It must never be part of a command we hand the user.

    Prose describing the flag is fine; a runnable `python -m vccp ...` line
    carrying it is not.
    """
    candidates = [repo_root / "README.md", *(repo_root / "scripts").glob("*")]
    offenders = []
    for path in candidates:
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if "-m vccp" in line and "--skip-check" in line:
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, f"--skip-check appears in a shipped command at {offenders}"


# --------------------------------------------------------------------------- #
# a finished stage whose inputs moved is not finished
# --------------------------------------------------------------------------- #
def test_a_stage_reruns_when_its_inputs_are_rewritten(tmp_path):
    """A `predict` rerun left `validate` and `package` silently skipped, and
    the .vcc on disk described predictions that no longer existed."""
    import time

    from vccp import cli as cli_mod

    code, run_paths = run("all", ["--allow-warnings"], tmp_path=tmp_path)
    assert code == 0
    digest = json.loads(run_paths.stage_marker("validate").read_text())["config_hash"]
    assert cli_mod._is_done(run_paths, "validate", digest) is True

    # Touch a prediction block, as a rerun of `predict` would.
    block = next(run_paths.predictions_dir.glob("*/*.npz"))
    time.sleep(0.01)
    block.touch()

    assert cli_mod._is_done(run_paths, "validate", digest) is False
    assert cli_mod._is_done(run_paths, "sanity", digest) is False
    # `package` reads the submission, which has not moved.
    package_digest = json.loads(run_paths.stage_marker("package").read_text())
    assert cli_mod._is_done(run_paths, "package", package_digest["config_hash"]) is True


def test_a_stage_without_inputs_still_resumes(tmp_path):
    from vccp import cli as cli_mod

    _, run_paths = run("check-data", tmp_path=tmp_path)
    digest = json.loads(run_paths.stage_marker("check-data").read_text())["config_hash"]
    assert cli_mod._is_done(run_paths, "check-data", digest) is True


def test_package_refuses_a_submission_older_than_the_predictions(tmp_path):
    import time

    from vccp.submit.stage import run_package

    code, run_paths = run("all", ["--allow-warnings"], tmp_path=tmp_path)
    assert code == 0

    time.sleep(0.01)
    next(run_paths.predictions_dir.glob("*/*.npz")).touch()

    from vccp.config import load_config

    cfg = tiny(load_config(MINI_CONFIG), tmp_path)
    with pytest.raises(RuntimeError, match="older than the predictions"):
        run_package(cfg, None)


def test_the_run_config_records_the_code_that_made_it(tmp_path):
    _, run_paths = run("check-data", tmp_path=tmp_path)
    recorded = yaml.safe_load(run_paths.config.read_text())["provenance"]

    assert set(recorded) >= {"git", "packages", "python", "started_at"}
    assert "torch" in recorded["packages"]
    assert "commit" in recorded["git"]
