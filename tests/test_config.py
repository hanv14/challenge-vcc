"""The config schema: defaults, validation, and the two shipped configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vccp.config import Config, ConfigError, Resources, config_to_dict, load_config
from vccp.runtime import resolve_device

from tests.conftest import MINI_CONFIG, SERVER_CONFIG


def test_mini_config_loads_with_cpu_device():
    cfg = load_config(MINI_CONFIG)
    # Decided at the M0 review: mini is explicit rather than relying on
    # `auto` happening to resolve to CPU in the container (PLAN.md §8.8).
    assert cfg.device == "cpu"
    assert resolve_device(cfg.device) == "cpu"
    assert cfg.data_root.is_dir()
    assert cfg.vcc_root.is_dir()


def test_server_config_loads_with_auto_device():
    cfg = load_config(SERVER_CONFIG)
    assert cfg.device == "auto"
    # The server's paths do not exist here, and that is not an error (§1.1).
    assert cfg.data_root.is_absolute()
    assert cfg.output_root.is_absolute()


@pytest.mark.parametrize("path", [MINI_CONFIG, SERVER_CONFIG])
def test_both_configs_carry_the_resources_block(path):
    cfg = load_config(path)
    assert cfg.resources == Resources(
        cpu_threads=4, num_workers=2, gpu_memory_fraction=0.5, max_ram_gb=64.0
    )


def test_relative_paths_resolve_against_the_repo_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(MINI_CONFIG)
    assert cfg.data_root.is_dir(), "mini_data must be found from any working directory"


def test_unknown_top_level_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {"data_root": "mini_data", "vcc_root": "mini_data/vcc", "output_root": "runs",
             "devise": "cpu"}
        )
    )
    with pytest.raises(ConfigError, match="devise"):
        load_config(path)


def test_unknown_resources_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {"data_root": "mini_data", "vcc_root": "mini_data/vcc", "output_root": "runs",
             "resources": {"cpu_thread": 4}}
        )
    )
    with pytest.raises(ConfigError, match="cpu_thread"):
        load_config(path)


def test_missing_required_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"data_root": "mini_data", "output_root": "runs"}))
    with pytest.raises(ConfigError, match="vcc_root"):
        load_config(path)


def test_invalid_values_are_rejected():
    with pytest.raises(ConfigError, match="device"):
        Config(data_root=MINI_CONFIG, vcc_root=MINI_CONFIG, output_root=MINI_CONFIG,
               device="gpu").validate()
    with pytest.raises(ConfigError, match="gpu_memory_fraction"):
        Resources(gpu_memory_fraction=1.5).validate()
    with pytest.raises(ConfigError, match="cpu_threads"):
        Resources(cpu_threads=0).validate()


def test_config_to_dict_is_serializable(mini_cfg):
    payload = config_to_dict(mini_cfg)
    yaml.safe_dump(payload)  # raises if a Path leaked through
    assert payload["resources"]["cpu_threads"] == 4


def test_the_mini_caveat_follows_the_data_root(mini_cfg, tmp_path):
    """A report that says "these numbers are not indicative" on real server
    data is worse than no caveat: it is shown at a defense."""
    import dataclasses

    assert mini_cfg.on_mini_data is True

    for real in ("/data/han/projects/VCC/data", tmp_path / "VCC" / "data"):
        cfg = dataclasses.replace(mini_cfg, data_root=Path(real))
        assert cfg.on_mini_data is False, real


def test_the_caveat_matches_a_component_not_a_substring(mini_cfg):
    """A path that merely contains the letters is not the mini data."""
    import dataclasses

    for lookalike in ("/data/minimal/VCC", "/srv/mini_data_backup/VCC", "/x/administrative"):
        cfg = dataclasses.replace(mini_cfg, data_root=Path(lookalike))
        assert cfg.on_mini_data is False, lookalike

    cfg = dataclasses.replace(mini_cfg, data_root=Path("/somewhere/mini_data"))
    assert cfg.on_mini_data is True


# --------------------------------------------------------------------------- #
# YAML does not read `3e-4` as a number
# --------------------------------------------------------------------------- #
def _config_with(tmp_path, extra: str, repo_root):
    """`mini.yaml` with its `train:` block replaced by `extra`."""
    import re

    base = (repo_root / "configs" / "mini.yaml").read_text()
    base = re.sub(r"^train:\n(  .*\n)+", "", base, count=1, flags=re.M)
    path = tmp_path / "probe.yaml"
    path.write_text(base + extra)
    return load_config(path, repo_root=repo_root)


def test_scientific_notation_without_a_decimal_point_is_still_a_number(tmp_path, repo_root):
    """YAML 1.1 needs a decimal point and a signed exponent, so `3e-4` parses
    as the *string* `'3e-4'` and reaches the field's own validate, where it
    fails several frames from the config that caused it.

    Almost every knob worth tuning by hand is a small float, so this is a trap
    the config is designed to walk into.
    """
    import yaml

    assert yaml.safe_load("x: 3e-4")["x"] == "3e-4", "the trap, stated"
    assert yaml.safe_load("x: 3.0e-4")["x"] == 0.0003

    cfg = _config_with(tmp_path, "train:\n  lr: 3e-4\n  l2sp_weight: 1E-3\n", repo_root)
    assert cfg.train.lr == pytest.approx(0.0003)
    assert isinstance(cfg.train.lr, float)
    assert cfg.train.l2sp_weight == pytest.approx(0.001)


def test_a_float_field_given_a_whole_number_becomes_a_float(tmp_path, repo_root):
    cfg = _config_with(tmp_path, "train:\n  lr: 1\n", repo_root)
    assert isinstance(cfg.train.lr, float) and cfg.train.lr == 1.0


def test_a_count_must_be_whole(tmp_path, repo_root):
    """`steps: 1.5` would otherwise travel into `range()`, which truncates."""
    with pytest.raises(ConfigError, match="whole number"):
        _config_with(tmp_path, "phase2:\n  steps: 1.5\n", repo_root)

    cfg = _config_with(tmp_path, "phase2:\n  steps: 200.0\n", repo_root)
    assert cfg.phase2.steps == 200 and isinstance(cfg.phase2.steps, int)


def test_a_value_that_is_not_a_number_names_its_key(tmp_path, repo_root):
    with pytest.raises(ConfigError, match=r"train\.lr must be a number, got 'abc'"):
        _config_with(tmp_path, "train:\n  lr: abc\n", repo_root)


def test_a_tuple_field_coerces_element_by_element(tmp_path, repo_root):
    cfg = _config_with(
        tmp_path, "rehearsal:\n  calibration_scales: [2.5e-1, 1]\n", repo_root
    )
    assert cfg.rehearsal.calibration_scales == (0.25, 1.0)


def test_booleans_are_left_alone(tmp_path, repo_root):
    """`bool` is a subclass of `int`, so a careless coercion turns True into 1."""
    cfg = _config_with(tmp_path, "train:\n  deterministic: true\n", repo_root)
    assert cfg.train.deterministic is True


def test_checkpoint_selection_is_checked():
    """A typo here would silently ship the wrong weights (D85)."""
    from vccp.config import Train

    Train(checkpoint_selection="final").validate()
    with pytest.raises(ConfigError, match="checkpoint_selection"):
        Train(checkpoint_selection="lowest").validate()


def test_every_config_in_the_repo_loads(repo_root):
    """The `3e-4` trap (D75) reached the server as a TypeError six frames
    from the config that caused it, because nothing here ever opened the
    file. Each of these is a config someone is told to run."""
    configs = sorted((repo_root / "configs").glob("*.yaml"))
    assert configs, "no configs found"
    for path in configs:
        load_config(path).validate()
