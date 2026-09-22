"""The config schema: defaults, validation, and the two shipped configs."""

from __future__ import annotations

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
