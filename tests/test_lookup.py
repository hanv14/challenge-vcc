"""The lookup: each target's change measured in a Replogle screen (D124).

The table is read from `mini_data`'s screens, so these check its contract
rather than its biology: one row per covered target on the challenge gene
axis, residuals centred on each screen's mean, nothing outside the genes the
screen measures and expresses, the target's own gene left to the knockdown
prior, reliabilities in [0, 1], the same table from the same inputs, and a
cache that rebuilds when its inputs change. Then the prediction stage: the
lookup reaches covered targets at its own scale and leaves the rest alone.
"""

from __future__ import annotations

import dataclasses
import json
import shutil

import numpy as np
import pytest

from vccp.config import ConfigError, load_config
from vccp.data import reference
from vccp.paths import DataPaths, RunPaths
from vccp.predict import generator as generator_mod
from vccp.predict import lookup as lookup_mod
from vccp.predict.generator import GeneratorSettings


def with_lookup(cfg, **changes):
    return dataclasses.replace(cfg, predict=dataclasses.replace(cfg.predict, **changes))


def round_inputs(cfg):
    paths = DataPaths(cfg)
    axis = reference.load_gene_names(paths.gene_names)
    manifest_targets = reference.load_pert_counts(paths.pert_counts, "target_gene").targets
    return paths, list(axis), sorted(manifest_targets)


@pytest.fixture(scope="module")
def table_and_inputs(tmp_path_factory):
    from tests.conftest import MINI_CONFIG

    cfg = load_config(MINI_CONFIG)
    cfg = dataclasses.replace(cfg, output_root=tmp_path_factory.mktemp("lookup"))
    cfg = with_lookup(cfg, lookup_scale=1.0, lookup_max_cells=40, lookup_control_cells=200)
    paths, axis, targets = round_inputs(cfg)
    return cfg, lookup_mod.build_lookup(cfg, paths, targets, axis), paths, axis, targets


def test_one_row_per_covered_target_on_the_challenge_axis(table_and_inputs):
    cfg, table, paths, axis, targets = table_and_inputs
    assert table.targets, "mini_data's K562 screen covers most of the round's targets"
    assert table.residual.shape == (len(table.targets), len(axis))
    assert sorted(table.targets + table.uncovered) == targets
    assert np.all(np.isfinite(table.residual))


def test_residuals_are_centred_on_each_screens_mean(table_and_inputs):
    _, table, _, axis, _ = table_and_inputs
    gene_index = {g: i for i, g in enumerate(axis)}
    for screen in set(table.screen):
        rows = [i for i, s in enumerate(table.screen) if s == screen]
        own = [gene_index[table.targets[i]] for i in rows if table.targets[i] in gene_index]
        mean = table.residual[rows].mean(axis=0)
        # Zeroing each target's own gene moves the mean there, and only there.
        mean[own] = 0.0
        assert np.abs(mean).max() < 1e-4


def test_only_measured_expressed_genes_move_and_never_the_targets_own(table_and_inputs):
    cfg, table, paths, axis, _ = table_and_inputs
    from vccp.data.replogle import load_replogle_context

    measured = np.zeros(len(axis), dtype=bool)
    for path in paths.discover_phase2_contexts().values():
        measured[load_replogle_context(path).challenge_idx] = True
    assert not np.any(table.residual[:, ~measured]), "a gene no screen measures moved"
    gene_index = {g: i for i, g in enumerate(axis)}
    for row, target in enumerate(table.targets):
        if target in gene_index:
            assert table.residual[row, gene_index[target]] == 0.0


def test_reliabilities_are_in_the_unit_interval(table_and_inputs):
    _, table, _, _, _ = table_and_inputs
    assert np.all((table.reliability >= 0) & (table.reliability <= 1))


def test_the_same_inputs_give_the_same_table(table_and_inputs):
    cfg, table, paths, axis, targets = table_and_inputs
    again = lookup_mod.build_lookup(cfg, paths, targets, axis)
    assert again.key == table.key
    np.testing.assert_array_equal(again.residual, table.residual)
    np.testing.assert_array_equal(again.reliability, table.reliability)


def test_the_change_carries_scale_and_weight(table_and_inputs):
    _, table, _, _, _ = table_and_inputs
    target = table.targets[0]
    row = table.residual[0]
    np.testing.assert_allclose(table.change(target, 0.5, "none"), 0.5 * row, rtol=1e-6)
    np.testing.assert_allclose(
        table.change(target, 0.5, "reliability"), 0.5 * table.reliability[0] * row, rtol=1e-6
    )
    assert table.change("NOT_A_TARGET", 0.5, "none") is None


def test_the_cache_is_reused_and_rebuilt_when_its_inputs_change(table_and_inputs):
    cfg, _, paths, axis, targets = table_and_inputs
    run_paths = RunPaths(cfg)
    assert lookup_mod.load_or_build(with_lookup(cfg, lookup_scale=0.0), paths, run_paths,
                                    targets, axis) is None
    first = lookup_mod.load_or_build(cfg, paths, run_paths, targets, axis)
    cache = run_paths.root / "sources" / "lookup.npz"
    stamp = cache.stat().st_mtime_ns
    second = lookup_mod.load_or_build(cfg, paths, run_paths, targets, axis)
    assert cache.stat().st_mtime_ns == stamp, "an unchanged table was rebuilt"
    np.testing.assert_array_equal(first.residual, second.residual)
    third = lookup_mod.load_or_build(with_lookup(cfg, lookup_max_cells=20), paths, run_paths,
                                     targets, axis)
    assert third.key != first.key


def test_an_unknown_screen_is_refused(mini_cfg):
    paths = DataPaths(mini_cfg)
    with pytest.raises(ValueError, match="not among the Replogle screens"):
        lookup_mod.chosen_screens(with_lookup(mini_cfg, lookup_screens=("NO_SUCH",)), paths)


def test_the_config_refuses_nonsense(mini_cfg):
    with pytest.raises(ConfigError):
        with_lookup(mini_cfg, lookup_scale=-1.0).predict.validate()
    with pytest.raises(ConfigError):
        with_lookup(mini_cfg, lookup_weighting="sometimes").predict.validate()


def test_an_addition_reaches_the_cells_at_its_own_size():
    """`effect_scale x model + addition`, whatever the effect scale is."""
    rng = np.random.default_rng(0)
    model, addition = rng.normal(size=50).astype(np.float32), rng.normal(size=50)
    for effect in (0.01, 0.5, 2.0):
        settings = GeneratorSettings(confidence_threshold=0.0, effect_scale=effect)
        applied = generator_mod.apply_settings(
            generator_mod.with_addition(model, addition, settings), settings
        )
        np.testing.assert_allclose(applied, effect * model + addition, rtol=1e-5, atol=1e-5)
    settings = GeneratorSettings(confidence_threshold=0.0, effect_scale=1.0)
    assert generator_mod.with_addition(model, None, settings) is model


def test_prediction_adds_the_lookup_to_covered_targets_only(pipeline_run, pipeline_cfg, tmp_path):
    """One finished run, predicted twice from copies: lookup off and on.

    Both predictions are made here rather than read from the shared run,
    because other tests re-run stages in that directory with settings of
    their own.
    """
    from vccp import cli
    from vccp.checklist import DEVIATION, _generator_status

    def predict(label: str, scale: float) -> RunPaths:
        root = tmp_path / label
        shutil.copytree(
            pipeline_run.root, root / pipeline_cfg.run_name,
            ignore=shutil.ignore_patterns("predictions", "submission", "sources"),
        )
        cfg = with_lookup(
            dataclasses.replace(pipeline_cfg, output_root=root),
            lookup_scale=scale, lookup_max_cells=40, lookup_control_cells=200,
        )
        cli._call("predict", cfg, cli.StageOptions(allow_warnings=True))
        return RunPaths(cfg)

    base_paths, lookup_paths = predict("off", 0.0), predict("on", 0.5)
    base_index = json.loads(base_paths.prediction_index.read_text())
    index = json.loads(lookup_paths.prediction_index.read_text())
    assert base_index["lookup"] is None
    assert index["lookup"]["scale"] == 0.5
    table = lookup_mod.LookupTable.load(lookup_paths.root / "sources" / "lookup.npz")
    for context in index["contexts"]:
        base = np.load(base_paths.fold_changes(context), allow_pickle=True)
        new = np.load(lookup_paths.fold_changes(context), allow_pickle=True)
        targets = [str(t) for t in base["targets"]]
        assert targets == [str(t) for t in new["targets"]]
        for i, target in enumerate(targets):
            moved = not np.array_equal(base["log2_fold_change"][i], new["log2_fold_change"][i])
            if table.covers(target) and np.any(table.residual[table.targets.index(target)]):
                assert moved, f"{context}/{target}: the lookup did not reach a covered target"
            elif not table.covers(target):
                assert not moved, f"{context}/{target}: an uncovered target moved"
    blocks = {(b["context"], b["target_gene"]): b for b in index["blocks"]}
    assert any(b["lookup"]["applied"] for b in blocks.values())
    assert _generator_status(base_paths)()[0] != DEVIATION or base_index["generator"].get(
        "shared_scale", -1.0) >= 0
    status, reason = _generator_status(lookup_paths)()
    assert status == DEVIATION and "D124" in reason


def test_gene_shrinkage_only_ever_shrinks(table_and_inputs):
    """The James–Stein factor is in [0, 1] per gene, and it is what is applied."""
    _, table, _, _, _ = table_and_inputs
    for screen, factor in table.gene_factor.items():
        assert factor.shape == table.residual.shape[1:]
        assert np.all((factor >= 0) & (factor <= 1)), screen
    for row, target in enumerate(table.targets):
        raw = table.change(target, 1.0, "none", "none")
        shrunk = table.change(target, 1.0, "none", "gene")
        np.testing.assert_allclose(
            shrunk, table.residual[row] * table.gene_factor[table.screen[row]], rtol=1e-6
        )
        assert np.count_nonzero(shrunk) <= np.count_nonzero(raw)
        assert np.sqrt(np.mean(shrunk**2)) <= np.sqrt(np.mean(raw**2)) + 1e-7


def test_the_saved_table_reads_back_whole(table_and_inputs, tmp_path):
    _, table, _, _, _ = table_and_inputs
    table.save(tmp_path / "lookup.npz")
    back = lookup_mod.LookupTable.load(tmp_path / "lookup.npz")
    assert back.targets == table.targets and back.key == table.key
    for screen in table.gene_factor:
        np.testing.assert_array_equal(back.gene_factor[screen], table.gene_factor[screen])
    target = table.targets[0]
    np.testing.assert_array_equal(back.change(target, 0.7, "reliability", "gene"),
                                  table.change(target, 0.7, "reliability", "gene"))


# --------------------------------------------------------------------------- #
# per-entry denoisers (D128)
# --------------------------------------------------------------------------- #
def test_each_entry_carries_a_z_where_the_residual_can_move(table_and_inputs):
    _, table, _, _, _ = table_and_inputs
    assert table.z is not None and table.z.shape == table.residual.shape
    assert np.all(np.isfinite(table.z) | (table.residual != 0))
    assert not np.any(table.z[table.residual == 0]), "a z where nothing can move"


def test_snr_denoisers_shrink_and_never_move_a_zero(table_and_inputs):
    _, table, _, _, _ = table_and_inputs
    for target in table.targets:
        raw = table.change(target, 1.0, "none", "none")
        np.testing.assert_array_equal(table.change(target, 1.0, "none", "snr", tau=0.0), raw)
        for kind in ("snr", "snr_hard"):
            out = table.change(target, 1.0, "none", kind, tau=2.0)
            assert np.all(np.abs(out) <= np.abs(raw) + 1e-7), kind
            assert not np.any(out[raw == 0]), kind


def test_a_table_cached_before_z_is_rebuilt_only_when_a_denoiser_needs_it(table_and_inputs):
    cfg, table, paths, axis, targets = table_and_inputs
    cfg = dataclasses.replace(cfg, run_name="old_cache")
    run_paths = RunPaths(cfg)
    cache = run_paths.root / "sources" / "lookup.npz"
    old = dataclasses.replace(table, z=None)
    old.save(cache)
    stamp = cache.stat().st_mtime_ns
    reused = lookup_mod.load_or_build(cfg, paths, run_paths, targets, axis)
    assert reused.z is None and cache.stat().st_mtime_ns == stamp
    rebuilt = lookup_mod.load_or_build(with_lookup(cfg, lookup_shrinkage="snr"), paths,
                                       run_paths, targets, axis)
    assert rebuilt.z is not None
    np.testing.assert_array_equal(rebuilt.residual, table.residual)


def test_the_lowrank_basis_is_built_cached_and_applied(table_and_inputs):
    cfg, table, paths, axis, targets = table_and_inputs
    cfg = with_lookup(dataclasses.replace(cfg, run_name="basis"), lookup_shrinkage="lowrank",
                      lookup_rank=3, lookup_basis_rank_max=8)
    run_paths = RunPaths(cfg)
    first = lookup_mod.load_or_build(cfg, paths, run_paths, targets, axis)
    assert set(first.bases) == set(first.screen)
    files = {s: run_paths.root / "sources" / f"lookup_basis_{s}.npz" for s in first.bases}
    stamps = {s: f.stat().st_mtime_ns for s, f in files.items()}
    second = lookup_mod.load_or_build(cfg, paths, run_paths, targets, axis)
    assert {s: f.stat().st_mtime_ns for s, f in files.items()} == stamps
    for screen, basis in second.bases.items():
        np.testing.assert_array_equal(basis.components, first.bases[screen].components)
        assert basis.components.shape[1] == len(axis)
    target = first.targets[0]
    raw = first.change(target, 1.0, "none", "none")
    low = first.change(target, 1.0, "none", "lowrank", rank=3)
    assert not np.array_equal(low, raw) and not np.any(low[raw == 0])
    both = first.change(target, 1.0, "none", "lowrank_snr", tau=1.0, rank=3)
    assert not np.any(both[raw == 0]) and np.all(np.isfinite(both))


def test_the_config_refuses_a_bad_denoiser(mini_cfg):
    with pytest.raises(ConfigError):
        with_lookup(mini_cfg, lookup_shrinkage="magic").predict.validate()
    with pytest.raises(ConfigError):
        with_lookup(mini_cfg, lookup_snr_tau=-1.0).predict.validate()
    with pytest.raises(ConfigError):
        with_lookup(mini_cfg, lookup_rank=50, lookup_basis_rank_max=20).predict.validate()
    assert mini_cfg.predict.lookup_shrinkage == "none", "the default is raw (D127)"
