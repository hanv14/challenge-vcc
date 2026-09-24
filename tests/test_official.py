"""The official scorer, wrapped, and the leaderboard scale (CLAUDE.md §6).

The metrics themselves are `cell-eval2`'s and are never reimplemented here, so
what these test is the wrapper's contract: that all six come back on a tiny
prediction, that everything the preset leaves loose is pinned, and that the
submission's missing control rows are supplied for scoring without ever being
written into a prediction.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from vccp.eval import official, scale

pytestmark = pytest.mark.skipif(
    not official.available(), reason="cell-eval2 is not installed in this environment"
)

PERT_COL = "target_gene"
CONTROL = "non-targeting"
N_GENES = 60


def make_adata(specs, rng):
    """Counts with a real, reproducible effect per perturbation."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    blocks, labels = [], []
    for label, n, rates in specs:
        blocks.append(rng.poisson(rates, size=(n, N_GENES)))
        labels += [label] * n
    matrix = sp.csr_matrix(np.vstack(blocks).astype(np.int32))
    matrix.eliminate_zeros()
    return ad.AnnData(
        matrix,
        obs=pd.DataFrame({PERT_COL: labels}, index=[f"c{i}" for i in range(len(labels))]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(N_GENES)]),
    )


@pytest.fixture(scope="module")
def tiny_pair():
    """(prediction, real) where the prediction recovers the real effects."""
    rng = np.random.default_rng(0)
    base = rng.uniform(5, 30, size=N_GENES)
    effects = {}
    for i in range(5):
        effect = np.ones(N_GENES)
        moved = rng.choice(N_GENES, 15, replace=False)
        effect[moved] = rng.uniform(0.2, 4.0, moved.size)
        effects[f"G{i}"] = base * effect

    real = make_adata(
        [(CONTROL, 300, base)] + [(t, 120, rates) for t, rates in effects.items()], rng
    )
    prediction = make_adata([(t, 120, rates) for t, rates in effects.items()], rng)
    return prediction, real


@pytest.fixture(scope="module")
def mini_cfg_module():
    from vccp.config import load_config

    from tests.conftest import MINI_CONFIG

    return load_config(MINI_CONFIG)


def test_all_six_metrics_come_back(mini_cfg_module, tiny_pair):
    prediction, real = tiny_pair
    result = official.score(
        mini_cfg_module, prediction, real, pert_col=PERT_COL, control_label=CONTROL
    )
    assert result.complete, f"missing: {set(official.SCORED) - set(result.metrics)}"
    assert set(result.scored) == set(official.SCORED)
    assert all(np.isfinite(value) for value in result.scored.values())


def test_the_nsig_diagnostics_sanity_check_7_needs_are_requested(mini_cfg_module, tiny_pair):
    prediction, real = tiny_pair
    result = official.score(
        mini_cfg_module, prediction, real, pert_col=PERT_COL, control_label=CONTROL
    )
    assert set(official.DIAGNOSTICS) <= set(result.metrics)


def test_everything_the_preset_leaves_loose_is_pinned(mini_cfg_module):
    config = official.build_config(mini_cfg_module, PERT_COL, CONTROL)
    described = official.describe_config(config)
    assert described["de_backend"] == mini_cfg_module.eval.de_backend != "auto"
    assert described["device"] == "cpu"
    assert described["pert_col"] == PERT_COL
    assert 1 <= config.num_threads <= mini_cfg_module.resources.cpu_threads


def test_the_scorer_never_asks_for_more_threads_than_the_pool_has(mini_cfg_module):
    """§1.2: respect what the environment says; never raise a pool to fit us."""
    import polars as pl

    assert official.scorer_threads(mini_cfg_module) <= pl.thread_pool_size()
    assert official.scorer_threads(mini_cfg_module) >= 1


def test_controls_are_added_for_scoring_and_the_prediction_is_left_alone(tiny_pair):
    """§8: the submission carries no control rows; the scorer needs some."""
    prediction, real = tiny_pair
    merged = official.with_controls(prediction, real, PERT_COL, CONTROL)

    assert CONTROL not in set(prediction.obs[PERT_COL].astype(str))
    assert CONTROL in set(merged.obs[PERT_COL].astype(str))
    assert merged.n_obs == prediction.n_obs + int(
        (real.obs[PERT_COL].astype(str) == CONTROL).sum()
    )
    assert list(merged.var_names) == list(prediction.var_names)


def test_a_real_side_without_controls_is_refused(tiny_pair):
    prediction, real = tiny_pair
    perturbed_only = real[real.obs[PERT_COL].astype(str) != CONTROL].copy()
    with pytest.raises(ValueError, match="no control group"):
        official.with_controls(prediction, perturbed_only, PERT_COL, CONTROL)


# --------------------------------------------------------------------------- #
# the leaderboard's 0-1 scale
# --------------------------------------------------------------------------- #
def test_the_scale_is_built_from_the_real_data_alone(mini_cfg_module, tiny_pair, tmp_path):
    _, real = tiny_pair
    reference = scale.build(
        mini_cfg_module, real, pert_col=PERT_COL, control_label=CONTROL,
        outdir=tmp_path / "bundle", bundle_id="test",
    )
    if not reference.built:
        pytest.skip(f"the two ends are not estimable on this data: {reference.reason}")
    assert reference.n_splits == scale.competition_rule()[0]
    assert (tmp_path / "bundle").is_dir()


def test_a_scale_that_could_not_be_built_says_why(mini_cfg_module):
    reference = scale.ScaleReference(False, "no data")
    placed = scale.rescale(object(), reference)
    assert placed["available"] is False
    assert placed["reason"] == "no data"


def test_the_no_change_points_are_the_ones_the_specification_names():
    """§6's table is what the sanity checks and the calibration measure from."""
    from vccp.eval import nochange

    assert set(nochange.NO_CHANGE) == set(official.SCORED)
    assert nochange.NO_CHANGE["pds_cosine"] == (0.5, True)
    assert nochange.NO_CHANGE["expr_mse_unbiased_capped_norm"] == (1.0, False)


# --------------------------------------------------------------------------- #
# thread pools: the scorer drives more than one, and they need not agree
# --------------------------------------------------------------------------- #
def test_the_cap_is_the_smallest_pool_not_just_polars(mini_cfg_module, monkeypatch):
    """`NUMBA_NUM_THREADS=1` with `POLARS_MAX_THREADS=4` is exactly what
    `scripts/server_env.sh` produces, and asking numba for 4 raises
    "The number of threads must be between 1 and 1" (DECISIONS.md D47)."""
    monkeypatch.setattr(official, "_pool_limits", lambda: {"polars": 8, "numba": 1})
    assert official.scorer_threads(mini_cfg_module) == 1

    monkeypatch.setattr(official, "_pool_limits", lambda: {"polars": 1, "numba": 8})
    assert official.scorer_threads(mini_cfg_module) == 1

    monkeypatch.setattr(official, "_pool_limits", lambda: {"polars": 8, "numba": 8})
    assert official.scorer_threads(mini_cfg_module) == mini_cfg_module.resources.cpu_threads


def test_the_cap_survives_a_pool_that_cannot_be_asked(mini_cfg_module, monkeypatch):
    monkeypatch.setattr(official, "_pool_limits", dict)
    assert official.scorer_threads(mini_cfg_module) == mini_cfg_module.resources.cpu_threads


def test_both_pools_the_scorer_uses_are_reported():
    pools = official._pool_limits()
    assert {"polars", "numba"} <= set(pools), pools
    assert all(value >= 1 for value in pools.values())


def test_a_thread_refusal_is_recognized():
    assert official._is_thread_refusal(ValueError("The number of threads must be between 1 and 1"))
    assert official._is_thread_refusal(ValueError("thread pool size exceeded"))
    assert not official._is_thread_refusal(ValueError("perturbation sets differ"))

    # numba's *other* refusal, raised when something already started its pool
    # at a different size. A different sentence from D47's and a different
    # exception type, so a retry that caught only ValueError caught half the
    # problem and let the rest take the rehearsal down with it.
    already_launched = RuntimeError(
        "Cannot set NUMBA_NUM_THREADS to a different value once the threads "
        "have been launched (currently have 1, trying to set 4)"
    )
    assert official._is_thread_refusal(already_launched)
    assert isinstance(already_launched, official.THREAD_REFUSAL_ERRORS)


def test_the_retry_covers_both_exception_types_and_nothing_else():
    """The message gate is what keeps the wider catch from swallowing bugs."""
    assert ValueError in official.THREAD_REFUSAL_ERRORS
    assert RuntimeError in official.THREAD_REFUSAL_ERRORS
    # A real scoring failure raised as either type still has to escape.
    for kind in official.THREAD_REFUSAL_ERRORS:
        assert not official._is_thread_refusal(
            kind("the sum of expr_distance_unbiased is not positive")
        )


def test_the_cap_holds_in_a_process_that_really_has_numba_capped(repo_root):
    """The bug only existed where the two variables disagreed, so this sets
    them the way the server does and asks a fresh process."""
    import subprocess
    import sys

    env = {
        **os.environ,
        "NUMBA_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "4",
        "PYTHONPATH": str(repo_root),
    }
    code = (
        "from vccp.config import load_config;"
        "from vccp.eval import official;"
        "cfg = load_config('configs/server.yaml');"
        "pools = official._pool_limits();"
        "threads = official.scorer_threads(cfg);"
        "print(pools, threads);"
        "assert pools['numba'] == 1, pools;"
        "assert threads == 1, threads"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=repo_root, env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
