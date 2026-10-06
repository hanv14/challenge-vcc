"""The lookup's per-entry denoisers and the moments they rest on (D128).

The denoisers are checked on synthetic residuals whose signal is known, so a
test can say what each must keep and what it must drop. The moments are
checked against the lookup's own estimator on `mini_data`, because the
instrument that chooses a denoiser is only worth what that match is worth.
"""

from __future__ import annotations

import numpy as np
import pytest

from vccp.data.replogle import load_replogle_context
from vccp.paths import DataPaths
from vccp.predict import denoise
from vccp.predict.screen_stats import (
    MAX_RUN_ROWS, log2_change, meansq_cp10k, null_log2_sd, residual_z, split_runs,
    stream_moments,
)
from vccp.sanity.checks import mean_cp10k


# --------------------------------------------------------------------------- #
# the SNR denoisers
# --------------------------------------------------------------------------- #
def test_soft_snr_keeps_strong_entries_and_drops_weak_ones():
    residual = np.array([1.0, 1.0, 1.0, -2.0, 0.0])
    z = np.array([10.0, 1.0, 0.1, -np.inf, 3.0])
    out = denoise.snr_soft(residual, z, tau=1.0)
    np.testing.assert_allclose(out[0], 100 / 101, rtol=1e-6)
    np.testing.assert_allclose(out[1], 0.5, rtol=1e-6)        # z² = tau halves it
    assert out[2] < 0.011
    assert out[3] == -2.0                                      # infinite evidence: whole
    assert out[4] == 0.0
    np.testing.assert_array_equal(denoise.snr_soft(residual, z, tau=0.0),
                                  residual.astype(np.float32))


def test_hard_snr_is_all_or_nothing():
    residual = np.array([1.0, -1.0, 0.5])
    z = np.array([3.0, -1.0, np.nan])
    np.testing.assert_array_equal(denoise.snr_hard(residual, z, tau=4.0), [1.0, 0.0, 0.0])


def test_an_undefined_z_reads_as_no_evidence():
    out = denoise.snr_soft(np.array([1.0]), np.array([np.nan]), tau=1.0)
    assert out[0] == 0.0 and np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# z and the noise scale
# --------------------------------------------------------------------------- #
def test_z_is_honest_where_a_target_shows_a_gene_not_at_all():
    """80 cells without one count of a gene the controls show at 0.1 counts
    per cell is unremarkable; a log-scale sd would call it a huge effect."""
    rng = np.random.default_rng(0)
    depth = 12_000
    control_counts = rng.poisson(0.1, size=5000).astype(float)
    control = control_counts / depth * 1e4
    mean_c, meansq_c = control.mean(), (control**2).mean()
    z = residual_z(80, np.array([0.0]), np.array([0.0]), 5000, np.array([mean_c]),
                   np.array([meansq_c]), np.array([0.0]))
    assert -3 < z[0] < 0
    # The log change sits at the floor, far from 0: the reason not to use it.
    assert log2_change(np.array([0.0]), np.array([mean_c]), 0.01)[0] < -2


def test_z_tracks_a_real_change_and_the_screens_mean():
    rng = np.random.default_rng(1)
    control = rng.poisson(20, size=5000) / 12_000 * 1e4
    down = rng.poisson(10, size=200) / 12_000 * 1e4
    args = (np.array([down.mean()]), np.array([(down**2).mean()]), 5000,
            np.array([control.mean()]), np.array([(control**2).mean()]))
    assert residual_z(200, *args, shift=np.array([0.0]))[0] < -15
    # If the whole screen halves this gene, this target is not special.
    assert abs(residual_z(200, *args, shift=np.array([-1.0]))[0]) < 3


def test_the_noise_scale_shrinks_with_cells():
    mean_c, meansq_c = np.array([2.0]), np.array([10.0])
    few = null_log2_sd(np.array([50]), 5000, mean_c, meansq_c, 0.01)
    many = null_log2_sd(np.array([200]), 5000, mean_c, meansq_c, 0.01)
    assert many[0, 0] < few[0, 0]


# --------------------------------------------------------------------------- #
# low rank
# --------------------------------------------------------------------------- #
def planted(n_targets=300, n_genes=120, rank=3, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n_targets, rank))
    components = rng.normal(size=(rank, n_genes))
    signal = loadings @ components
    return signal, signal + noise * rng.normal(size=signal.shape)


def test_the_basis_recovers_a_planted_low_rank_signal():
    signal, observed = planted()
    basis = denoise.fit_basis(observed, np.ones_like(observed), k_max=10, seed=0)
    projected = basis.project(observed, 3)
    err_raw = np.mean((observed - signal) ** 2)
    err_proj = np.mean((projected - signal) ** 2)
    assert err_proj < 0.2 * err_raw


def test_the_basis_is_deterministic_and_sign_fixed():
    _, observed = planted(seed=3)
    a = denoise.fit_basis(observed, np.ones_like(observed), k_max=5, seed=0)
    b = denoise.fit_basis(observed.copy(), np.ones_like(observed), k_max=5, seed=0)
    np.testing.assert_array_equal(a.components, b.components)
    largest = np.argmax(np.abs(a.components), axis=1)
    assert np.all(a.components[np.arange(5), largest] > 0)


def test_genes_that_never_move_are_left_out_of_the_basis():
    _, observed = planted()
    observed[:, :10] = 0.0
    basis = denoise.fit_basis(observed, np.ones_like(observed), k_max=4, seed=0)
    assert np.all(basis.scale[:10] == 0) and np.all(basis.components[:, :10] == 0)
    assert np.all(basis.project(observed, 4)[:, :10] == 0)


def test_lowrank_snr_keeps_a_sparse_effect_the_components_miss():
    signal, observed = planted(noise=0.3)
    basis = denoise.fit_basis(observed, np.ones_like(observed), k_max=5, seed=0)
    spike = observed[0].copy()
    spike[7] += 25.0
    z = spike / 0.3
    kept = denoise.lowrank_snr(spike, z, basis, 3, tau=4.0)
    projected = denoise.lowrank(spike, basis, 3)
    assert kept[7] - projected[7] > 20.0


def test_apply_refuses_a_denoiser_without_its_inputs():
    with pytest.raises(ValueError, match="z"):
        denoise.apply(denoise.SNR, np.ones(3))
    with pytest.raises(ValueError, match="basis"):
        denoise.apply(denoise.LOWRANK, np.ones(3))
    with pytest.raises(ValueError, match="unknown"):
        denoise.apply("magic", np.ones(3))


# --------------------------------------------------------------------------- #
# the streaming moments
# --------------------------------------------------------------------------- #
def test_streamed_moments_equal_the_lookups_estimator(mini_cfg):
    context = load_replogle_context(DataPaths(mini_cfg).discover_phase2_contexts()["K562_gwps"])
    rows = context.cells["row"].to_numpy()
    rng = np.random.default_rng(0)
    first = np.sort(rng.choice(rows, 200, replace=False))
    second = np.sort(rng.choice(np.setdiff1d(rows, first), 40, replace=False))
    # A tiny RAM budget forces many blocks.
    moments = stream_moments(context, [first, second], max_ram_gb=0.01)
    for i, group in enumerate((first, second)):
        block = context.raw_counts(group)
        assert moments.n[i] == group.size
        np.testing.assert_allclose(moments.mean[i], mean_cp10k(block), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(moments.meansq[i], meansq_cp10k(block), rtol=1e-10)


def test_a_cell_in_two_groups_is_refused(mini_cfg):
    context = load_replogle_context(DataPaths(mini_cfg).discover_phase2_contexts()["K562_gwps"])
    with pytest.raises(ValueError, match="two groups"):
        stream_moments(context, [np.array([1, 2]), np.array([2, 3])], max_ram_gb=1.0)


def test_runs_cover_every_row_and_stay_bounded():
    rng = np.random.default_rng(0)
    for rows in (np.sort(rng.choice(200_000, 6_000, replace=False)), np.arange(20_000)):
        runs = split_runs(rows)
        np.testing.assert_array_equal(np.concatenate(runs), rows)
        assert max(r[-1] - r[0] for r in runs) < MAX_RUN_ROWS
