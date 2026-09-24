"""The OT pairing that gives a control cell a perturbed target (PLAN_PERCELL §4).

The two limits are what these tests are mostly about. `epsilon -> inf` has to
reproduce the pooled pseudobulk target exactly, because that is the objective
the pipeline used before and it stays reachable from the same config key;
`epsilon -> 0` has to approach a hard assignment. If either limit drifts, one
end of the sweep stops meaning what the plan says it means.

`epsilon` here is a fraction of the batch's mean cost, so the values below
are the ones the sweep will actually use.
"""

from __future__ import annotations

import math

import pytest
import torch

from vccp.train import ot


def _batch(n_control=6, n_perturbed=8, n_genes=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    control = torch.randn(n_control, n_genes, generator=generator)
    perturbed = torch.randn(n_perturbed, n_genes, generator=generator) + 0.4
    return control, perturbed


def test_infinite_epsilon_is_the_pooled_target():
    """The limit the pooled arm lives at: every control gets the same mean."""
    control, perturbed = _batch()
    target, _ = ot.paired_target(control, perturbed, float("inf"))

    pooled = perturbed.mean(dim=0)
    assert torch.allclose(target, pooled.expand_as(target), atol=1e-6)


def test_large_epsilon_approaches_the_pooled_target():
    """Not just the special case: the limit is continuous."""
    control, perturbed = _batch()
    pooled = perturbed.mean(dim=0).expand(control.shape[0], -1)

    near = ot.paired_target(control, perturbed, 1e4)[0]
    far = ot.paired_target(control, perturbed, 0.05)[0]
    assert (near - pooled).abs().max() < (far - pooled).abs().max()
    assert torch.allclose(near, pooled, atol=1e-3)


def test_small_epsilon_approaches_a_hard_assignment():
    """Each control cell ends up matched to its own nearest perturbed cell."""
    n = 6
    identity = torch.eye(n) * 8.0
    control = identity
    perturbed = identity + 0.01

    _, log_coupling = ot.paired_target(control, perturbed, 0.01)
    matched = torch.softmax(log_coupling, dim=1).argmax(dim=1)
    assert torch.equal(matched, torch.arange(n))

    diagnostics = ot.coupling_diagnostics(log_coupling)
    assert diagnostics["effective_partners"] < 1.1
    assert diagnostics["max_weight"] > 0.9


def test_epsilon_orders_the_concentration():
    """The sweep has to actually move the pairing, monotonically."""
    control, perturbed = _batch(n_control=12, n_perturbed=12, n_genes=6)
    partners = [
        ot.coupling_diagnostics(
            ot.paired_target(control, perturbed, epsilon)[1]
        )["effective_partners"]
        for epsilon in (0.005, 0.01, 0.02, 0.05)
    ]
    assert partners == sorted(partners), partners
    assert partners[0] < partners[-1]


@pytest.mark.parametrize("epsilon", [0.02, 0.05, 1.0, float("inf")])
def test_marginals_are_uniform(epsilon):
    """Rows sum to 1/B and columns to 1/M, at every epsilon of the sweep."""
    control, perturbed = _batch(n_control=7, n_perturbed=9)
    _, log_coupling = ot.paired_target(control, perturbed, epsilon)
    coupling = torch.exp(log_coupling)

    assert torch.allclose(coupling.sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(
        coupling.sum(dim=1), torch.full((7,), 1 / 7), atol=1e-4
    )
    assert torch.allclose(
        coupling.sum(dim=0), torch.full((9,), 1 / 9), atol=1e-4
    )
    # And the row-normalized form the target is built from.
    assert torch.allclose(
        torch.softmax(log_coupling, dim=1).sum(dim=1), torch.ones(7), atol=1e-6
    )


def test_target_is_a_convex_combination_of_real_cells():
    """A weighted average of cells, never an extrapolation beyond them."""
    control, perturbed = _batch()
    for epsilon in (0.01, 0.05, float("inf")):
        target, _ = ot.paired_target(control, perturbed, epsilon)
        assert (target >= perturbed.min(dim=0).values - 1e-5).all()
        assert (target <= perturbed.max(dim=0).values + 1e-5).all()


def test_the_pairing_can_be_reused_on_another_gene_set():
    """How the rest genes borrow the panel's pairing for the delta term."""
    control, perturbed = _batch(n_genes=4)
    rest = torch.randn(perturbed.shape[0], 11)

    target, log_coupling = ot.paired_target(control, perturbed, 0.05, values=rest)
    assert target.shape == (control.shape[0], 11)
    assert torch.allclose(target, ot.barycentric_target(log_coupling, rest))


def test_the_cost_is_per_gene_so_epsilon_is_portable():
    """Two panels of different width give costs on the same scale."""
    generator = torch.Generator().manual_seed(3)
    narrow = torch.randn(4, 10, generator=generator)
    wide = narrow.repeat(1, 5)

    cost_narrow = ot.squared_cost(narrow, narrow + 1.0)
    cost_wide = ot.squared_cost(wide, wide + 1.0)
    assert torch.allclose(cost_narrow, cost_wide, atol=1e-5)


def test_the_mask_keeps_unmeasured_genes_out_of_the_cost():
    """Genes the screen does not measure carry no truth to pair on."""
    control, perturbed = _batch(n_genes=6)
    mask = torch.zeros(6, dtype=torch.bool)
    mask[:3] = True

    masked = ot.squared_cost(control, perturbed, mask)
    direct = ot.squared_cost(control[:, :3], perturbed[:, :3])
    assert torch.allclose(masked, direct, atol=1e-6)

    # And the unmeasured half genuinely cannot influence it.
    noisy = perturbed.clone()
    noisy[:, 3:] += 50.0
    assert torch.allclose(ot.squared_cost(control, noisy, mask), masked, atol=1e-6)


def test_nothing_differentiates_through_sinkhorn():
    """The coupling builds the target; it is not part of the model."""
    control, perturbed = _batch()
    control.requires_grad_(True)
    perturbed.requires_grad_(True)

    target, log_coupling = ot.paired_target(control, perturbed, 0.05)
    assert not target.requires_grad
    assert not log_coupling.requires_grad


def test_clear_errors_rather_than_silent_nonsense():
    control, perturbed = _batch(n_genes=5)

    with pytest.raises(ValueError, match="positive"):
        ot.sinkhorn_log(torch.zeros(3, 3), 0.0)
    with pytest.raises(ValueError, match="genes on the control side"):
        ot.squared_cost(control, torch.randn(4, 7))
    with pytest.raises(ValueError, match="mask covers"):
        ot.squared_cost(control, perturbed, torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="keeps no gene"):
        ot.squared_cost(control, perturbed, torch.zeros(5, dtype=torch.bool))
    with pytest.raises(ValueError, match="not finite"):
        broken = perturbed.clone()
        broken[0, 0] = float("nan")
        ot.squared_cost(control, broken)
    with pytest.raises(ValueError, match="the pairing is by row"):
        ot.paired_target(control, perturbed, 0.05, values=torch.randn(3, 4))
    with pytest.raises(ValueError, match="nothing to transport"):
        same = torch.ones(4, 5)
        ot.paired_target(same, same.clone(), 0.05)


def test_runs_on_the_shapes_phase2_will_use():
    """A realistic batch: the cost matrix is small, the panel is not."""
    n_cells, n_genes = 256, 955
    control, perturbed = _batch(n_control=n_cells, n_perturbed=n_cells, n_genes=n_genes)
    target, log_coupling = ot.paired_target(control, perturbed, 0.02)
    assert target.shape == (n_cells, n_genes)
    assert log_coupling.shape == (n_cells, n_cells)
    assert torch.isfinite(target).all()
    assert not math.isnan(ot.coupling_diagnostics(log_coupling)["effective_partners"])


def test_the_diagnostics_report_the_premise_and_the_residual():
    """cost_spread checks the premise; marginal_drift checks the solver."""
    control, perturbed = _batch(n_control=32, n_perturbed=32, n_genes=40)
    cost = ot.squared_cost(control, perturbed)
    log_coupling = ot.sinkhorn_log(cost, 0.05 * float(cost.mean()))

    diagnostics = ot.coupling_diagnostics(log_coupling, cost)
    assert diagnostics["cost_spread"] > 0
    assert diagnostics["marginal_drift"] == pytest.approx(
        ot.marginal_drift(log_coupling)
    )

    # A cost with no structure cannot pair anything: every row is uniform,
    # whatever epsilon says. This is the degenerate case cost_spread warns of.
    flat = torch.full((8, 8), 2.0)
    uninformative = ot.sinkhorn_log(flat, 0.05 * 2.0)
    assert ot.coupling_diagnostics(uninformative, flat)["cost_spread"] == 0.0
    assert ot.coupling_diagnostics(uninformative)["effective_partners"] == pytest.approx(
        8.0, rel=1e-3
    )
