"""Entropic optimal transport: building a per-cell target where none exists.

Replogle is a destructive assay. Control cell *i* was never perturbed and
perturbed cell *j* was never observed unperturbed, so the mapping the
specification asks for — a cell and a perturbation token in, that cell
perturbed out (CLAUDE.md §4.2) — has no `y` for its `x`. Supervision has to
construct one.

This module constructs it. Given a batch of control cells and a batch of
cells perturbed for the same target, it solves an entropic optimal transport
problem between them and returns, for each control cell, the
coupling-weighted average of the perturbed cells it was matched to. That
average is the training target.

Why *entropic* and not a hard one-to-one matching: at ~12,000 UMIs the
per-cell Poisson scatter is around 25x the perturbation response (variance
1.0 against 0.04 in control-SD units). A hard assignment under that much
noise is close to arbitrary, and a squared-error loss against an arbitrarily
chosen single cell fits noise. The barycentric average is the
variance-reduced version of the same idea.

Why matching on "nuisance" is the mechanism rather than a flaw: the cost is
dominated by sequencing depth and cell state, not by the perturbation.
Pairing like with like on those factors means the residual difference
between paired cells is closer to the perturbation than the difference
between a random control and a random perturbed cell is.

**`epsilon` is a fraction of the batch's mean cost, not an absolute
distance.** Measured on a 955-gene panel in control-SD units, the mean cost
is ~2.0 and the whole pooled-to-hard transition happens between 0.005 and
0.05 of it — a window too narrow, and too dependent on a screen's noise
level, for an absolute value to land in reliably on every screen. A fraction
lands in the same place on any panel and any screen. `sinkhorn_log` below
still takes an absolute value, because it is the solver; the conversion
happens once, in `paired_target`.

`epsilon` spans the whole design space, which is why it is one config key
rather than a fork:

* `epsilon -> inf` — uniform coupling, every control matched to everything
  equally, and the barycentric target collapses to the perturbed pseudobulk.
  **That is the pooled loss the pipeline used before this.**
* `epsilon -> 0` — the coupling concentrates on the nearest perturbed cell.
* in between — local averages.

So the pooled formulation is not a separate code path to be maintained
against this one; it is this one at an extreme of its only parameter, and
the rehearsal can measure where between the two the optimum sits.

The coupling is **not** differentiated. It builds the target; it is not part
of the model. Every function here runs under `torch.no_grad` and returns a
detached tensor, so a caller cannot accidentally backpropagate through
Sinkhorn — which would buy nothing here and bias the gradient.
"""

from __future__ import annotations

import math

import torch

#: A ceiling, not a cost: `sinkhorn_log` stops as soon as the rows converge,
#: which on a realistic 256x256 batch takes 2 iterations at `epsilon` 0.05
#: and 36 at 0.01. Below about 0.01 nothing converges in any affordable
#: number of iterations, which is a fact about the problem rather than about
#: this ceiling — `coupling_diagnostics` reports the residual so a run at
#: that end of the sweep is visible instead of silently approximate. The
#: caller's config key is `phase2.ot_iterations`.
DEFAULT_ITERATIONS = 200

#: Stop once no row of the coupling is further than this, in log space, from
#: the marginal it should have. A test on the marginals is scale-free, where
#: a test on how far the potentials moved is not: the potentials carry a
#: factor of `epsilon`, so an absolute threshold on them would stop a
#: small-`epsilon` run almost immediately.
CONVERGENCE_TOLERANCE = 1e-4


def squared_cost(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """`C[i, j]` = mean squared difference between source *i* and target *j*.

    `source` is `(B, G)`, `target` is `(M, G)`, the result is `(B, M)`.

    The mean — rather than the sum — is what makes `epsilon` portable: a
    screen measuring 716 panel genes and one measuring 812 give costs on the
    same scale, so one swept value of `epsilon` means the same thing in both
    and on the server's full panel.

    `mask` is an optional `(G,)` boolean over genes the screen measures
    (§4.5 takes its masks from `genes.csv`). Unmeasured genes carry no truth,
    so letting them into the cost would pair cells on values that are not
    there.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"expected (cells, genes) for both sides, got {tuple(source.shape)} "
            f"and {tuple(target.shape)}"
        )
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"{source.shape[1]} genes on the control side against "
            f"{target.shape[1]} on the perturbed side"
        )

    if mask is None:
        n_genes = source.shape[1]
        if n_genes == 0:
            raise ValueError("the cost needs at least one gene")
        cost = torch.cdist(source, target) ** 2 / n_genes
    else:
        keep = mask.to(torch.bool).reshape(-1)
        if keep.shape[0] != source.shape[1]:
            raise ValueError(
                f"the mask covers {keep.shape[0]} genes but the cells have "
                f"{source.shape[1]}"
            )
        n_genes = int(keep.sum())
        if n_genes == 0:
            raise ValueError(
                "the mask keeps no gene, so no cost can be formed; this screen "
                "measures none of the panel"
            )
        cost = torch.cdist(source[:, keep], target[:, keep]) ** 2 / n_genes

    if not torch.isfinite(cost).all():
        raise ValueError(
            "the transport cost is not finite; the drawn cells carry NaN or inf "
            "in the genes the mask keeps"
        )
    return cost


def sinkhorn_log(
    cost: torch.Tensor,
    epsilon: float,
    iterations: int = DEFAULT_ITERATIONS,
    tolerance: float = CONVERGENCE_TOLERANCE,
) -> torch.Tensor:
    """The entropic OT coupling for `cost`, as `log P`, with uniform marginals.

    Solved in the log domain: at the small end of the `epsilon` sweep
    `exp(-C / epsilon)` underflows float32 outright, while `logsumexp`
    subtracts its own maximum and stays exact.

    Returns `(B, M)` such that `exp(result)` sums to 1 over the whole matrix,
    with row sums `1/B` and column sums `1/M`.
    """
    if cost.ndim != 2:
        raise ValueError(f"expected a (B, M) cost matrix, got {tuple(cost.shape)}")
    if epsilon <= 0:
        raise ValueError(
            f"epsilon must be positive, got {epsilon}; the hard-assignment limit "
            "is approached with a small positive value, not with zero"
        )

    n_source, n_target = cost.shape
    log_a = -math.log(n_source)
    log_b = -math.log(n_target)

    with torch.no_grad():
        if math.isinf(epsilon):
            # The uniform coupling. Taken as a limit rather than computed,
            # because C / inf is zero and the potentials would then be
            # inf * 0. This is the pooled objective, and the sweep includes
            # it so the pooled arm is reachable from the same config key.
            return torch.full_like(cost, log_a + log_b)

        scaled = cost / epsilon
        f = torch.zeros(n_source, dtype=cost.dtype, device=cost.device)
        g = torch.zeros(n_target, dtype=cost.dtype, device=cost.device)

        for _ in range(max(1, int(iterations))):
            # The f update makes the row marginals exact and the g update
            # makes the column marginals exact, each at the other's expense.
            # So the rows are what drifts, and they are what to test.
            f = epsilon * (log_a - torch.logsumexp(g / epsilon - scaled, dim=1))
            g = epsilon * (log_b - torch.logsumexp(f[:, None] / epsilon - scaled, dim=0))
            log_coupling = f[:, None] / epsilon + g[None, :] / epsilon - scaled
            drift = torch.max(torch.abs(torch.logsumexp(log_coupling, dim=1) - log_a))
            if float(drift) < tolerance:
                break

        return log_coupling


def marginal_drift(log_coupling: torch.Tensor) -> float:
    """How far the coupling's row sums are from uniform, in log space.

    The residual `sinkhorn_log` stops on. Reported rather than asserted,
    because at the concentrated end of the `epsilon` sweep no affordable
    number of iterations converges and the barycentric target is still a
    perfectly usable weighted average — it just is not the exact OT one, and
    that should be visible in the training log rather than assumed away.
    """
    with torch.no_grad():
        uniform = -math.log(log_coupling.shape[0])
        return float(torch.max(torch.abs(torch.logsumexp(log_coupling, dim=1) - uniform)))


def barycentric_target(log_coupling: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Each source cell's target: the coupling-weighted average of `values`.

    `log_coupling` is `(B, M)` from `sinkhorn_log`, `values` is `(M, G)` — the
    perturbed cells, over whatever gene set the loss will use. The result is
    `(B, G)`.

    The coupling is row-normalized first, so each row is a probability
    distribution over the perturbed cells and the target is an average of
    real cells rather than a scaled-down one.
    """
    if log_coupling.shape[1] != values.shape[0]:
        raise ValueError(
            f"the coupling matches {log_coupling.shape[1]} targets but "
            f"{values.shape[0]} value rows were given"
        )
    with torch.no_grad():
        weights = torch.softmax(log_coupling, dim=1)
        return (weights @ values).detach()


def paired_target(
    control: torch.Tensor,
    perturbed: torch.Tensor,
    epsilon: float,
    *,
    mask: torch.Tensor | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Control cells in, one constructed perturbed target per control cell out.

    The whole module in one call: cost, coupling, barycentric projection.
    Returns `(target, log_coupling)` — the coupling comes back because the
    same pairing is reused for the rest genes in the delta consistency term
    (PLAN_PERCELL.md §5), and recomputing it there would both cost a second
    Sinkhorn and risk pairing the two halves of one cell differently.

    `epsilon` is a **fraction of this batch's mean cost** (see the module
    docstring), so the same swept value means the same thing on any panel and
    any screen. It is resolved against the batch rather than against a
    per-screen constant so that a screen whose cells happen to be noisier
    than usual does not silently slide along the sweep; the cost is a little
    step-to-step variation in the effective `epsilon`, which is small next to
    the variation the cell draw itself contributes.

    `values` defaults to `perturbed` itself. Pass a different matrix with the
    same row order to project onto another gene set — that is how the rest
    genes borrow this pairing.
    """
    with torch.no_grad():
        cost = squared_cost(control, perturbed, mask)
        scale = float(cost.mean())
        if scale <= 0:
            raise ValueError(
                "every control cell is identical to every perturbed cell, so there "
                "is nothing to transport"
            )
        log_coupling = sinkhorn_log(cost, epsilon * scale, iterations)
        projected = perturbed if values is None else values
        if projected.shape[0] != perturbed.shape[0]:
            raise ValueError(
                f"{projected.shape[0]} value rows against {perturbed.shape[0]} perturbed "
                "cells; the pairing is by row, so the two must line up"
            )
        return barycentric_target(log_coupling, projected), log_coupling


def coupling_diagnostics(
    log_coupling: torch.Tensor, cost: torch.Tensor | None = None
) -> dict[str, float]:
    """How concentrated the pairing is, for the training log.

    `effective_partners` is the perplexity of a row — 1.0 means each control
    cell was matched to a single perturbed cell, `M` means it was matched to
    all of them equally (the pooled limit). It is the number that says where
    on the `epsilon` sweep a run *actually* sits, as opposed to where its
    config says it sits, and it is the first thing to look at if per-cell
    training behaves exactly like pooled training.

    `cost_spread` — the standard deviation of the cost over its mean — is the
    other one to watch, and it is a check on the premise rather than on the
    solver. In high dimension pairwise distances concentrate: on 955 genes of
    independent noise every control cell is nearly equidistant from every
    perturbed cell, the coupling carries almost no information, and OT
    degenerates into pooling whatever `epsilon` says. Real cells differ in
    depth and cell state, which is what the pairing is meant to exploit, so
    a spread near zero on real data means the mechanism is not there.
    """
    with torch.no_grad():
        weights = torch.softmax(log_coupling, dim=1)
        entropy = -(weights * torch.log(weights.clamp_min(1e-30))).sum(dim=1)
        diagnostics = {
            "effective_partners": float(torch.exp(entropy).mean()),
            "max_weight": float(weights.max(dim=1).values.mean()),
            "n_targets": int(log_coupling.shape[1]),
            "marginal_drift": marginal_drift(log_coupling),
        }
        if cost is not None:
            mean = float(cost.mean())
            diagnostics["cost_mean"] = mean
            diagnostics["cost_spread"] = float(cost.std()) / mean if mean else 0.0
        return diagnostics
