"""Does the OT coupling carry any information? (PLAN_PERCELL.md §11)

The premise of the per-cell redesign is that a control cell can be matched
to the perturbed cells it most resembles, so that the residual difference
between matched cells is closer to the perturbation than the difference
between a random pair. This asks whether that premise holds on real cells,
**before** any training is built on it.

It has to be asked because of distance concentration. In 955 dimensions the
pairwise costs between independent noise vectors span 1.69 to 2.47 around a
mean of 2.04 — every control cell is nearly equidistant from every perturbed
cell. A coupling built on such a cost is uniform whatever `epsilon` says, the
barycentric target collapses to the perturbed pseudobulk, and OT degenerates
into exactly the pooled objective it was meant to replace. Silently: a
degenerate coupling looks identical to a correctly configured pooled run, so
without this it would cost a server cycle to notice.

Four numbers, per screen and per `epsilon`:

* **cost_spread** — the cost's standard deviation over its mean. The raw
  material. Under independent noise on `G` genes this is about
  `sqrt(8 / G) / 2`, which the report prints beside it as `cost_spread_null`
  so the measured value can be read against the value that means nothing.
* **effective_partners** — the perplexity of a coupling row. 1 means each
  control cell was matched to a single perturbed cell, `B` means it was
  matched to all of them equally, which is the pooled limit.
* **target_spread** — the decisive one. How much the barycentric targets
  vary between control cells, against how much the perturbed cells
  themselves vary. **Zero means every control cell is handed the same
  target, so the loss is the pooled loss.** One means each cell is handed a
  distinct cell. This is the quantity the training step actually sees.
* **residual_reduction** — the mechanism as a refutable number:
  `E||T_i - c_i||^2 / E||pooled_mean - c_i||^2`. Below one means pairing
  brought each control cell's target nearer to it, so the residual the model
  is asked to explain is the part pairing could not account for. At one the
  pairing bought nothing and the per-cell target is the pooled one at a
  different address.
* **library_size_pearson** — whether the pairing matches on sequencing
  depth. §4 claims the coupling pairs like with like on depth and cell
  state, and that this is the mechanism rather than a flaw. If this is near
  zero the coupling is pairing on noise, and the mechanism is not there.

A **null arm** pairs control cells against *other control cells*. Its
`target_spread` and `library_size_pearson` are what the same machinery
produces when there is no perturbation to find, so the perturbed arm is read
against it rather than against zero.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import torch

from ..config import Config
from ..logging_utils import get_logger
from ..train import ot

#: Below this `target_spread` at the most concentrated `epsilon`, every
#: control cell is handed essentially the pooled mean and the per-cell loss
#: is the pooled loss under another name.
DEGENERATE_TARGET_SPREAD = 0.02

#: Below this `library_size_pearson` at every `epsilon`, the coupling is not
#: matching on depth, so the mechanism §4 claims is not the one operating.
UNINFORMATIVE_LIBRARY_PEARSON = 0.1

#: The sweep, as fractions of the batch's mean cost (DECISIONS.md D52).
EPSILONS = (0.005, 0.01, 0.02, 0.05, float("inf"))

CONTROL_ARM = "control_vs_control_null"
PERTURBED_ARM = "control_vs_perturbed"


def null_cost_spread(n_genes: int) -> float:
    """The `cost_spread` independent noise would give on `n_genes` genes.

    Two independent standard-normal cells differ by `N(0, 2)` per gene, so
    the per-gene squared difference has mean 2 and variance 8, and averaging
    over `G` genes divides the variance by `G`. Printed beside the measured
    value so a spread that means nothing is recognizable as such.
    """
    return math.sqrt(8.0 / max(n_genes, 1)) / 2.0


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation, or NaN where one side has no variation to correlate.

    At `epsilon = inf` every source cell is coupled to the same weighted mean,
    so the partner values are constant up to floating-point wobble. Testing
    for an exactly zero standard deviation would let that wobble through as a
    meaningless correlation near zero, which reads as "the pairing carries no
    depth information" rather than "there is no pairing".
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    for side in (a, b):
        scale = np.abs(side).mean()
        if side.std() <= 1e-9 * max(scale, 1.0):
            return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def target_spread(log_coupling: torch.Tensor, values: torch.Tensor) -> float:
    """How much the barycentric targets vary, against the cells' own spread.

    The quantity the per-cell loss sees. If it is zero the loss is the pooled
    loss whatever `epsilon` is set to.
    """
    targets = ot.barycentric_target(log_coupling, values)
    between = float(((targets - targets.mean(dim=0)) ** 2).mean())
    within = float(((values - values.mean(dim=0)) ** 2).mean())
    return between / max(within, 1e-12)


def residual_reduction(log_coupling: torch.Tensor, source: torch.Tensor, values: torch.Tensor) -> float:
    """How much closer the paired target is than the pooled one.

    The mechanism §4 claims, stated as a number the check can refute:

        E||T_i - c_i||^2  /  E||pooled_mean - c_i||^2

    Below 1 means pairing brought each control cell's target nearer to it,
    which is what "matching like with like on depth and cell state" has to
    mean if it means anything — the residual the model is then asked to
    explain is the part that pairing could *not* account for, and is closer
    to the perturbation than the raw difference of two random cells.

    At 1.0 the pairing has bought nothing and the per-cell target is the
    pooled one at a different address.
    """
    with torch.no_grad():
        targets = ot.barycentric_target(log_coupling, values)
        paired = float(((targets - source) ** 2).mean())
        pooled = float(((values.mean(dim=0) - source) ** 2).mean())
        return paired / max(pooled, 1e-12)


def partner_library_size(log_coupling: torch.Tensor, library: np.ndarray) -> np.ndarray:
    """Each source cell's coupling-weighted mean partner library size."""
    with torch.no_grad():
        weights = torch.softmax(log_coupling, dim=1).cpu().numpy()
    return weights @ np.asarray(library, dtype=np.float64)


def measure_arm(
    source_panel: torch.Tensor,
    target_panel: torch.Tensor,
    source_library: np.ndarray,
    target_library: np.ndarray,
    epsilons=EPSILONS,
    iterations: int = ot.DEFAULT_ITERATIONS,
) -> dict[str, Any]:
    """Every number, for one pair of cell batches."""
    cost = ot.squared_cost(source_panel, target_panel)
    n_genes = int(source_panel.shape[1])
    per_epsilon = {}

    for epsilon in epsilons:
        log_coupling = ot.sinkhorn_log(
            cost, epsilon * float(cost.mean()) if math.isfinite(epsilon) else epsilon, iterations
        )
        diagnostics = ot.coupling_diagnostics(log_coupling, cost)
        partners = partner_library_size(log_coupling, target_library)
        per_epsilon[str(epsilon)] = {
            **diagnostics,
            "target_spread": target_spread(log_coupling, target_panel),
            "residual_reduction": residual_reduction(
                log_coupling, source_panel, target_panel
            ),
            "library_size_pearson": pearson(np.log1p(source_library), np.log1p(partners)),
        }

    return {
        "n_source_cells": int(source_panel.shape[0]),
        "n_target_cells": int(target_panel.shape[0]),
        "n_genes": n_genes,
        "cost_mean": float(cost.mean()),
        "cost_spread": float(cost.std()) / max(float(cost.mean()), 1e-12),
        "cost_spread_null": null_cost_spread(n_genes),
        "per_epsilon": per_epsilon,
    }


def verdict(arms: dict[str, Any]) -> tuple[str, str]:
    """`(verdict, reading)` from the perturbed arm's numbers.

    `residual_reduction` does not decide the verdict, but it qualifies it.
    The two numbers that matter pull in opposite directions: `target_spread`
    wants a small `epsilon`, where each control cell gets a distinct target,
    and `residual_reduction` wants a large one, because a target averaged
    over few cells carries its own sampling noise and that noise can exceed
    what the pairing removed. The sweep exists to find the balance, and the
    reading says where it currently sits so the sweep starts from a measured
    place rather than a guess.
    """
    perturbed = arms.get(PERTURBED_ARM, {}).get("per_epsilon", {})
    if not perturbed:
        return "not-measured", "no screen produced a coupling"

    finite = {k: v for k, v in perturbed.items() if k != "inf"}
    if not finite:
        return "not-measured", "the sweep produced no finite epsilon"

    smallest = min(finite, key=float)
    spread = finite[smallest]["target_spread"]
    correlations = [
        v["library_size_pearson"]
        for v in finite.values()
        if not math.isnan(v["library_size_pearson"])
    ]
    best_library = max(correlations) if correlations else float("nan")

    if spread < DEGENERATE_TARGET_SPREAD:
        return "degenerate", (
            f"at epsilon {smallest} the barycentric targets vary only "
            f"{spread:.4f} as much as the perturbed cells do, so every control "
            "cell is handed essentially the pooled mean. The per-cell loss "
            "would be the pooled loss under another name, and OT should be "
            "abandoned rather than tuned."
        )
    if math.isnan(best_library) or best_library < UNINFORMATIVE_LIBRARY_PEARSON:
        return "pairs-on-noise", (
            f"the targets do vary between cells ({spread:.3f} at epsilon "
            f"{smallest}), but the pairing does not track sequencing depth at "
            f"any epsilon (best pearson {best_library:.3f}). The mechanism "
            "§4 claims — matching like with like on depth and cell state — is "
            "not the one operating, so the residual after pairing is not "
            "obviously closer to the perturbation."
        )
    residuals = {k: v["residual_reduction"] for k, v in finite.items()}
    best_epsilon = min(residuals, key=residuals.get)
    reading = (
        f"the barycentric targets vary {spread:.3f} as much as the perturbed "
        f"cells at epsilon {smallest}, and the pairing tracks sequencing depth "
        f"(pearson up to {best_library:.3f}). The premise holds."
    )
    if residuals[smallest] >= 1.0:
        reading += (
            f" But pairing does not reduce the residual at that end: at epsilon "
            f"{smallest} the paired target sits {residuals[smallest]:.2f}x as far "
            f"from its control cell as the pooled mean does, because a target "
            f"averaged over {finite[smallest]['effective_partners']:.1f} cells "
            f"carries its own sampling noise. The residual is smallest at "
            f"epsilon {best_epsilon} ({residuals[best_epsilon]:.2f}x), where the "
            f"targets vary {finite[best_epsilon]['target_spread']:.3f}. Those two "
            "wants pull opposite ways, which is what the sweep is for."
        )
    return "informative", reading


def run_coupling_check(
    cfg: Config,
    screen: str | None = None,
    n_targets: int = 4,
    n_cells: int = 256,
    epsilons=EPSILONS,
) -> dict[str, Any]:
    """The diagnostic, over one screen's real cells."""
    from ..data import reference
    from ..paths import DataPaths, RunPaths
    from ..phases.phase2 import draw_cells, load_contexts
    from ..runtime import resolve_device

    log = get_logger()
    device = resolve_device(cfg.device)
    run_paths = RunPaths(cfg)
    run_paths.ensure()

    axis = reference.load_gene_names(DataPaths(cfg).gene_names)
    gene_index = {symbol: i for i, symbol in enumerate(axis)}
    contexts = load_contexts(cfg, device, gene_index)

    name = screen or sorted(contexts)[0]
    if name not in contexts:
        raise KeyError(f"unknown screen {name!r}; this data has {sorted(contexts)}")
    tensors = contexts[name]
    cells = tensors.context.cells.set_index("row")

    rng = np.random.default_rng(cfg.seed)
    available = int(tensors.control_rows.size)
    per_draw = min(n_cells, available // 2)
    if per_draw < 2:
        raise RuntimeError(
            f"{name} has {available} control cells, too few to draw two disjoint batches"
        )

    drawn = rng.choice(tensors.control_rows, 2 * per_draw, replace=False)
    source_rows, null_rows = drawn[:per_draw], drawn[per_draw:]
    source_panel, _ = draw_cells(tensors, None, per_draw, rng, device, rows=source_rows)
    null_panel, _ = draw_cells(tensors, None, per_draw, rng, device, rows=null_rows)
    library = lambda rows: cells.loc[list(rows), "library_size"].to_numpy()  # noqa: E731

    log.info(
        "coupling check: %s, %d control cells against %d, %d panel genes",
        name, per_draw, per_draw, int(source_panel.shape[1]),
    )

    arms = {
        CONTROL_ARM: measure_arm(
            source_panel, null_panel, library(source_rows), library(null_rows), epsilons
        )
    }

    # The perturbed arm pools several targets' cells, because one target's
    # cells are too few on mini to give the coupling anything to work with
    # and the question here is about the geometry, not about any one target.
    targets = [t for t in tensors.train_targets if t in tensors.pseudobulk][:n_targets]
    perturbed_rows: list[int] = []
    for target in targets:
        rows = cells.index[
            (cells["target_gene"].astype(str) == target) & (~cells["is_control"])
        ].to_numpy()
        if rows.size:
            take = min(max(per_draw // max(len(targets), 1), 1), rows.size)
            perturbed_rows.extend(rng.choice(rows, take, replace=False).tolist())

    if len(perturbed_rows) >= 2:
        perturbed_panel, _ = draw_cells(
            tensors, None, len(perturbed_rows), rng, device, rows=np.asarray(perturbed_rows)
        )
        arms[PERTURBED_ARM] = {
            "targets": targets,
            **measure_arm(
                source_panel,
                perturbed_panel,
                library(source_rows),
                library(perturbed_rows),
                epsilons,
            ),
        }

    decision, reading = verdict(arms)
    report = {
        "screen": name,
        "device": device,
        "epsilons": [str(e) for e in epsilons],
        "arms": arms,
        "verdict": decision,
        "reading": reading,
        "thresholds": {
            "degenerate_target_spread": DEGENERATE_TARGET_SPREAD,
            "uninformative_library_pearson": UNINFORMATIVE_LIBRARY_PEARSON,
        },
    }

    for arm, entry in arms.items():
        log.info(
            "  %s: cost mean %.3f, spread %.4f (noise alone would give %.4f)",
            arm, entry["cost_mean"], entry["cost_spread"], entry["cost_spread_null"],
        )
        log.info(
            "    %-8s %-10s %-14s %-12s %-10s",
            "epsilon", "partners", "target spread", "residual", "lib r",
        )
        for epsilon, values in entry["per_epsilon"].items():
            log.info(
                "    %-8s %-10.1f %-14.4f %-12.4f %-10.3f",
                epsilon,
                values["effective_partners"],
                values["target_spread"],
                values["residual_reduction"],
                values["library_size_pearson"],
            )

    log.info("")
    log.info("verdict: %s", decision.upper())
    log.info("  %s", reading)

    path = run_paths.reports / "coupling_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    log.info("written -> %s", path)
    return report
