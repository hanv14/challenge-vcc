"""What could the panel→rest mapping reach? (PLAN_MAPPING.md §B1)

Phase 2's mapping ended at `map_control_mse` 0.965, 1.144 and 1.109 across the
three screens, with `map_control_pearson` never above 0.08. That was first read
against 1.0, which is what predicting each gene's control mean scores over the
whole control population, and so as "worse than the zero-initialized model on
two screens". It was not: the 32 cells an evaluation draws have their own
no-change error, 0.97 to 1.07 depending on the draw, and an arm whose mapping
received no gradient at all scores the same 1.07 (D105). Phase 2 now reports
`map_control_ratio_to_no_change` on the same cells. The pearson reading stands.
Before trying to improve the mapping, this bounds it, with two numbers that
need no GPU and no training:

* **the linear ceiling.** Ridge regression from the panel to the rest genes,
  on control cells, in the same control-SD units the model uses. A linear map
  is a weak model. If it reaches a correlation the gene-token model does not,
  the model is the problem. If it lands at the same place, the panel does not
  predict single-cell rest genes at this sequencing depth and no architecture
  will change that.
* **the target's reliability.** Split each cell's counts in two by binomial
  thinning, normalize each half at half the library size, and correlate one
  half's rest genes against the other's. No predictor can correlate with a
  target better than the target correlates with itself. Corrected for the
  halved depth by Spearman–Brown, `2r / (1 + r)`, that is the ceiling on
  *any* method, including the upper bounds the rehearsal reports.

Both are computed exactly as `map_control_pearson` is — a flattened
correlation of z-scored values — so they sit beside the model's own number
without a units argument.

A third fit, on **library size alone**, separates the two things the panel
could be carrying. If one predictor does as well as 724, the mapping is a
depth correction with extra steps; §4's claim is that it is more than that.

Streamed: the normal equations are accumulated a block of rows at a time, so
the 2-million-cell K562 file is never held (CLAUDE.md §3.3, §1.2). Nothing is
trained and no checkpoint is written.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..config import Config
from ..logging_utils import get_logger

#: Ridge penalties, as a multiple of `n_fit` so the same values mean the same
#: thing whatever the sample size. 0 is ordinary least squares, which is
#: underdetermined when there are fewer cells than panel genes and says so
#: loudly rather than being left out. The top end is wide because a ceiling
#: chosen at the edge of its grid is a statement about the grid: when the best
#: penalty is the largest offered, `best_at_grid_edge` is set and the number
#: is a lower bound on the linear ceiling, not the ceiling.
PENALTIES = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
#: Control cells read per block. 4,096 x ~8,000 genes of float64 is about
#: 260 MB, well inside `resources.max_ram_gb`, and the file is read in one
#: sorted pass per block.
BLOCK_CELLS = 4096
#: Below this many held-out cells the evaluation is not worth reporting.
MIN_EVAL_CELLS = 64
#: A correlation the model would have to beat before the mapping counts as
#: underperforming a linear baseline rather than the task being hard.
LINEAR_MARGIN = 0.05
#: Below this ceiling on the achievable correlation, per-cell prediction is not
#: worth pursuing whatever the model does — there is nothing there to reach.
#: Compared against `sqrt(reliability)`, not against the reliability (D101).
MIN_ACHIEVABLE_PEARSON = 0.10


def _z(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float64)


def _design(panel: np.ndarray) -> np.ndarray:
    """The panel with an intercept column, which is never penalized."""
    return np.hstack([np.ones((panel.shape[0], 1), dtype=np.float64), panel])


def _depth_design(library: np.ndarray) -> np.ndarray:
    """Intercept and log library size: one predictor, for the depth baseline."""
    return np.hstack([
        np.ones((library.shape[0], 1), dtype=np.float64),
        np.log1p(library.astype(np.float64))[:, None],
    ])


class Normals:
    """`XᵀX` and `XᵀY`, accumulated over blocks of cells."""

    def __init__(self, n_features: int, n_outputs: int) -> None:
        self.xtx = np.zeros((n_features, n_features), dtype=np.float64)
        self.xty = np.zeros((n_features, n_outputs), dtype=np.float64)
        self.n = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.n += x.shape[0]

    def solve(self, penalty: float) -> np.ndarray:
        """Ridge coefficients. The intercept (column 0) is not penalized."""
        ridge = np.eye(self.xtx.shape[0]) * (penalty * max(self.n, 1))
        ridge[0, 0] = 0.0
        try:
            return np.linalg.solve(self.xtx + ridge, self.xty)
        except np.linalg.LinAlgError:
            # A singular system at penalty 0 is information, not a failure:
            # the least-squares fit is then not identified and the penalized
            # fits below are the answer.
            return np.linalg.lstsq(self.xtx + ridge, self.xty, rcond=None)[0]


def score(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """The two readings, in the same form Phase 2 reports them."""
    from ..train.loop import pearson

    residual = float(np.mean((prediction - truth) ** 2))
    no_change = float(np.mean(truth**2))
    return {
        "mse": residual,
        "mse_no_change": no_change,
        "mse_ratio_to_no_change": residual / max(no_change, 1e-12),
        "pearson": pearson(prediction, truth),
    }


def split_half_reliability(
    counts: np.ndarray,
    library: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    is_panel: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """How well a cell's rest genes correlate with themselves, and what that
    permits a predictor to reach.

    Each gene's count is split binomially at p = 0.5 and each half normalized
    at half the library size, which leaves CP10K on the original scale in
    expectation. The correlation between the halves is the **reliability** `r`
    at half depth; Spearman–Brown, `2r / (1 + r)`, lifts it to the full depth
    the model sees. `r` is the share of the target's variance that is
    reproducible signal rather than sampling noise.

    **`r` is not the ceiling on a predictor.** Two fallible measurements of the
    same quantity correlate `r` with each other, but a *noiseless* predictor of
    the underlying signal correlates `sqrt(r)` with either of them — the
    classical attenuation bound. Reporting `r` as the ceiling understates it
    badly: at `r = 0.086` the ceiling is 0.29, not 0.086, and the first server
    run produced a ridge regression at 0.14 that would have looked impossible
    (DECISIONS.md D101). The two derived bounds are therefore both reported:

    * `max_achievable_pearson` = `sqrt(r)` — no predictor correlates higher;
    * `best_possible_mse_ratio` = `1 - r` — no predictor's mse against the
      no-change baseline goes lower, because the noise cannot be predicted.

    One approximation: `log1p` is not linear, so thinning the counts and then
    normalizing is not exactly a linear split of the model's target. At ~11,000
    UMIs over ~7,000 rest genes most counts are 0 or 1, where `log1p` is very
    nearly linear, so the error is small — but it is an approximation and the
    bound should be read as indicative rather than exact.
    """
    from data_prep.phase_data import log1p_cp10k

    first = rng.binomial(counts, 0.5)
    second = counts - first
    half_library = library.astype(np.float64) / 2.0

    a = _z(log1p_cp10k(first, half_library), mean, std)[:, ~is_panel]
    b = _z(log1p_cp10k(second, half_library), mean, std)[:, ~is_panel]

    from ..train.loop import pearson

    half = pearson(a, b)
    if not np.isfinite(half):
        reliability = float("nan")
    else:
        reliability = 2 * half / (1 + half)
    usable = np.isfinite(reliability) and reliability > 0
    return {
        "half_depth_pearson": half,
        # The reliability at full depth: the share of the target's variance
        # that is signal. Spearman-Brown, 2r/(1+r).
        "reliability": reliability,
        # What that permits. sqrt for a correlation, 1-r for the error: a
        # predictor can match the signal exactly and still not touch the noise.
        "max_achievable_pearson": float(np.sqrt(reliability)) if usable else float("nan"),
        "best_possible_mse_ratio": 1.0 - reliability if np.isfinite(reliability) else float("nan"),
        "note": "reliability is Spearman-Brown 2r/(1+r); the ceiling on a "
        "predictor's correlation is its square root, not itself (D101)",
    }


def _blocks(rows: np.ndarray, size: int):
    for start in range(0, rows.size, size):
        yield rows[start : start + size]


def _read(context, rows: np.ndarray, library_by_row: dict) -> tuple[np.ndarray, np.ndarray]:
    """`(counts, library)` for `rows`, over this screen's measured genes."""
    counts = context.raw_counts(rows)
    library = np.array([library_by_row[int(row)] for row in rows], dtype=np.float64)
    return counts, library


def measure_context(
    context,
    cfg: Config,
    n_fit: int,
    n_eval: int,
    penalties=PENALTIES,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """The ceilings for one screen."""
    from data_prep.phase_data import log1p_cp10k

    log = get_logger()
    rng = rng or np.random.default_rng(cfg.seed)

    control = context.cells.loc[context.cells["is_control"]]
    rows = control["row"].to_numpy()
    if rows.size < MIN_EVAL_CELLS * 2:
        return {
            "measured": False,
            "reason": f"{rows.size} control cells is too few to fit and evaluate on",
            "n_control_cells": int(rows.size),
        }

    library_by_row = dict(
        zip(control["row"].to_numpy().tolist(), control["library_size"].to_numpy().tolist())
    )
    shuffled = rng.permutation(rows)
    evaluate = shuffled[: min(n_eval, max(MIN_EVAL_CELLS, rows.size // 5))]
    fit = shuffled[evaluate.size : evaluate.size + n_fit]

    mean, std = context.control_stats()
    is_panel = (context.genes["split"] == "panel").to_numpy()

    log.info(
        "  %s: fitting on %d control cells, evaluating on %d (%d panel -> %d rest)",
        context.name, fit.size, evaluate.size, int(is_panel.sum()), int((~is_panel).sum()),
    )

    panel_normals = Normals(int(is_panel.sum()) + 1, int((~is_panel).sum()))
    depth_normals = Normals(2, int((~is_panel).sum()))
    for block in _blocks(fit, BLOCK_CELLS):
        counts, library = _read(context, block, library_by_row)
        values = _z(log1p_cp10k(counts, library), mean, std)
        panel_normals.add(_design(values[:, is_panel]), values[:, ~is_panel])
        depth_normals.add(_depth_design(library), values[:, ~is_panel])
        del counts, values

    eval_counts, eval_library = _read(context, evaluate, library_by_row)
    eval_values = _z(log1p_cp10k(eval_counts, eval_library), mean, std)
    eval_panel = _design(eval_values[:, is_panel])
    eval_depth = _depth_design(eval_library)
    eval_rest = eval_values[:, ~is_panel]

    ridge = {}
    for penalty in penalties:
        coefficients = panel_normals.solve(penalty)
        ridge[f"{penalty:g}"] = score(eval_panel @ coefficients, eval_rest)
        log.info(
            "    penalty %-6g mse/no-change %.4f  pearson %+.4f",
            penalty,
            ridge[f"{penalty:g}"]["mse_ratio_to_no_change"],
            ridge[f"{penalty:g}"]["pearson"],
        )

    depth = score(eval_depth @ depth_normals.solve(0.0), eval_rest)
    log.info(
        "    library size alone   mse/no-change %.4f  pearson %+.4f",
        depth["mse_ratio_to_no_change"], depth["pearson"],
    )

    reliability = split_half_reliability(
        eval_counts, eval_library, mean, std, is_panel, rng
    )
    log.info(
        "    target reliability   %.4f of the variance is signal; a predictor can "
        "reach pearson %+.4f and mse/no-change %.4f at best",
        reliability["reliability"],
        reliability["max_achievable_pearson"],
        reliability["best_possible_mse_ratio"],
    )

    best = max(ridge, key=lambda key: ridge[key]["pearson"])
    at_edge = float(best) == max(float(key) for key in ridge)
    if at_edge:
        log.warning(
            "    %s: the best penalty is the largest offered (%s), so the linear "
            "ceiling is a lower bound — widen PENALTIES and rerun",
            context.name, best,
        )
    return {
        "measured": True,
        "n_control_cells": int(rows.size),
        "n_fit_cells": int(fit.size),
        "n_eval_cells": int(evaluate.size),
        "n_panel": int(is_panel.sum()),
        "n_rest": int((~is_panel).sum()),
        "ridge": ridge,
        "best_penalty": best,
        # A ceiling chosen at the edge of the grid is a statement about the
        # grid. Reported so a reader does not take it for the ceiling.
        "best_at_grid_edge": bool(at_edge),
        "linear_ceiling": ridge[best],
        "depth_only": depth,
        "target_reliability": reliability,
    }


def model_numbers(cfg) -> dict[str, Any]:
    """Phase 2's own mapping numbers, per screen, if the run has them.

    Read rather than recomputed: the point is to put the ceiling beside the
    number it is a ceiling for, and that number is already in the artifact.
    """
    from ..paths import RunPaths

    path = RunPaths(cfg).phase_metrics("phase2")
    if not path.is_file():
        return {"available": False, "reason": f"{path} has not been written"}

    metrics = json.loads(path.read_text())
    per_context = metrics.get("validation_per_context") or {}
    return {
        "available": bool(per_context),
        "describes": metrics.get("validation_per_context_describes"),
        "per_context": {
            name: {
                key: values[key]
                for key in (
                    "map_control_mse",
                    "map_control_no_change",
                    "map_control_ratio_to_no_change",
                    "map_control_pearson",
                )
                if key in values
            }
            for name, values in per_context.items()
        },
    }


def verdict(contexts: dict[str, Any], model: dict[str, Any]) -> tuple[str, str]:
    """Which branch of PLAN_MAPPING §B the measurement chooses.

    * `model-underperforms` — a linear fit beats the gene-token model by more
      than `LINEAR_MARGIN`. The task is doable and the model is not doing it:
      §B3, and `phase2/curves.csv` says which part.
    * `not-predictable-per-cell` — the linear fit is no better than the model
      *and* the target is barely reliable. A single cell's rest genes are
      mostly sampling noise, so §B2's reframe is the answer rather than a
      fallback.
    * `linear-is-no-better` — the linear fit adds nothing but the target is
      reliable, so the information is there and neither model reaches it. That
      is a harder, more interesting case and belongs to §B3 as well.
    """
    measured = {name: entry for name, entry in contexts.items() if entry.get("measured")}
    if not measured:
        return "not-measured", "no screen had enough control cells"

    linear = {name: entry["linear_ceiling"]["pearson"] for name, entry in measured.items()}
    ceiling = {
        name: entry["target_reliability"]["max_achievable_pearson"]
        for name, entry in measured.items()
    }
    available = (model or {}).get("per_context") or {}
    theirs = {
        name: available[name]["map_control_pearson"]
        for name in measured
        if name in available and "map_control_pearson" in available[name]
    }

    if not theirs:
        return "no-model-to-compare", (
            "the ceilings are measured but `phase2/metrics.json` has no "
            "`map_control_pearson` to read them against: run the phase2 stage in "
            f"this run directory first. Linear ceiling per screen: {linear}."
        )

    gaps = {name: linear[name] - theirs[name] for name in theirs}
    highest_ceiling = max(v for v in ceiling.values() if np.isfinite(v)) if ceiling else 0.0
    share = {
        name: (theirs[name] / ceiling[name], linear[name] / ceiling[name])
        for name in theirs
        if np.isfinite(ceiling.get(name, float("nan"))) and ceiling[name] > 0
    }
    shares = ", ".join(
        f"{name} {100 * model:.0f}% against the linear fit's {100 * ridge:.0f}%"
        for name, (model, ridge) in sorted(share.items())
    )
    edge = sorted(name for name, entry in measured.items() if entry.get("best_at_grid_edge"))
    caveat = (
        f" (the best penalty is at the edge of the grid for {', '.join(edge)}, so "
        "their linear ceiling is a lower bound)"
        if edge
        else ""
    )
    if max(gaps.values()) > LINEAR_MARGIN:
        ahead = max(gaps, key=gaps.get)
        return "model-underperforms", (
            f"a ridge regression from the panel beats the gene-token model on "
            f"{ahead} by {gaps[ahead]:+.3f} pearson ({linear[ahead]:+.3f} against "
            f"{theirs[ahead]:+.3f}), and by {gaps} across the screens measured. "
            f"Against the ceiling the target's reliability permits, the model "
            f"reaches {shares}. The information is in the panel, a linear map "
            "extracts a good share of it and the model extracts a fraction, so this "
            "is §B3: read phase2/curves.csv for which of the three loss terms is "
            "winning, then the loss weights and the latent width." + caveat
        )
    if highest_ceiling < MIN_ACHIEVABLE_PEARSON:
        return "not-predictable-per-cell", (
            "neither the model nor a ridge regression predicts a single cell's rest "
            f"genes (ridge {linear}, model {theirs}), and the target has almost no "
            f"signal to predict: the highest correlation the reliability permits is "
            f"{highest_ceiling:.3f}, against a threshold of "
            f"{MIN_ACHIEVABLE_PEARSON}. At this sequencing depth most of those genes "
            "are at zero or one count, so the quantity Phase 2 is asked for is "
            "mostly sampling noise. No architecture fixes that. §B2 — predict one "
            "rest-gene fold change per (context, target) and take cell-to-cell "
            "variation from the control resample, which is all the six metrics ever "
            "score." + caveat
        )
    return "linear-is-no-better", (
        f"a linear fit does not beat the model (ridge {linear}, model {theirs}), but "
        f"the target has signal to predict — up to pearson {highest_ceiling:.3f} — "
        f"and neither reaches it ({shares}). That is §B3 with the capacity questions "
        "ahead of the loss weights: 64 latents carry every cell's panel through to "
        "thousands of gene queries." + caveat
    )


def run_mapping_ceiling(
    cfg: Config,
    screen: str | None = None,
    n_fit: int = 20000,
    n_eval: int = 2048,
    penalties=PENALTIES,
) -> dict[str, Any]:
    """The diagnostic, over one screen or all of them."""
    from ..data import replogle as replogle_data
    from ..paths import DataPaths, RunPaths

    log = get_logger()
    data_paths = DataPaths(cfg)
    run_paths = RunPaths(cfg)

    discovered = data_paths.discover_phase2_contexts()
    if screen is not None:
        if screen not in discovered:
            raise ValueError(
                f"no screen {screen!r} under {data_paths.phase2_dir}; found "
                f"{sorted(discovered)}"
            )
        discovered = {screen: discovered[screen]}
    if not discovered:
        raise FileNotFoundError(f"no Replogle screens under {data_paths.phase2_dir}")

    log.info("mapping ceiling: %s", ", ".join(discovered))
    contexts = {}
    for name, directory in discovered.items():
        context = replogle_data.load_replogle_context(directory)
        contexts[name] = measure_context(
            context, cfg, n_fit=n_fit, n_eval=n_eval, penalties=penalties
        )

    model = model_numbers(cfg)
    decision, reading = verdict(contexts, model)
    report = {
        "description": "what a linear map reaches from the panel, and how "
        "reliable a single cell's rest genes are (PLAN_MAPPING.md §B1)",
        "penalties": [f"{p:g}" for p in penalties],
        "n_fit_requested": n_fit,
        "n_eval_requested": n_eval,
        "screens": contexts,
        "model": model,
        "verdict": decision,
        "reading": reading,
        "thresholds": {
            "linear_margin": LINEAR_MARGIN,
            "min_achievable_pearson": MIN_ACHIEVABLE_PEARSON,
        },
    }

    log.info("")
    log.info("verdict: %s", decision.upper())
    log.info("  %s", reading)

    path = run_paths.reports / "mapping_ceiling.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    log.info("written -> %s", path)
    return report
