"""Figures for the proposal defense (CLAUDE.md §5).

One message per figure, large type, labelled axes with units, and a legend
whenever more than one series is on screen. Each function reads artifacts a
previous stage already wrote and returns the path it drew, or `None` with a
reason when the input it needs is not there — the summary then says so
instead of showing an empty panel.

Colours are the first three slots of a categorical palette validated for
colour-vision deficiency (worst all-pairs ΔE 9.2 CVD / 24.0 normal). The
aqua slot sits below 3:1 against the surface, so every figure that uses it
also carries direct value labels, and `summary.md` repeats every number as a
table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

#: Categorical slots 1–3, fixed order, never cycled.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"

ARMS = ("method", "upper_bound", "floor")
ARM_LABELS = {"method": "method", "upper_bound": "upper bound", "floor": "no-change floor"}


def style(cfg):
    """Projector-legible defaults, applied to every figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": cfg.report.figure_dpi,
            "savefig.dpi": cfg.report.figure_dpi,
            "font.size": cfg.report.font_size,
            "axes.titlesize": cfg.report.font_size + 4,
            "axes.labelsize": cfg.report.font_size + 1,
            "axes.edgecolor": MUTED,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
        }
    )
    return plt


def _finish(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    fig.clf()
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


#: A 2px surface-coloured edge is the gap between adjacent fills.
BAR_EDGE = {"edgecolor": SURFACE, "linewidth": 2.0}


def _bar_labels(ax, bars, fmt="{:.2f}"):
    """Direct value labels — the relief the palette's contrast WARN requires."""
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 5 if height >= 0 else -5),
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=plt_fontsize() - 4,
            color=MUTED,
        )
    ax.margins(y=0.18)


def plt_fontsize() -> int:
    import matplotlib.pyplot as plt

    return int(plt.rcParams["font.size"])


# --------------------------------------------------------------------------- #
# rehearsal
# --------------------------------------------------------------------------- #
def rehearsal_official(cfg, report: dict[str, Any], path: Path):
    """The end-to-end variant, each arm against the no-change point."""
    from ..rehearsal import common

    plt = style(cfg)
    rows = []
    for entry in report.get("variants", []):
        if entry["variant"] != "cross_context":
            continue
        values = {}
        for arm in ARMS:
            scored = entry["arms"].get(arm, {}).get(common.END_TO_END, {})
            values[arm] = scored.get("normalized", {}).get("objective")
        if any(v is not None for v in values.values()):
            rows.append((f"{entry['context']}\n({entry.get('n_targets', '?')} targets)", values))

    if not rows:
        return None, "the cross-context variant produced no scored arm"

    fig, ax = plt.subplots(figsize=(max(8, 3.4 * len(rows)), 5.4))
    x = np.arange(len(rows))
    width = 0.26
    for i, arm in enumerate(ARMS):
        heights = [r[1].get(arm) if r[1].get(arm) is not None else np.nan for r in rows]
        bars = ax.bar(x + (i - 1) * width, heights, width, color=SERIES[i],
                      label=ARM_LABELS[arm], **BAR_EDGE)
        _bar_labels(ax, bars, "{:+.3f}")
    ax.axhline(0.0, color=MUTED, linewidth=1.5)
    ax.annotate("predicting no change", (0.01, 0.0), xycoords=("axes fraction", "data"),
                fontsize=plt_fontsize() - 3, color=MUTED, va="bottom")
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("mean of the six metrics\n(0 = no change, 1 = perfect)")
    ax.set_title(
        "Cross-context rehearsal: whole predicted cells,\nscored as the challenge scores them",
        loc="left",
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    return _finish(fig, path), None


def rehearsal_per_gene(cfg, report: dict[str, Any], path: Path):
    """Variants 1 and 3, on the genes each of them predicts."""
    plt = style(cfg)
    rows = []
    for entry in report.get("variants", []):
        for key in ("rest_genes", "hidden_genes"):
            if key not in entry["arms"].get("method", {}):
                continue
            values = {
                arm: entry["arms"].get(arm, {}).get(key, {}).get("mse_ratio_to_no_change")
                for arm in ARMS
            }
            rows.append((f"{entry['variant']}\n{entry['context']}", values))

    if not rows:
        return None, "neither partial variant reported per-gene scores"

    fig, ax = plt.subplots(figsize=(max(8, 3.4 * len(rows)), 5.4))
    x = np.arange(len(rows))
    width = 0.26
    for i, arm in enumerate(ARMS):
        heights = [r[1].get(arm) if r[1].get(arm) is not None else np.nan for r in rows]
        bars = ax.bar(x + (i - 1) * width, heights, width, color=SERIES[i],
                      label=ARM_LABELS[arm], **BAR_EDGE)
        _bar_labels(ax, bars, "{:.2f}")
    ax.axhline(1.0, color=MUTED, linewidth=1.5, linestyle="--")
    ax.annotate("no change", (0.01, 1.0), xycoords=("axes fraction", "data"),
                fontsize=plt_fontsize() - 3, color=MUTED, va="bottom")
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("error ÷ no-change error\n(lower is better)")
    ax.set_title(
        "Partial rehearsal variants:\nthe genes each one predicts", loc="left"
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    return _finish(fig, path), None


def rehearsal_scatter(cfg, report: dict[str, Any], path: Path):
    """Predicted against measured change, for a few held-out targets."""
    plt = style(cfg)
    panels = []
    for entry in report.get("variants", []):
        predictions = entry.get("predictions") or {}
        truth = entry.get("truth") or {}
        for target in sorted(set(predictions) & set(truth)):
            panels.append((entry["context"], target,
                           np.asarray(predictions[target]), np.asarray(truth[target])))
    panels = panels[: cfg.report.max_scatter_targets]
    if not panels:
        return None, "the rehearsal stored no per-target predictions with their truth"

    # Small multiples share one scale, or the panels cannot be compared.
    everything = np.concatenate([np.abs(np.concatenate([t, p])) for _, _, p, t in panels])
    limit = max(float(np.nanpercentile(everything, 99.5)), 0.1)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), squeeze=False)
    for ax, (context, target, predicted, true) in zip(axes[0], panels):
        ax.scatter(true, predicted, s=10, alpha=0.4, color=SERIES[0], edgecolors="none")
        ax.plot([-limit, limit], [-limit, limit], color=MUTED, linewidth=1.2, linestyle="--")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_title(f"{target} ({context})")
        ax.set_xlabel("measured log2 fold change")
        ax.grid(True)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("predicted log2 fold change")
    fig.suptitle("Held-out targets: predicted against measured change, gene by gene",
                 fontsize=plt_fontsize() + 4)
    return _finish(fig, path), None


# --------------------------------------------------------------------------- #
# forgetting, Phase 3, knockdown, curves
# --------------------------------------------------------------------------- #
def forgetting(cfg, report: dict[str, Any], path: Path):
    """Phase 1's validation score under each saved core."""
    plt = style(cfg)
    metric = report.get("headline_metric", "sig_pearson")
    rows = [
        (name, (entry.get("metrics") or {}).get(metric))
        for name, entry in (report.get("checkpoints") or {}).items()
    ]
    rows = [(n, v) for n, v in rows if v is not None and np.isfinite(v)]
    if not rows:
        return None, "reports/forgetting.json holds no scored checkpoint"

    fig, ax = plt.subplots(figsize=(max(6, 2.0 * len(rows)), 4.8))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows], color=SERIES[0],
                  width=0.55, **BAR_EDGE)
    _bar_labels(ax, bars, "{:+.4f}")
    reference = rows[0][1]
    ax.axhline(reference, color=MUTED, linewidth=1.2, linestyle="--")
    ax.annotate("Phase 1 itself", (0.01, reference), xycoords=("axes fraction", "data"),
                fontsize=plt_fontsize() - 3, color=MUTED, va="bottom")
    ax.set_ylabel(f"Phase 1 held-out {metric.replace('_', ' ')}")
    ax.set_xlabel("core the score was measured under")
    ax.set_title(
        "Guard against forgetting:\nPhase 1's validation after each later phase", loc="left"
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    return _finish(fig, path), None


def phase3_adaptation(cfg, metrics: dict[str, Any], path: Path):
    """What adapting on each context's controls did to the mapping."""
    plt = style(cfg)
    contexts = metrics.get("contexts", {})
    if not contexts:
        return None, "phase3/metrics.json holds no contexts"

    names = sorted(contexts)
    before = [contexts[n].get("loss_before", np.nan) for n in names]
    after = [contexts[n].get("loss_after", np.nan) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, 2.0 * len(names)), 4.8))
    x = np.arange(len(names))
    width = 0.34
    bars_before = ax.bar(x - width / 2, before, width, color=SERIES[0], label="before",
                         **BAR_EDGE)
    bars_after = ax.bar(x + width / 2, after, width, color=SERIES[1], label="after",
                        **BAR_EDGE)
    _bar_labels(ax, bars_before, "{:.3f}")
    _bar_labels(ax, bars_after, "{:.3f}")
    ax.set_xticks(x, names)
    ax.set_xlabel("challenge context")
    ax.set_ylabel("panel → rest error on control cells\n(mean squared error, control-SD units)")
    ax.set_title("Phase 3: adapting on each context's control cells", loc="left")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    return _finish(fig, path), None


def knockdown(cfg, records: list[dict[str, Any]], sources: dict[tuple, str], path: Path):
    """Each target gene's own predicted level against its control level."""
    plt = style(cfg)
    usable = [r for r in records if r.get("control_mean_cpm", 0) > 0]
    if not usable:
        return None, "no target gene is expressed in its context's controls"

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    groups: dict[str, list] = {}
    for record in usable:
        source = sources.get((record["context"], record["target_gene"]), "pooled")
        kind = "per-target fold_expr" if source.startswith("per_target") else "pooled median"
        groups.setdefault(kind, []).append(record)

    for i, (kind, rows) in enumerate(sorted(groups.items())):
        ax.scatter(
            [r["control_mean_cpm"] for r in rows],
            [max(r["predicted_mean_cpm"], 1e-3) for r in rows],
            s=46, alpha=0.85, color=SERIES[i % len(SERIES)], edgecolors=SURFACE,
            linewidths=1.2, label=f"{kind} (n={len(rows)})",
        )

    levels = [r["control_mean_cpm"] for r in usable]
    span = np.array([max(min(levels) * 0.5, 1e-3), max(levels) * 2])
    ax.plot(span, span, color=MUTED, linestyle="--", linewidth=1.2, label="no change")
    ax.plot(span, span * cfg.sanity.knockdown_max_ratio, color=MUTED, linestyle=":",
            linewidth=1.2, label=f"{cfg.sanity.knockdown_max_ratio:g}× control (check 3)")
    ax.axvline(cfg.sanity.knockdown_min_control_cpm, color=MUTED, linewidth=1.0, alpha=0.45)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.annotate(
        "expression floor →", (cfg.sanity.knockdown_min_control_cpm, 0.02),
        xycoords=("data", "axes fraction"), fontsize=plt_fontsize() - 4,
        color=MUTED, va="bottom", ha="right",
    )
    ax.set_xlabel("control mean expression of the target gene (CP10K)")
    ax.set_ylabel("predicted mean expression (CP10K)")
    ax.set_title("Is each target's own gene down?", loc="left")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    return _finish(fig, path), None


def training_curves(cfg, curves: dict[str, Any], path: Path):
    """Phase 1 and Phase 2 training loss, step by step."""
    plt = style(cfg)
    present = {name: frame for name, frame in curves.items() if frame is not None and len(frame)}
    if not present:
        return None, "no phase wrote a curve"

    fig, axes = plt.subplots(1, len(present), figsize=(5.4 * len(present), 4.4), squeeze=False)
    for ax, (name, frame) in zip(axes[0], sorted(present.items())):
        ax.plot(frame["step"], frame["loss"], color=SERIES[0], label="training loss")
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.grid(True)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("loss")
    fig.suptitle("Training curves", fontsize=plt_fontsize() + 4)
    return _finish(fig, path), None
