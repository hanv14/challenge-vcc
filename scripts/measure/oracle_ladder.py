#!/usr/bin/env python
"""Measurement 3: the oracle ladder on one held-out Replogle screen.

    python scripts/measure/oracle_ladder.py --config configs/server.yaml \
        --held-out rpe1 --source K562_gwps --n-targets 300

The leaderboard's 0 is an oracle (docs/metrics.md §6a): the held-out
context's own mean perturbation response, from the hidden truth. This places
no-model predictions built from real data on that scale, all through the
shipped count-space generator, so the gaps between rungs say what each kind
of knowledge is worth in a cell line the predictor never saw perturbed:

    floor          predict no change
    own_mean       the held-out screen's own mean response, leave-target-out (x1, x2)
    own_mean_b     own_mean x1 again with a different control draw: generator noise
    src_mean       the source screen's mean response over the same targets (x1, x2)
    src_lookup     the source screen's own change for each target, as is
    own_mean+src_res  own mean + a x (source target-specific residual), a = 0.5, 1
    src_mean+src_res  source mean + 0.5 x source residual
    own_target     the held-out screen's own change per target: the generator's
                   ceiling (optimistic, it shares the truth's sampling noise)

A change is log2 of the ratio of per-gene mean CP10K, target cells over
control cells, both floored at `predict.cpm_floor` — the ratio the generator
applies. Every arm is generated with threshold 0 and scale 1, from the same
control draw per target (D7), and scored with the six official metrics on the
screen's own 0 = baseline / 1 = replicate scale, read on `from_replicate`
(D120). A target bootstrap gives each arm's standard error and each arm's
paired difference from `own_mean`.

Measurement only: trains nothing, writes no checkpoint, changes no pipeline
state. CPU only. Writes `<output_root>/<run-name>/measure/` and prints a table.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import os  # noqa: E402

from vccp.runtime import set_thread_env  # noqa: E402

set_thread_env()
# cell-eval2's per-target progress bars would bury the table this prints.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402

from vccp.config import ConfigError, apply_overrides, load_config  # noqa: E402
from vccp.data.replogle import load_replogle_context  # noqa: E402
from vccp.eval import official, scale  # noqa: E402
from vccp.logging_utils import get_logger, setup_logging  # noqa: E402
from vccp.paths import DataPaths  # noqa: E402
from vccp.predict.generator import GeneratorSettings  # noqa: E402
from vccp.rehearsal import common  # noqa: E402
from vccp.runtime import seed_everything  # noqa: E402
from vccp.sanity.checks import mean_cp10k  # noqa: E402

PERT_COL = "target_gene"
SHORT = dict(zip(official.SCORED, ("pds", "expr", "nmae", "fid", "reach", "jac")))
EXPR = "expr_mse_unbiased_capped_norm"
EXPR_NUM, EXPR_DEN = "expr_mse_unbiased_capped", "expr_distance_unbiased"


def rows_of(context, target: str, cap: int, rng) -> np.ndarray:
    rows = context.cells.loc[context.cells["target_gene"].astype(str) == target, "row"].to_numpy()
    if cap and rows.size > cap:
        rows = np.sort(rng.choice(rows, cap, replace=False))
    return rows


def log2fc(target_counts: np.ndarray, control_cp10k: np.ndarray, floor: float) -> np.ndarray:
    return np.log2(
        np.maximum(mean_cp10k(target_counts), floor) / np.maximum(control_cp10k, floor)
    ).astype(np.float32)


def bootstrap(per_target: dict[str, dict[str, float]], targets: list[str], ends, n_boot, rng):
    """Resample targets; recompute each member's panel value and its score.

    Five members are means over targets (NaN skipped, as cell-eval2 does);
    expression is a ratio of sums of its two per-target legs. The ends of the
    scale are held fixed, so this is the spread due to which targets are in
    the panel, not due to the anchors. The expression leg's panel-level
    budget factor (#348) is not resampled.
    """
    from cell_eval2.scoring import score_one

    base, policy = ends
    index = np.arange(len(targets))
    table = {
        m: np.array([per_target.get(m, {}).get(t, np.nan) for t in targets], dtype=float)
        for m in [*official.SCORED, EXPR_NUM, EXPR_DEN]
    }
    have_expr = np.isfinite(table[EXPR_NUM]).any() and np.isfinite(table[EXPR_DEN]).any()
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(index, index.size, replace=True)
        scores = []
        for m in official.SCORED:
            if m == EXPR:
                if not have_expr:
                    continue
                value = np.nansum(table[EXPR_NUM][pick]) / np.nansum(table[EXPR_DEN][pick])
            else:
                value = np.nanmean(table[m][pick])
            scores.append(score_one(float(value), base[m], policy[m]))
        draws[b] = np.mean(scores)
    return draws, have_expr


def per_target_values(result) -> dict[str, dict[str, float]]:
    frame = result.per_perturbation
    out: dict[str, dict[str, float]] = {}
    if frame is None:
        return out
    for row in frame.iter_rows(named=True):
        value = row.get("value")
        if value is None:
            continue
        out.setdefault(str(row["metric"]), {})[str(row["perturbation"])] = float(value)
    return out


def scale_ends(reference):
    """Each member's baseline value and its replicate-anchored policy, from the bundle."""
    import polars as pl
    from cell_eval2.catalog import CATALOG

    root = Path(reference.bundle_dir)
    base_row = pl.read_csv(root / "baseline_agg.csv").filter(pl.col("statistic") == "mean")
    anchor = pl.read_parquet(root / "anchor_agg.parquet")
    replicate = {r["metric"]: float(r["replicate"]) for r in anchor.iter_rows(named=True)}
    base = {m: float(base_row[m][0]) for m in official.SCORED}
    policy = {
        m: dataclasses.replace(
            CATALOG[m].scoring, anchor=replicate[m], allow_negative_baseline=False
        )
        for m in official.SCORED
    }
    return base, policy, replicate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", default="measure")
    parser.add_argument("--held-out", required=True, help="the screen treated as a new context")
    parser.add_argument("--source", required=True, help="the screen knowledge comes from")
    parser.add_argument("--n-targets", type=int, default=300)
    parser.add_argument("--min-cells", type=int, default=30,
                        help="cells a target needs in both screens to enter the panel")
    parser.add_argument("--max-cells", type=int, default=0,
                        help="cap on real cells per target in the held-out truth (0 = all)")
    parser.add_argument("--source-cells", type=int, default=200,
                        help="cap on source cells read per target")
    parser.add_argument("--control-cells", type=int, default=0,
                        help="held-out control cells (0 = rehearsal.max_control_cells)")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--arms", default="",
                        help="comma-separated subset of arms to score (default: all)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        cfg = apply_overrides(load_config(args.config), args.set)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    cfg = dataclasses.replace(cfg, run_name=args.run_name)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    outdir = Path(cfg.output_root) / cfg.run_name / "measure"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"oracle_ladder_{args.held_out}_from_{args.source}"
    setup_logging(outdir / f"{stem}.log")
    log = get_logger()
    seed_everything(cfg.seed)
    if not official.available():
        print("cell-eval2 is not installed; this measurement needs it", file=sys.stderr)
        return 2

    contexts = DataPaths(cfg).discover_phase2_contexts()
    for name in (args.held_out, args.source):
        if name not in contexts:
            print(f"no Replogle screen {name!r}; found {sorted(contexts)}", file=sys.stderr)
            return 2
    held = load_replogle_context(contexts[args.held_out])
    src = load_replogle_context(contexts[args.source])
    rng = np.random.default_rng(cfg.seed)
    floor = cfg.predict.cpm_floor

    # ---- the panel: random shared targets with enough cells on both sides
    n_held, n_src = held.n_cells_per_target(), src.n_cells_per_target()
    shared = sorted(
        t for t in set(held.targets) & set(src.targets)
        if n_held.get(t, 0) >= args.min_cells and n_src.get(t, 0) >= args.min_cells
    )
    if len(shared) < 3:
        print(f"only {len(shared)} shared targets with >= {args.min_cells} cells", file=sys.stderr)
        return 1
    targets = sorted(rng.choice(shared, min(args.n_targets, len(shared)), replace=False).tolist())
    log.info("panel: %d of %d shared targets (random, seed %d), held out %s, source %s",
             len(targets), len(shared), cfg.seed, args.held_out, args.source)

    # ---- the truth: held-out targets' real cells, plus controls
    t0 = time.time()
    n_controls = args.control_cells or cfg.rehearsal.max_control_cells
    control_rows = held.cells.loc[held.cells["is_control"].astype(bool), "row"].to_numpy()
    if n_controls and control_rows.size > n_controls:
        control_rows = np.sort(rng.choice(control_rows, n_controls, replace=False))
    control_counts = held.raw_counts(control_rows).astype(np.int32)
    held_rows = {t: rows_of(held, t, args.max_cells, rng) for t in targets}
    all_rows = np.concatenate([held_rows[t] for t in targets])
    block = held.raw_counts(all_rows).astype(np.int32)
    real_labels = [t for t in targets for _ in range(held_rows[t].size)]
    cells_per_target = {t: int(held_rows[t].size) for t in targets}
    gene_names = held.genes["gene_symbol"].astype(str).tolist()
    log.info("truth: %d perturbed cells, %d controls, %d genes (%.0fs)",
             block.shape[0], control_counts.shape[0], len(gene_names), time.time() - t0)

    control_cp10k = mean_cp10k(control_counts)
    labels = np.asarray(real_labels)
    own = {t: log2fc(block[labels == t], control_cp10k, floor) for t in targets}

    # ---- the source screen's changes, on the held-out gene axis
    t0 = time.time()
    src_genes = src.genes["gene_symbol"].astype(str).tolist()
    position = {g: i for i, g in enumerate(src_genes)}
    shared_genes = np.array([position.get(g, -1) for g in gene_names])
    on_axis = shared_genes >= 0
    src_control_rows = src.cells.loc[src.cells["is_control"].astype(bool), "row"].to_numpy()
    if src_control_rows.size > n_controls:
        src_control_rows = np.sort(rng.choice(src_control_rows, n_controls, replace=False))
    src_control_cp10k = mean_cp10k(src.raw_counts(src_control_rows))
    src_rows = {t: rows_of(src, t, args.source_cells, rng) for t in targets}
    src_block = src.raw_counts(np.concatenate([src_rows[t] for t in targets]))
    src_labels = np.asarray([t for t in targets for _ in range(src_rows[t].size)])
    source = {}
    for t in targets:
        values = log2fc(src_block[src_labels == t], src_control_cp10k, floor)
        mapped = np.zeros(len(gene_names), dtype=np.float32)
        mapped[on_axis] = values[shared_genes[on_axis]]
        source[t] = mapped
    del src_block
    log.info("source: %d of %d held-out genes measured in %s (%.0fs)",
             int(on_axis.sum()), len(gene_names), args.source, time.time() - t0)

    # ---- the arms
    own_sum = np.sum([own[t] for t in targets], axis=0)
    n = len(targets)
    own_mean = {t: (own_sum - own[t]) / (n - 1) for t in targets}  # leave-target-out
    src_mean_vec = np.mean([source[t] for t in targets], axis=0)
    src_res = {t: source[t] - src_mean_vec for t in targets}
    zero = np.zeros(len(gene_names), dtype=np.float32)
    arms = {
        "floor": ({t: zero for t in targets}, "ladder"),
        "own_mean": (own_mean, "ladder"),
        "own_mean_b": (own_mean, "ladder_b"),
        "own_mean_x2": ({t: 2 * own_mean[t] for t in targets}, "ladder"),
        "src_mean": ({t: src_mean_vec for t in targets}, "ladder"),
        "src_mean_x2": ({t: 2 * src_mean_vec for t in targets}, "ladder"),
        "src_lookup": (source, "ladder"),
        "own_mean+0.5src_res": ({t: own_mean[t] + 0.5 * src_res[t] for t in targets}, "ladder"),
        "own_mean+1.0src_res": ({t: own_mean[t] + src_res[t] for t in targets}, "ladder"),
        "src_mean+0.5src_res": ({t: src_mean_vec + 0.5 * src_res[t] for t in targets}, "ladder"),
        "own_target": (own, "ladder"),
    }
    if args.arms:
        wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = sorted(set(wanted) - set(arms))
        if unknown:
            print(f"unknown arm(s) {unknown}; choose from {sorted(arms)}", file=sys.stderr)
            return 2
        arms = {a: arms[a] for a in arms if a in wanted}

    # ---- the scale, built on this truth
    t0 = time.time()
    truth = common.to_anndata(
        np.vstack([block, control_counts]),
        real_labels + ["non-targeting"] * control_counts.shape[0],
        gene_names, PERT_COL,
    )
    reference = scale.build(cfg, truth, pert_col=PERT_COL, control_label="non-targeting",
                            outdir=outdir / f"{stem}_bundle", bundle_id=stem)
    if reference.built:
        base, policy, replicate = scale_ends(reference)
        log.info("scale built (%.0fs)", time.time() - t0)
    else:
        # §6: where the two ends cannot be built, the raw values stand alone.
        base, policy, replicate = None, None, None
        log.warning("the scale could not be built, raw metrics only: %s", reference.reason)

    settings = GeneratorSettings(confidence_threshold=0.0, effect_scale=1.0)
    report = {
        "held_out": args.held_out, "source": args.source, "n_targets": n,
        "n_shared_targets": len(shared), "targets": targets,
        "cells_per_target_median": float(np.median(list(cells_per_target.values()))),
        "n_controls": int(control_counts.shape[0]), "n_genes": len(gene_names),
        "n_genes_in_source": int(on_axis.sum()), "seed": cfg.seed,
        "baseline": base, "replicate": replicate,
        "scale_reason": None if reference.built else reference.reason,
        "scorer_version": official.scorer_version(), "arms": {},
    }
    draws_by_arm = {}
    for arm, (profiles, variant) in arms.items():
        t0 = time.time()
        predictions = {t: common.TargetPrediction(t, profiles[t]) for t in targets}
        counts, labels_pred = common.build_cells(
            predictions, control_counts, cells_per_target, settings, variant, cfg.seed
        )
        prediction = common.to_anndata(counts, labels_pred, gene_names, PERT_COL)
        result = official.score(cfg, prediction, truth, pert_col=PERT_COL,
                                control_label="non-targeting")
        placed = scale.rescale(result, reference)
        per_target = per_target_values(result)
        nsig = result.nsig_counts()
        ratios = [v["pred"] / v["real"] for v in nsig.values() if v["real"] > 0]
        # The same bootstrap indices for every arm, so differences are paired.
        if reference.built:
            draws, have_expr = bootstrap(per_target, targets, (base, policy), args.bootstrap,
                                         np.random.default_rng(cfg.seed + 1))
            draws_by_arm[arm] = draws
        else:
            draws, have_expr = np.full(2, np.nan), False
        report["arms"][arm] = {
            "raw": result.scored,
            "leaderboard": placed,
            "bootstrap_se": float(np.std(draws, ddof=1)),
            "bootstrap_includes_expr": bool(have_expr),
            "median_nsig_pred_over_real": float(np.median(ratios)) if ratios else None,
            "rms_log2fc": float(np.sqrt(np.mean([np.mean(profiles[t] ** 2) for t in targets]))),
        }
        log.info("  %-22s %+.4f (se %.4f)  %.0fs", arm, placed.get("avg_score", float("nan")),
                 report["arms"][arm]["bootstrap_se"], time.time() - t0)
        del counts, prediction

    if "own_mean" in draws_by_arm:
        for arm, draws in draws_by_arm.items():
            diff = draws - draws_by_arm["own_mean"]
            report["arms"][arm]["paired_diff_vs_own_mean"] = {
                "mean": float(np.mean(diff)), "se": float(np.std(diff, ddof=1)),
            }

    path = outdir / f"{stem}.json"
    path.write_text(json.dumps(report, indent=2, default=str))

    print(f"\n{args.held_out} held out, knowledge from {args.source}: {n} targets, "
          f"median {report['cells_per_target_median']:.0f} cells/target, "
          f"{report['n_controls']} controls, {len(gene_names)} genes")
    if base is not None:
        print("  0 =  " + "  ".join(f"{SHORT[m]} {base[m]:+.3f}" for m in official.SCORED))
        print("  1 =  " + "  ".join(f"{SHORT[m]} {replicate[m]:+.3f}" for m in official.SCORED))
        print("  per-member columns: from_replicate scores")
    else:
        print(f"  NO SCALE ({reference.reason}); per-member columns are RAW metric values")
    print(f"\n  {'arm':22s} {'score':>8s} {'se':>6s} {'vs own_mean':>13s} {'nsig p/r':>8s} "
          f"{'rms lfc':>7s} | " + "  ".join(f"{SHORT[m]:>6s}" for m in official.SCORED))
    for arm, entry in report["arms"].items():
        board = entry["leaderboard"]
        per = board.get("from_replicate") or entry["raw"]
        diff = entry.get("paired_diff_vs_own_mean", {})
        ratio = entry["median_nsig_pred_over_real"]
        print(f"  {arm:22s} {board.get('avg_score', float('nan')):+8.4f} "
              f"{entry['bootstrap_se']:6.4f} "
              f"{diff.get('mean', float('nan')):+7.4f}±{diff.get('se', float('nan')):.3f} "
              f"{(ratio if ratio is not None else float('nan')):8.2f} {entry['rms_log2fc']:7.3f} | "
              + "  ".join(f"{per.get(m, float('nan')):+6.3f}" for m in official.SCORED))
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
