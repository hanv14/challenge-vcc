#!/usr/bin/env python
"""Measurement 4: what about a knockdown's response transfers across cell lines.

    python scripts/measure/transfer_structure.py --config configs/server.yaml

Every response is split into the screen's (or cell line's) **mean response**,
shared by all its targets, and each target's **residual** from it. The
leaderboard's 0 already contains the new context's true mean response
(docs/metrics.md §6a), so only residuals can score above 0, and the question
here is how much of a residual measured in one cell line is still there in
another.

**Replogle**, every pair of screens, on the targets they share. Per target, a
change is log2 of the ratio of mean CP10K (target cells over control cells,
floored at `predict.cpm_floor`), computed on all its sampled cells and on two
random halves. Genes: measured in both screens and at least `--min-cp10k` in
both screens' controls. Reported per screen: how much of the response energy
is mean and how much is residual, after subtracting sampling noise (from the
halves); the residuals' split-half reliability. Per pair: the correlation of
the two mean responses; the per-target correlation of residuals, raw and
corrected for both sides' reliability, by effect-size tercile; and the
out-of-sample R² of predicting one screen's residuals as `alpha` times the
other's, with `alpha` fitted on half the targets.

**LINCS** (Level 5 signatures, landmark genes, CRISPR knockout): the cell
lines with at least `--lincs-min-targets` targets, on the targets present in
all of them. A two-way split, line mean + target effect + interaction (the
interaction includes noise: Level 5 has no replicate split here). Then, for
each line held out: (a) how well the average of the *other* lines' residuals
for a target predicts its residual in the held-out line, against using one
other line; (b) how well the held-out line's mean response is predicted by
the other lines' average, by the lines nearest it in control expression, and
by a ridge from control expression.

Measurement only: trains nothing, writes nothing but its report. CPU only.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vccp.runtime import set_thread_env  # noqa: E402

set_thread_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vccp.config import ConfigError, apply_overrides, load_config  # noqa: E402
from vccp.data.lincs import load_phase1  # noqa: E402
from vccp.data.replogle import load_replogle_context  # noqa: E402
from vccp.logging_utils import get_logger, setup_logging  # noqa: E402
from vccp.paths import DataPaths  # noqa: E402
from vccp.runtime import seed_everything  # noqa: E402
from vccp.sanity.checks import mean_cp10k  # noqa: E402

#: Targets read per batch, so a screen's cells are never all in memory at once.
BATCH_TARGETS = 50


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else float("nan")


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")


# --------------------------------------------------------------------------- #
# Replogle
# --------------------------------------------------------------------------- #
def screen_changes(context, targets, n_cells, n_controls, floor, rng):
    """Per target: (full, half A, half B) log2 fold changes over the screen's genes."""
    controls = context.cells.loc[context.cells["is_control"].astype(bool), "row"].to_numpy()
    if controls.size > n_controls:
        controls = np.sort(rng.choice(controls, n_controls, replace=False))
    control_cp10k = mean_cp10k(context.raw_counts(controls))
    by_target = context.cells.loc[~context.cells["is_control"].astype(bool)]
    by_target = by_target.groupby(by_target["target_gene"].astype(str))["row"]
    rows = {}
    for target in targets:
        r = by_target.get_group(target).to_numpy()
        if r.size > n_cells:
            r = rng.choice(r, n_cells, replace=False)
        rows[target] = rng.permutation(r)

    def lfc(block):
        return np.log2(np.maximum(mean_cp10k(block), floor) / np.maximum(control_cp10k, floor))

    out = {}
    for start in range(0, len(targets), BATCH_TARGETS):
        batch = targets[start:start + BATCH_TARGETS]
        flat = np.concatenate([rows[t] for t in batch])
        block = context.raw_counts(flat)
        offset = 0
        for t in batch:
            cells = block[offset:offset + rows[t].size]
            offset += rows[t].size
            half = cells.shape[0] // 2
            out[t] = (lfc(cells), lfc(cells[:half]), lfc(cells[half:]))
    return out, control_cp10k


def decompose(changes: dict, targets: list[str], genes: np.ndarray):
    """Mean response, residuals and their noise, on one gene subset."""
    full = np.stack([changes[t][0][genes] for t in targets])
    half_a = np.stack([changes[t][1][genes] for t in targets])
    half_b = np.stack([changes[t][2][genes] for t in targets])
    n = len(targets)
    # Noise of a full-depth change ~ var(A - B) / 4, per target, summed over genes.
    noise = ((half_a - half_b) ** 2).sum(axis=1) / 4
    mean = full.mean(axis=0)
    res, res_a, res_b = full - mean, half_a - half_a.mean(0), half_b - half_b.mean(0)
    total = float((full ** 2).sum(axis=1).mean() - noise.mean())
    mean_energy = float(mean @ mean - noise.mean() / n)
    res_energy = float((res ** 2).sum(axis=1).mean() - noise.mean() * (1 - 1 / n))
    reliability = np.array([spearman_brown(corr(res_a[i], res_b[i])) for i in range(n)])
    return {
        "mean": mean, "res": res, "reliability": reliability,
        "energy": {
            "total_signal": total, "mean_signal": mean_energy, "residual_signal": res_energy,
            "mean_share_of_signal": mean_energy / (mean_energy + res_energy)
            if mean_energy + res_energy > 0 else float("nan"),
            "noise_per_target": float(noise.mean()),
            "residual_signal_to_noise": res_energy / float(noise.mean())
            if noise.mean() > 0 else float("nan"),
        },
    }


def replogle_pair(name1, name2, ctx1, ctx2, args, floor, rng, cache, log):
    n1, n2 = ctx1.n_cells_per_target(), ctx2.n_cells_per_target()
    shared = sorted(
        t for t in set(ctx1.targets) & set(ctx2.targets)
        if n1.get(t, 0) >= args.min_cells and n2.get(t, 0) >= args.min_cells
    )
    if len(shared) < 10:
        return {"skipped": f"{len(shared)} shared targets with >= {args.min_cells} cells"}
    targets = sorted(rng.choice(shared, min(args.n_targets, len(shared)), replace=False).tolist())
    for name, ctx in ((name1, ctx1), (name2, ctx2)):
        missing = [t for t in targets if t not in cache[name]["changes"]]
        if missing:
            t0 = time.time()
            changes, control = screen_changes(ctx, missing, args.cells, args.control_cells, floor, rng)
            cache[name]["changes"].update(changes)
            cache[name]["control"] = control
            log.info("  %s: %d targets read (%.0fs)", name, len(missing), time.time() - t0)

    g1 = ctx1.genes["gene_symbol"].astype(str).tolist()
    g2 = ctx2.genes["gene_symbol"].astype(str).tolist()
    pos2 = {g: i for i, g in enumerate(g2)}
    both = [(i, pos2[g]) for i, g in enumerate(g1) if g in pos2]
    i1 = np.array([a for a, _ in both])
    i2 = np.array([b for _, b in both])
    expressed = (cache[name1]["control"][i1] >= args.min_cp10k) & (
        cache[name2]["control"][i2] >= args.min_cp10k)
    i1, i2 = i1[expressed], i2[expressed]

    d1 = decompose(cache[name1]["changes"], targets, i1)
    d2 = decompose(cache[name2]["changes"], targets, i2)
    per_target = []
    for k, t in enumerate(targets):
        r = corr(d1["res"][k], d2["res"][k])
        rel1, rel2 = d1["reliability"][k], d2["reliability"][k]
        corrected = r / np.sqrt(rel1 * rel2) if rel1 > 0.05 and rel2 > 0.05 else float("nan")
        per_target.append({
            "target": t, "r": r, "r_corrected": corrected, "rel1": rel1, "rel2": rel2,
            "size1": float(np.sqrt((d1["res"][k] ** 2).mean())),
            "size2": float(np.sqrt((d2["res"][k] ** 2).mean())),
        })
    frame = pd.DataFrame(per_target)
    frame["tercile"] = pd.qcut(frame["size1"].rank(method="first"), 3,
                               labels=["weak", "middle", "strong"])

    def summary(f):
        return {
            "n": int(len(f)),
            "median_r": float(f["r"].median()),
            "median_r_corrected": float(f["r_corrected"].median()),
            "n_corrected": int(f["r_corrected"].notna().sum()),
            "median_rel1": float(f["rel1"].median()), "median_rel2": float(f["rel2"].median()),
        }

    # Out-of-sample transfer: screen 2's residual as alpha x screen 1's, both directions.
    half = rng.permutation(len(targets))
    fit, test = half[: len(half) // 2], half[len(half) // 2:]

    def transfer(a, b):
        alpha = float((a[fit] * b[fit]).sum() / (a[fit] ** 2).sum())
        r2 = 1 - ((b[test] - alpha * a[test]) ** 2).sum() / (b[test] ** 2).sum()
        return {"alpha": alpha, "r2_out_of_sample": float(r2)}

    return {
        "n_targets": len(targets), "n_shared": len(shared), "n_genes": int(i1.size),
        "energy": {name1: d1["energy"], name2: d2["energy"]},
        "mean_response_corr": corr(d1["mean"], d2["mean"]),
        "mean_response_rms": {name1: float(np.sqrt((d1["mean"] ** 2).mean())),
                              name2: float(np.sqrt((d2["mean"] ** 2).mean()))},
        "residual_corr": {
            "all": summary(frame),
            **{str(k): summary(g) for k, g in frame.groupby("tercile", observed=True)},
        },
        f"predict_{name2}_from_{name1}": transfer(d1["res"], d2["res"]),
        f"predict_{name1}_from_{name2}": transfer(d2["res"], d1["res"]),
    }


# --------------------------------------------------------------------------- #
# LINCS
# --------------------------------------------------------------------------- #
def lincs_structure(phase1_path: Path, args, rng) -> dict:
    data = load_phase1(phase1_path)
    obs = data.obs.reset_index(drop=True)
    keep = obs["has_sig"].to_numpy(bool)
    obs = obs[keep].reset_index(drop=True)
    sig = data.layers["sig"][keep]
    ctrl = data.layers["ctrl"][keep]
    key = obs["cell_line"].astype(str) + "\t" + obs["target_gene"].astype(str)
    # One profile per (line, target): the mean over time points.
    sig_lt = pd.DataFrame(sig).groupby(key.values).mean()
    lines_targets = sig_lt.index.to_series().str.split("\t", expand=True)
    sig_lt.index = pd.MultiIndex.from_arrays([lines_targets[0], lines_targets[1]],
                                             names=["line", "target"])
    per_line = sig_lt.groupby(level="line").size()
    lines = sorted(per_line[per_line >= args.lincs_min_targets].index)
    if len(lines) < 3:
        return {"skipped": f"only {len(lines)} lines with >= {args.lincs_min_targets} targets"}
    counts = sig_lt.loc[lines].groupby(level="target").size()
    targets = sorted(counts[counts == len(lines)].index)
    if len(targets) < args.lincs_min_common:
        return {"skipped": f"only {len(targets)} targets in all of {lines}"}

    cube = np.stack([sig_lt.xs(line, level="line").loc[targets].to_numpy() for line in lines])  # L x T x G
    grand = cube.mean(axis=(0, 1))
    line_eff = cube.mean(axis=1) - grand                 # L x G
    target_eff = cube.mean(axis=0) - grand               # T x G
    inter = cube - grand - line_eff[:, None, :] - target_eff[None, :, :]
    ss = {
        "line_mean": float((line_eff ** 2).sum() * len(targets)),
        "target_effect": float((target_eff ** 2).sum() * len(lines)),
        "interaction_plus_noise": float((inter ** 2).sum()),
    }
    total = sum(ss.values()) + float((grand ** 2).sum() * len(lines) * len(targets))
    shares = {k: v / total for k, v in ss.items()}
    shares["grand_mean"] = 1 - sum(shares.values())

    # (a) a target's residual in a held-out line, from the other lines.
    line_means = cube.mean(axis=1)                       # L x G
    resid = cube - line_means[:, None, :]
    per_line = {}
    ctrl_lt = pd.DataFrame(ctrl).groupby(obs["cell_line"].astype(str).values).mean()
    ctrl_prof = ctrl_lt.loc[lines].to_numpy()
    ctrl_dev = ctrl_prof - ctrl_prof.mean(axis=0)
    for li, line in enumerate(lines):
        others = [j for j in range(len(lines)) if j != li]
        avg_other = resid[others].mean(axis=0)
        one_other = resid[rng.choice(others)]
        r_avg = [corr(resid[li, t], avg_other[t]) for t in range(len(targets))]
        r_one = [corr(resid[li, t], one_other[t]) for t in range(len(targets))]
        # (b) the held-out line's mean response.
        truth = line_means[li]
        mean_others = line_means[others].mean(axis=0)
        sims = np.array([corr(ctrl_dev[li], ctrl_dev[j]) for j in others])
        nearest = [others[j] for j in np.argsort(-sims)[: args.lincs_k]]
        knn = line_means[nearest].mean(axis=0)
        # Ridge from control deviation to mean-response deviation, fitted on
        # the other lines (dual form: n lines << genes).
        x = ctrl_dev[others] - ctrl_dev[others].mean(0)
        y = line_means[others] - mean_others
        gram = x @ x.T
        lam = args.ridge * np.trace(gram) / len(others)
        coef = np.linalg.solve(gram + lam * np.eye(len(others)), y)
        ridge = mean_others + (ctrl_dev[li] - ctrl_dev[others].mean(0)) @ x.T @ coef

        def r2(pred):
            return float(1 - ((truth - pred) ** 2).sum() / ((truth - truth.mean()) ** 2).sum())

        per_line[line] = {
            "residual_r_from_avg_of_others_median": float(np.nanmedian(r_avg)),
            "residual_r_from_one_other_median": float(np.nanmedian(r_one)),
            "mean_resp_r2": {"avg_of_others": r2(mean_others), f"knn{args.lincs_k}": r2(knn),
                             "ridge_on_controls": r2(ridge)},
            "mean_resp_corr": {"avg_of_others": corr(truth, mean_others),
                               f"knn{args.lincs_k}": corr(truth, knn),
                               "ridge_on_controls": corr(truth, ridge)},
            "nearest_lines": [lines[j] for j in nearest],
        }
    return {
        "lines": lines, "n_targets": len(targets), "n_genes": int(cube.shape[2]),
        "variance_shares": shares, "per_line": per_line,
        "line_mean_corr_matrix": pd.DataFrame(line_means.T, columns=lines).corr().round(3).to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", default="measure")
    parser.add_argument("--n-targets", type=int, default=300, help="targets per pair of screens")
    parser.add_argument("--min-cells", type=int, default=40)
    parser.add_argument("--cells", type=int, default=150, help="cells read per target")
    parser.add_argument("--control-cells", type=int, default=3000)
    parser.add_argument("--min-cp10k", type=float, default=0.5,
                        help="control mean CP10K a gene needs in both screens")
    parser.add_argument("--lincs-min-targets", type=int, default=1000)
    parser.add_argument("--lincs-min-common", type=int, default=10,
                        help="targets the chosen lines must share for the LINCS split")
    parser.add_argument("--lincs-k", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--skip-replogle", action="store_true")
    parser.add_argument("--skip-lincs", action="store_true")
    args = parser.parse_args(argv)

    try:
        cfg = apply_overrides(load_config(args.config), args.set)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    cfg = dataclasses.replace(cfg, run_name=args.run_name)
    outdir = Path(cfg.output_root) / cfg.run_name / "measure"
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir / "transfer_structure.log")
    log = get_logger()
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    paths = DataPaths(cfg)
    report = {"seed": cfg.seed, "args": vars(args) | {"config": str(args.config)}}

    if not args.skip_replogle:
        contexts = {n: load_replogle_context(d)
                    for n, d in paths.discover_phase2_contexts().items()}
        cache = {n: {"changes": {}, "control": None} for n in contexts}
        report["replogle"] = {}
        for a, b in itertools.combinations(sorted(contexts), 2):
            log.info("pair %s / %s", a, b)
            report["replogle"][f"{a}~{b}"] = replogle_pair(
                a, b, contexts[a], contexts[b], args, cfg.predict.cpm_floor, rng, cache, log)

    if not args.skip_lincs:
        log.info("LINCS")
        report["lincs"] = lincs_structure(paths.phase1_lincs, args, rng)

    path = outdir / "transfer_structure.json"
    path.write_text(json.dumps(report, indent=2, default=float))
    print_report(report)
    print(f"\nwritten: {path}")
    return 0


def print_report(report: dict) -> None:
    for pair, r in (report.get("replogle") or {}).items():
        print(f"\n== Replogle {pair}")
        if "skipped" in r:
            print("   skipped:", r["skipped"]); continue
        print(f"   {r['n_targets']} of {r['n_shared']} shared targets, {r['n_genes']} genes")
        for screen, e in r["energy"].items():
            print(f"   {screen:16s} mean share of signal {e['mean_share_of_signal']:.2f}  "
                  f"residual signal/noise {e['residual_signal_to_noise']:.2f}")
        print(f"   mean responses: corr {r['mean_response_corr']:+.3f}, rms "
              + ", ".join(f"{k} {v:.3f}" for k, v in r["mean_response_rms"].items()))
        print("   residual corr per target (median): tercile n raw corrected rel1 rel2")
        for k, s in r["residual_corr"].items():
            print(f"     {k:7s} {s['n']:4d} {s['median_r']:+.3f} {s['median_r_corrected']:+.3f} "
                  f"(n={s['n_corrected']}) {s['median_rel1']:.2f} {s['median_rel2']:.2f}")
        for key, v in r.items():
            if key.startswith("predict_"):
                print(f"   {key}: alpha {v['alpha']:+.3f}, out-of-sample R2 {v['r2_out_of_sample']:+.3f}")
    lincs = report.get("lincs")
    if lincs:
        print("\n== LINCS")
        if "skipped" in lincs:
            print("   skipped:", lincs["skipped"]); return
        print(f"   {len(lincs['lines'])} lines x {lincs['n_targets']} targets x {lincs['n_genes']} genes")
        print("   variance shares:", {k: round(v, 3) for k, v in lincs["variance_shares"].items()})
        print("   held-out line: residual r (avg of others | one other) | mean-response R2 "
              "(avg others, knn, ridge on controls)")
        for line, v in lincs["per_line"].items():
            m = v["mean_resp_r2"]
            print(f"   {line:10s} {v['residual_r_from_avg_of_others_median']:+.3f} | "
                  f"{v['residual_r_from_one_other_median']:+.3f} | "
                  + "  ".join(f"{x:+.3f}" for x in m.values()))


if __name__ == "__main__":
    sys.exit(main())
