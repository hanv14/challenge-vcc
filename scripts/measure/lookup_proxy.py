#!/usr/bin/env python
"""The lookup proxy: choose the lookup's denoiser offline, inside one screen.

    # stage 1, the round's targets only (one pass over their cells):
    python scripts/measure/lookup_proxy.py --config configs/server.yaml --targets round
    # stage 2, every target of the screen (one pass over the whole file):
    python scripts/measure/lookup_proxy.py --config configs/server.yaml --targets all

The leaderboard rewards the K562 lookup's target-specific *direction* (D126),
and the raw lookup is about 85% sampling noise. This measures how well each
denoiser recovers a screen's own target-specific change, entry by entry,
without an upload:

1. Every target's cells (at most `predict.lookup_max_cells`, as the lookup
   reads) are shuffled and cut into four quarters; the controls into two
   disjoint halves of at most `predict.lookup_control_cells`. Separate
   controls keep the two halves' noise independent: a shared control mean
   would make them agree for nothing. One streaming pass over the screen
   gives every group's CP10K moments (`predict/screen_stats.py`), cached.
2. Three pairings of the quarters ({0,1}|{2,3}, {0,2}|{1,3}, {0,3}|{1,2})
   give three half-A / half-B splits. Half A is built exactly as the lookup
   builds its table (the same change, floor, expression gate, centring on
   the panel's mean, own gene zeroed, the same denoisers from
   `predict/denoise.py`), at half depth; half B is the truth.
3. Each denoised half-A residual is scored against half B with
   `eval/lookup_proxy.py`: a pds-like rank, an nmae-like error on the genes
   half B calls significant, a reach-like ordering score and, as a
   diagnostic, a Pearson correlation. Per-target values are averaged over
   the pairings and compared with the raw residual by a paired target
   bootstrap.

Two panels: the round's covered targets (ranked among themselves, the
scorer's panel-wide target-gene exclusion), and, with `--targets all`, every
target of the screen (each ranked among all, own gene excluded; at most
`--max-query-targets` queries). The low-rank denoisers need every target and
run only with `--targets all`; their basis is fitted on half A only.

**Validation gate.** On the leaderboard the raw lookup beat the per-gene
James–Stein shrinkage (L4 − L2: pds +0.070, reach +0.030, nmae −0.017,
D127). The instrument is trusted only if raw beats `gene` on the pds-like
and reach-like scores here by more than twice their bootstrap sd. It reports
nmae-like's direction against the leaderboard's (gene better) too.

**Depth.** Half A has half a target's cells. tau is in z² units, so an SNR
setting chosen here carries to the full-depth table; a rank may not (a
noisier input supports fewer components). `--depth quarter` scores a single
quarter against the opposite half, so the optimum's drift from quarter to
half depth can be read and extrapolated to full depth.

Measurement only: trains nothing, changes no pipeline state, CPU only.
Writes `<output_root>/<run-name>/measure/lookup_proxy_<screen>_<targets>.json`
and `.log`, and caches the moments beside them.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vccp.runtime import set_thread_env  # noqa: E402

set_thread_env()

import numpy as np  # noqa: E402

from vccp.config import ConfigError, apply_overrides, load_config  # noqa: E402
from vccp.data import reference  # noqa: E402
from vccp.data.replogle import load_replogle_context  # noqa: E402
from vccp.eval import lookup_proxy as score  # noqa: E402
from vccp.logging_utils import get_logger, setup_logging  # noqa: E402
from vccp.paths import DataPaths  # noqa: E402
from vccp.predict import denoise  # noqa: E402
from vccp.predict.screen_stats import (  # noqa: E402
    GroupMoments, log2_change, null_log2_sd, residual_z, stream_moments,
)
from vccp.runtime import seed_everything  # noqa: E402

#: A target enters the panel with at least this many cells: two per quarter.
MIN_CELLS = 8
QUARTERS = 4
PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
MEMBERS = ("pds", "nmae", "reach", "pearson")
#: Which way is better, per member.
HIGHER_IS_BETTER = {"pds": True, "nmae": False, "reach": True, "pearson": True}

DEFAULT_SNR_TAUS = "0.25,0.5,1,2,4,8,16"
DEFAULT_HARD_TAUS = "1,2,4,6.63,9,16"
DEFAULT_RANKS = "5,10,20,50,100,200"
DEFAULT_COMBINED = "10:1,20:1,50:1,20:4,50:4"


# --------------------------------------------------------------------------- #
# stage 1: the moments
# --------------------------------------------------------------------------- #
def target_rng(seed: int, screen: str, target: str) -> np.random.Generator:
    """One generator per target, so a target's quarters do not depend on which
    other targets were read: the round's stats are a subset of all's."""
    digest = hashlib.sha256(f"{seed}:lookup-proxy:{screen}:{target}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:8], 16))


def quarter_groups(context, targets: list[str], max_cells: int, control_cells: int,
                   seed: int) -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    cells = context.cells
    perturbed = cells.loc[~cells["is_control"].astype(bool)]
    grouped = perturbed.groupby(perturbed["target_gene"].astype(str), observed=True)["row"]
    groups = []
    for target in targets:
        rows = grouped.get_group(target).to_numpy()
        rng = target_rng(seed, context.name, target)
        if rows.size > max_cells:
            rows = rng.choice(rows, max_cells, replace=False)
        rows = rng.permutation(rows)
        n = rows.size
        for q in range(QUARTERS):
            groups.append(np.sort(rows[q * n // QUARTERS:(q + 1) * n // QUARTERS]))
    controls = cells.loc[cells["is_control"].astype(bool), "row"].to_numpy()
    controls = target_rng(seed, context.name, "<controls>").permutation(controls)
    half = min(controls.size // 2, control_cells)
    return groups, (np.sort(controls[:half]), np.sort(controls[half:2 * half]))


def stats_key(cfg, screen: str, path: Path, targets: list[str]) -> str:
    material = {
        "screen": screen, "path": str(path), "targets": sorted(targets), "seed": cfg.seed,
        "max_cells": cfg.predict.lookup_max_cells,
        "control_cells": cfg.predict.lookup_control_cells, "quarters": QUARTERS,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def load_or_stream(cfg, context, targets: list[str], cache: Path) -> GroupMoments:
    log = get_logger()
    key = stats_key(cfg, context.name, context.directory, targets)
    if cache.is_file():
        data = np.load(cache, allow_pickle=True)
        if str(data["key"]) == key:
            log.info("moments: cached %s", cache)
            return GroupMoments(n=data["n"], mean=data["mean"].astype(np.float64),
                                meansq=data["meansq"].astype(np.float64))
        log.info("moments: cached %s has another key; streaming again", cache)
    groups, (ctrl_a, ctrl_b) = quarter_groups(
        context, targets, cfg.predict.lookup_max_cells, cfg.predict.lookup_control_cells,
        cfg.seed,
    )
    log.info("moments: %d targets x %d quarters + 2 control halves (%d, %d cells), "
             "%d cells in all", len(targets), QUARTERS, ctrl_a.size, ctrl_b.size,
             sum(g.size for g in groups) + ctrl_a.size + ctrl_b.size)
    started = time.time()
    moments = stream_moments(context, groups + [ctrl_a, ctrl_b], cfg.resources.max_ram_gb,
                             label=f"proxy {context.name}")
    log.info("moments: streamed in %.0f s", time.time() - started)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, n=moments.n, mean=moments.mean.astype(np.float32),
             meansq=moments.meansq.astype(np.float32), key=np.array(key),
             targets=np.array(targets, dtype=object))
    return moments


# --------------------------------------------------------------------------- #
# stage 2: one split
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Half:
    """One side of a split: per-target moments and change, and the change in
    each of its two parts (for the per-gene factor)."""

    n: np.ndarray            # (targets,)
    mean: np.ndarray         # (targets, genes) mean CP10K
    meansq: np.ndarray
    lfc: np.ndarray
    sub_a: np.ndarray | None
    sub_b: np.ndarray | None
    control: tuple[int, np.ndarray, np.ndarray]

    def residual(self, rows: np.ndarray, expressed: np.ndarray,
                 own: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The lookup's residual over `rows` (change minus their mean change,
        expressed genes, own gene zeroed), and the panel's mean change."""
        sub = np.where(expressed[None, :], self.lfc[rows], 0.0)
        mean = sub.mean(axis=0)
        out = sub - mean[None, :]
        out[:, ~expressed] = 0.0
        zero_own(out, own[rows])
        return out, mean

    def z(self, rows: np.ndarray, shift: np.ndarray, expressed: np.ndarray,
          own: np.ndarray) -> np.ndarray:
        """Each entry's z against the panel's mean response, as the lookup's."""
        n_c, mean_c, meansq_c = self.control
        z = residual_z(self.n[rows], self.mean[rows], self.meansq[rows], n_c, mean_c,
                       meansq_c, shift)
        z[:, ~expressed] = 0.0
        zero_own(z, own[rows])
        return z


def zero_own(matrix: np.ndarray, own_rows: np.ndarray) -> None:
    has = own_rows >= 0
    matrix[np.flatnonzero(has), own_rows[has]] = 0.0


def build_half(moments: GroupMoments, n_targets: int, quarters: tuple[int, ...],
               control_group: int, floor: float) -> Half:
    control = moments.merged([control_group])
    index = np.arange(n_targets) * QUARTERS
    parts = [index + q for q in quarters]
    w = [moments.n[p].astype(np.float64)[:, None] for p in parts]
    total = np.maximum(sum(w), 1.0)
    mean = sum(wi * moments.mean[p] for wi, p in zip(w, parts)) / total
    meansq = sum(wi * moments.meansq[p] for wi, p in zip(w, parts)) / total
    n = sum(moments.n[p] for p in parts).astype(np.float64)
    sub_a = sub_b = None
    if len(parts) == 2:
        sub_a = log2_change(moments.mean[parts[0]], control[1], floor)
        sub_b = log2_change(moments.mean[parts[1]], control[1], floor)
    return Half(n=n, mean=mean, meansq=meansq, lfc=log2_change(mean, control[1], floor),
                sub_a=sub_a, sub_b=sub_b, control=control)


@dataclasses.dataclass
class Panel:
    name: str
    rows: np.ndarray         # rows of the stats' target list
    queries: np.ndarray      # positions within `rows` scored on pds
    panel_exclusion: bool    # the scorer's panel-wide target-gene exclusion


def settings_grid(args, with_basis: bool) -> list[tuple[str, str, float, int]]:
    """(label, kind, tau, rank) for every arm."""
    grid = [("raw", denoise.NONE, 0.0, 0), ("gene", denoise.GENE, 0.0, 0)]
    grid += [(f"snr tau={t:g}", denoise.SNR, t, 0) for t in floats(args.snr_taus)]
    grid += [(f"snr_hard tau={t:g}", denoise.SNR_HARD, t, 0) for t in floats(args.hard_taus)]
    if with_basis:
        grid += [(f"lowrank k={k}", denoise.LOWRANK, 0.0, k) for k in ints(args.ranks)]
        for item in args.combined.split(","):
            if item.strip():
                k, t = item.split(":")
                grid.append((f"lowrank_snr k={int(k)} tau={float(t):g}", denoise.LOWRANK_SNR,
                             float(t), int(k)))
    return grid


def floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def score_split(moments: GroupMoments, n_targets: int, a_quarters, b_quarters,
                panels: list[Panel], own: np.ndarray, cfg, args, grid, log) -> dict:
    """Per panel, per arm, per member: per-target values for one split.

    Everything is scored in residual space: half A's denoised residual against
    half B's residual, on the genes whose half-B residual is significant.
    """
    floor = cfg.predict.cpm_floor
    ctrl_a, ctrl_b = n_targets * QUARTERS, n_targets * QUARTERS + 1
    A = build_half(moments, n_targets, a_quarters, ctrl_a, floor)
    B = build_half(moments, n_targets, b_quarters, ctrl_b, floor)
    gate_cp10k = cfg.predict.lookup_min_control_cp10k
    expressed = (A.control[1] >= gate_cp10k) & (B.control[1] >= gate_cp10k)

    basis = None
    if any(kind in denoise.NEEDS_BASIS for _, kind, _, _ in grid):
        all_rows = np.arange(n_targets)
        res_all, _ = A.residual(all_rows, expressed, own)
        noise = null_log2_sd(A.n, A.control[0], A.control[1], A.control[2], floor)
        noise[:, ~expressed] = 0.0
        k_max = max(k for _, kind, _, k in grid if kind in denoise.NEEDS_BASIS)
        basis = denoise.fit_basis(res_all, noise, k_max, cfg.seed)
        del res_all, noise
        log.info("    basis: %d targets, %d components (top singular values %s)",
                 n_targets, basis.k_max, np.round(basis.singular_values[:5], 1).tolist())

    out = {}
    for panel in panels:
        rows = panel.rows
        res_a, mean_a = A.residual(rows, expressed, own)
        res_b, mean_b = B.residual(rows, expressed, own)
        z_a = A.z(rows, mean_a, expressed, own)
        z_b = B.z(rows, mean_b, expressed, own)
        own_rows = own[rows]
        valid = np.broadcast_to(expressed[None, :], res_b.shape).copy()
        zero_own(valid, own_rows)
        valid = valid.astype(bool)
        significant = score.bh_significant(score.two_sided_p(z_b), valid, args.alpha)
        confidence_scale = np.sqrt(np.maximum(A.control[1], 0.0))[None, :]
        exclude = None
        if panel.panel_exclusion:
            exclude = np.zeros(res_b.shape[1], dtype=bool)
            exclude[own_rows[own_rows >= 0]] = True
        gene = None
        if A.sub_a is not None:
            full = np.where(expressed[None, :], A.lfc[rows], 0.0).astype(np.float32)
            sub_a = np.where(expressed[None, :], A.sub_a[rows], 0.0).astype(np.float32)
            sub_b = np.where(expressed[None, :], A.sub_b[rows], 0.0).astype(np.float32)
            gene = denoise.gene_factor(full, full.mean(axis=0), sub_a, sub_b)
        raw_rms = float(np.sqrt(np.mean(res_a**2)))
        arms = {}
        for label, kind, tau, rank in grid:
            if kind == denoise.GENE and gene is None:
                continue
            if kind in denoise.NEEDS_BASIS and rank > basis.k_max:
                continue  # the screen has fewer targets than components asked for
            pred = denoise.apply(kind, res_a, z=z_a, basis=basis, gene=gene,
                                 tau=tau, rank=max(rank, 1))
            pred = np.where(res_a != 0, pred, 0.0).astype(np.float64)
            arms[label] = {
                "pds": score.pds_like(pred[panel.queries], res_b, panel.queries,
                                      exclude=exclude,
                                      own_gene=None if exclude is not None
                                      else own_rows[panel.queries]),
                "nmae": score.nmae_like(pred, res_b, significant, args.min_gate),
                "reach": score.reach_like(pred, np.abs(pred) * confidence_scale, res_b,
                                          significant, min_gate=args.min_gate),
                "pearson": score.pearson_like(pred, res_b, valid & (res_a != 0)),
                "energy": float(np.sqrt(np.mean(pred**2))) / raw_rms if raw_rms else 0.0,
                "kept": float(np.count_nonzero(pred)) / max(np.count_nonzero(res_a), 1),
            }
        out[panel.name] = {
            "arms": arms,
            "n_targets": int(rows.size),
            "n_queries": int(panel.queries.size),
            "median_significant": float(np.median(significant.sum(axis=1))),
            "raw_rms": raw_rms,
        }
    return out


# --------------------------------------------------------------------------- #
# aggregation and the gate
# --------------------------------------------------------------------------- #
def aggregate(splits: list[dict], panel: str, n_boot: int, seed: int) -> dict:
    labels = list(splits[0][panel]["arms"])
    table = {}
    raw = {m: per_target(splits, panel, "raw", m) for m in MEMBERS}
    for label in labels:
        row = {}
        for member in MEMBERS:
            values = per_target(splits, panel, label, member)
            sign = 1.0 if HIGHER_IS_BETTER[member] else -1.0
            gain, sd = score.paired_bootstrap(sign * (values - raw[member]), n_boot, seed)
            row[member] = float(np.nanmean(values)) if np.isfinite(values).any() else None
            # Targets the member could score: nmae and reach need min_gate
            # significant half-B genes, which weak targets rarely have.
            row[f"{member}_n"] = int(np.isfinite(values).sum())
            row[f"{member}_gain"] = gain
            row[f"{member}_gain_sd"] = sd
        row["energy"] = float(np.mean([s[panel]["arms"][label]["energy"] for s in splits]))
        row["kept"] = float(np.mean([s[panel]["arms"][label]["kept"] for s in splits]))
        table[label] = row
    return table


def per_target(splits: list[dict], panel: str, label: str, member: str) -> np.ndarray:
    """A member's per-target values, averaged over the splits."""
    stack = np.stack([np.asarray(s[panel]["arms"][label][member], dtype=np.float64)
                      for s in splits])
    if not stack.shape[1]:
        return np.zeros(0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


def gate(table: dict) -> dict:
    """Raw against the per-gene shrinkage, as L4 against L2 (D127)."""
    if "gene" not in table:
        return {"verdict": "not-run", "reason": "the per-gene arm needs half-depth splits"}
    gene = table["gene"]
    checks = {}
    for member in ("pds", "reach"):
        # gene's gain over raw, sign-adjusted: negative = raw better.
        gain, sd = gene[f"{member}_gain"], gene[f"{member}_gain_sd"]
        raw_better = gain is not None and sd is not None and np.isfinite(sd) and -gain > 2 * sd
        checks[member] = {"raw_minus_gene": -gain, "sd": sd, "raw_better_by_2sd": bool(raw_better)}
    nmae_gain = gene["nmae_gain"]
    checks["nmae"] = {"gene_better": bool(nmae_gain is not None and nmae_gain > 0),
                      "gene_gain": nmae_gain, "leaderboard": "gene better by 0.017"}
    passed = checks["pds"]["raw_better_by_2sd"] and checks["reach"]["raw_better_by_2sd"]
    return {"verdict": "pass" if passed else "fail", "checks": checks}


def best_per_family(table: dict) -> dict:
    """Each family's best arm by pds + reach gain over raw, nmae not worse."""
    out = {}
    for label, row in table.items():
        family = label.split(" ")[0]
        if family in ("raw", "gene"):
            continue
        if row["nmae_gain"] is not None and np.isfinite(row["nmae_gain"]) and \
                row["nmae_gain"] < -2 * (row["nmae_gain_sd"] or 0):
            continue
        merit = np.nansum([row["pds_gain"], row["reach_gain"]])
        if family not in out or merit > out[family][1]:
            out[family] = (label, float(merit))
    return {family: {"arm": label, "pds_plus_reach_gain": merit}
            for family, (label, merit) in out.items()}


def print_table(name: str, table: dict, n_targets: int) -> None:
    print(f"\n== panel: {name} ({n_targets} targets); gains are over raw, "
          "sign-adjusted so positive = better, +/- bootstrap sd ==")
    head = f"{'arm':<26}{'energy':>7}{'kept':>6}"
    for member in MEMBERS:
        head += f"{member:>9}{'gain':>9}{'sd':>8}"
    print(head)
    for label, row in table.items():
        line = f"{label:<26}{row['energy']:>7.3f}{row['kept']:>6.2f}"
        for member in MEMBERS:
            value, gain, sd = row[member], row[f"{member}_gain"], row[f"{member}_gain_sd"]
            line += f"{value if value is not None else float('nan'):>9.4f}"
            line += f"{gain if gain is not None else float('nan'):>+9.4f}"
            line += f"{sd if sd is not None else float('nan'):>8.4f}"
        print(line)


# --------------------------------------------------------------------------- #
def choose_screen(contexts: dict, round_targets: list[str], wanted: str | None) -> str:
    if wanted:
        if wanted not in contexts:
            raise SystemExit(f"no Replogle screen {wanted!r}; found {sorted(contexts)}")
        return wanted
    best, cover = None, -1
    for name, path in sorted(contexts.items()):
        ctx = load_replogle_context(path)
        counts = ctx.n_cells_per_target()
        n = sum(int(counts.get(t, 0)) >= MIN_CELLS for t in round_targets)
        if n > cover:
            best, cover = name, n
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", default="measure")
    parser.add_argument("--screen", default=None,
                        help="the Replogle screen (default: the one covering most round targets)")
    parser.add_argument("--targets", choices=("round", "all"), default="round")
    parser.add_argument("--depth", choices=("half", "quarter"), default="half")
    parser.add_argument("--pairings", type=int, default=len(PAIRINGS))
    parser.add_argument("--max-query-targets", type=int, default=1000,
                        help="queries ranked on pds in the all-target panel")
    parser.add_argument("--snr-taus", default=DEFAULT_SNR_TAUS)
    parser.add_argument("--hard-taus", default=DEFAULT_HARD_TAUS)
    parser.add_argument("--ranks", default=DEFAULT_RANKS)
    parser.add_argument("--combined", default=DEFAULT_COMBINED,
                        help="rank:tau pairs for lowrank_snr")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-gate", type=int, default=score.MIN_GATE,
                        help="significant half-B genes a target needs on nmae and reach "
                             "(the scorer's min_gate_size)")
    parser.add_argument("--bootstrap", type=int, default=1000)
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

    paths = DataPaths(cfg)
    contexts = paths.discover_phase2_contexts()
    round_targets = sorted(reference.load_pert_counts(paths.pert_counts, "target_gene").targets)
    screen = choose_screen(contexts, round_targets, args.screen)
    stem = f"lookup_proxy_{screen}_{args.targets}" + ("_quarter" if args.depth == "quarter" else "")
    setup_logging(outdir / f"{stem}.log")
    log = get_logger()
    seed_everything(cfg.seed)
    started = time.time()

    context = load_replogle_context(contexts[screen])
    counts = context.n_cells_per_target()
    eligible = sorted(t for t in context.targets if int(counts.get(t, 0)) >= MIN_CELLS)
    in_round = sorted(set(eligible) & set(round_targets))
    targets = in_round if args.targets == "round" else eligible
    if len(in_round) < 3:
        print(f"{screen} covers {len(in_round)} round targets with >= {MIN_CELLS} cells; "
              "too few to score", file=sys.stderr)
        return 2
    log.info("lookup proxy: screen %s, %d round targets covered, %d targets read (%s), "
             "seed %d", screen, len(in_round), len(targets), args.targets, cfg.seed)

    # The all-target cache serves the round too: quarters are drawn per target.
    cache_all = outdir / f"lookup_proxy_moments_{screen}_all.npz"
    cache = outdir / f"lookup_proxy_moments_{screen}_{args.targets}.npz"
    if args.targets == "round" and cache_all.is_file():
        data = np.load(cache_all, allow_pickle=True)
        names = [str(t) for t in data["targets"]]
        if str(data["key"]) == stats_key(cfg, screen, context.directory, names) and \
                set(targets) <= set(names):
            log.info("moments: taking the round's targets from %s", cache_all)
            pick = np.array([names.index(t) for t in targets])
            groups = (pick[:, None] * QUARTERS + np.arange(QUARTERS)[None, :]).ravel()
            groups = np.r_[groups, len(names) * QUARTERS, len(names) * QUARTERS + 1]
            moments = GroupMoments(n=data["n"][groups],
                                   mean=data["mean"][groups].astype(np.float64),
                                   meansq=data["meansq"][groups].astype(np.float64))
        else:
            moments = load_or_stream(cfg, context, targets, cache)
    else:
        moments = load_or_stream(cfg, context, targets, cache)

    symbols = context.genes["gene_symbol"].astype(str).tolist()
    column = {g: i for i, g in enumerate(symbols)}
    own = np.array([column.get(t, -1) for t in targets], dtype=np.int64)
    position = {t: i for i, t in enumerate(targets)}
    round_rows = np.array([position[t] for t in in_round])
    panels = [Panel("round", round_rows, np.arange(round_rows.size), panel_exclusion=True)]
    if args.targets == "all":
        all_rows = np.arange(len(targets))
        queries = all_rows
        if queries.size > args.max_query_targets:
            queries = np.sort(np.random.default_rng(cfg.seed).choice(
                queries, args.max_query_targets, replace=False))
        panels.append(Panel("all", all_rows, queries, panel_exclusion=False))

    grid = settings_grid(args, with_basis=args.targets == "all")
    if args.targets != "all":
        log.info("low-rank arms need every target of the screen: run with --targets all")
    splits = []
    for number, (a, b) in enumerate(PAIRINGS[:max(1, args.pairings)]):
        if args.depth == "quarter":
            a = a[:1]
        log.info("  split %d: half A = quarters %s, truth = quarters %s", number, a, b)
        splits.append(score_split(moments, len(targets), a, b, panels, own, cfg, args, grid, log))

    report = {
        "screen": screen, "targets_read": args.targets, "depth": args.depth,
        "seed": cfg.seed, "n_round_covered": len(in_round), "n_targets_read": len(targets),
        "splits": len(splits), "alpha": args.alpha, "min_gate": args.min_gate,
        "settings": {"lookup_max_cells": cfg.predict.lookup_max_cells,
                     "lookup_control_cells": cfg.predict.lookup_control_cells,
                     "lookup_min_control_cp10k": cfg.predict.lookup_min_control_cp10k,
                     "cpm_floor": cfg.predict.cpm_floor},
        "panels": {},
    }
    for panel in panels:
        table = aggregate(splits, panel.name, args.bootstrap, cfg.seed)
        verdict = gate(table)
        report["panels"][panel.name] = {
            "n_targets": splits[0][panel.name]["n_targets"],
            "n_queries": splits[0][panel.name]["n_queries"],
            "median_significant_genes": splits[0][panel.name]["median_significant"],
            "raw_rms": splits[0][panel.name]["raw_rms"],
            "gate": verdict,
            "best_per_family": best_per_family(table),
            "arms": table,
        }
        print_table(panel.name, table, splits[0][panel.name]["n_targets"])
        print("targets scored: " + ", ".join(
            f"{m} {table['raw'][m + '_n']}" for m in MEMBERS))
        print(f"gate ({panel.name}): {verdict['verdict']}"
              + (f"  {json.dumps(verdict.get('checks', {}), default=float)}"
                 if "checks" in verdict else f"  {verdict.get('reason', '')}"))
        print(f"best per family ({panel.name}): "
              f"{json.dumps(report['panels'][panel.name]['best_per_family'])}")
    report["seconds"] = round(time.time() - started, 1)
    path = outdir / f"{stem}.json"
    path.write_text(json.dumps(report, indent=2, default=float))
    log.info("lookup proxy: %s (%.0f s)", path, report["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
