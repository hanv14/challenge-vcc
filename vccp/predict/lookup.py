"""The lookup: a target's own change, measured in a Replogle screen (D124).

REDESIGN.md §4, rank 2. The leaderboard's 0 is the context's true mean
response (docs/metrics.md §6a), so only target-specific changes can score
above it. The model's target-specific part costs points on the leaderboard
(D118). This is the simplest alternative: for every target a Replogle screen
measured, its change in that screen **minus that screen's mean change over
the round's targets**, carried to the challenge context as is.

    lookup(t) = lfc_screen(t) - mean_{t' covered by the screen} lfc_screen(t')

A change is log2 of the ratio of per-gene mean CP10K, the target's cells over
the screen's control cells, both floored at `predict.cpm_floor`: the ratio the
count generator applies. It is placed on the challenge gene axis through the
screen's `challenge_idx`. A gene the screen does not measure, or measures
below `predict.lookup_min_control_cp10k` in its controls, gets 0, and so does
the target's own gene (the knockdown prior speaks for it, and no metric
scores it).

Each target is read from one screen: among `predict.lookup_screens` (every
screen when empty), the one with the most cells for it. Its residual is taken
against the mean of the targets read from that same screen, so it is a
deviation from that screen's own mean response.

A target's **reliability** is the Spearman–Brown corrected correlation of its
residual between two random halves of its cells, clipped to [0, 1]. With
`predict.lookup_weighting: reliability` it multiplies the target's change, so
a target whose screen measurement is mostly noise contributes little.

**Per-gene shrinkage** (`predict.lookup_shrinkage: gene`; the default is
`none`, raw, since the leaderboard refuted it, D127). At a few hundred cells
most of a residual is sampling noise, spread over thousands of genes. For
each gene and screen, the noise in a full-depth change is estimated from
the halves as mean_t (half_A − half_B)² / 4, and the residual's total power
as mean_t residual²; the gene's residuals are multiplied by
max(0, 1 − noise / total), the James–Stein factor. A gene whose residuals are
no larger than their noise goes to 0; one with real between-target variation
keeps most of it. Because it pools each gene across targets, it also zeroes
the large, rare entries that distinguish targets, which is why raw scored
0.0195 higher. The factors are stored with the table and applied when a
change is asked for, so the cache does not depend on the choice.

**Per-entry denoisers** (D128): `snr`, `snr_hard`, `lowrank` and
`lowrank_snr` (`denoise.py`). The table stores each entry's z for the SNR
ones: the target's mean CP10K tested against the screen's mean response,
from the per-cell CP10K variance of the target's cells and the controls'
(`screen_stats.residual_z`). The low-rank ones project onto the top
components of *every* target the screen measured, so each screen's basis is
built by one streaming pass over its cells (`screen_stats.py`) and cached
beside the table as `sources/lookup_basis_<screen>.npz`. A denoiser never
moves an entry the raw residual leaves at exactly 0: a gene the screen does
not measure or express, or the target's own gene.

**Matched energy** (`predict.lookup_match_energy`, D130). A shrinking
denoiser also makes the lookup smaller, and on the leaderboard pds tracks the
lookup's share of the predicted change (L5, L6: D129), so at the same scale
it would lose to dilution whatever its direction. With matched energy the
denoised table is multiplied by one global factor, rms(raw) / rms(denoised)
over every covered target, so its total size equals the raw table's and only
how that size is spread across entries changes: entries with strong evidence
grow, noise shrinks. One factor for all targets keeps the denoiser's own
weighting between them. It is a rule, recomputed from whatever table a round
builds.

The table depends on the data, the round's targets, the gene axis and the
lookup settings, not on the model, so it is built once per run and cached
under `sources/`, keyed by all of them. A cache with another key is rebuilt.
A cache from before D128 has no z; it is rebuilt when a denoiser asks for
one, from the same cells, so its residuals do not change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_utils import get_logger
from . import denoise
from .denoise import LowRankBasis

#: Targets read per batch, so a screen's cells are never all in memory at once.
BATCH_TARGETS = 64
#: A screen counts as measuring a target only with at least this many cells,
#: the fewest that can be split into two halves.
MIN_CELLS = 2

NONE, RELIABILITY, GENE = "none", "reliability", "gene"


@dataclass
class LookupTable:
    """One row per covered target, over the challenge gene axis."""

    targets: list[str]
    residual: np.ndarray
    reliability: np.ndarray
    screen: list[str]
    n_cells: np.ndarray
    key: str
    #: Each screen's mean change over its covered targets, on the same axis.
    screen_means: dict[str, np.ndarray] = field(default_factory=dict)
    #: Each screen's per-gene shrinkage factor in [0, 1], on the same axis.
    gene_factor: dict[str, np.ndarray] = field(default_factory=dict)
    uncovered: list[str] = field(default_factory=list)
    #: Each entry's z (the target against the screen's mean response), same
    #: shape as `residual`; None in a table cached before D128.
    z: np.ndarray | None = None
    #: Each screen's low-rank basis, attached by `load_or_build` when a
    #: denoiser needs it; not stored in the table's file.
    bases: dict[str, LowRankBasis] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._row = {t: i for i, t in enumerate(self.targets)}
        self._energy: dict[tuple, float] = {}

    def covers(self, target: str) -> bool:
        return target in self._row

    def weight(self, target: str, weighting: str) -> float:
        if weighting == RELIABILITY:
            return float(self.reliability[self._row[target]])
        return 1.0

    def residual_for(self, target: str, shrinkage: str, tau: float = 1.0,
                     rank: int = 1) -> np.ndarray:
        row = self._row[target]
        raw = self.residual[row]
        if shrinkage == NONE:
            return raw
        if shrinkage == GENE:
            return raw * self.gene_factor[self.screen[row]]
        if shrinkage in denoise.NEEDS_Z and self.z is None:
            raise ValueError(
                f"lookup_shrinkage {shrinkage!r} needs each entry's z, which this table "
                "predates; rebuild it (load_or_build does when it is asked for)"
            )
        if shrinkage in denoise.NEEDS_BASIS and self.screen[row] not in self.bases:
            raise ValueError(
                f"lookup_shrinkage {shrinkage!r} needs screen {self.screen[row]}'s low-rank "
                "basis, which was not attached"
            )
        out = denoise.apply(
            shrinkage, raw,
            z=None if self.z is None else self.z[row],
            basis=self.bases.get(self.screen[row]),
            tau=tau, rank=rank,
        )
        # Never move what the raw residual leaves at exactly 0: unmeasured or
        # unexpressed genes, and the target's own gene.
        return np.where(raw != 0, out, 0.0).astype(np.float32)

    def energy_factor(self, shrinkage: str, tau: float = 1.0, rank: int = 1) -> float:
        """rms(raw) / rms(denoised) over every covered target: the one factor
        that gives the denoised table the raw table's total size."""
        if shrinkage == NONE:
            return 1.0
        key = (shrinkage, float(tau), int(rank))
        if key not in self._energy:
            raw = float(np.sum(self.residual.astype(np.float64) ** 2))
            used = sum(
                float(np.sum(self.residual_for(t, shrinkage, tau, rank).astype(np.float64) ** 2))
                for t in self.targets
            )
            self._energy[key] = float(np.sqrt(raw / used)) if used > 0 else 1.0
        return self._energy[key]

    def change(
        self, target: str, scale: float, weighting: str, shrinkage: str = NONE,
        tau: float = 1.0, rank: int = 1, match_energy: bool = False,
    ) -> np.ndarray | None:
        """`scale x weight x (denoised) residual`, times the energy factor when
        `match_energy`; None for a target no screen measured."""
        if target not in self._row:
            return None
        factor = self.energy_factor(shrinkage, tau, rank) if match_energy else 1.0
        return (
            scale * factor * self.weight(target, weighting)
            * self.residual_for(target, shrinkage, tau, rank)
        ).astype(np.float32)

    def describe(
        self, target: str, scale: float, weighting: str, shrinkage: str = NONE,
        tau: float = 1.0, rank: int = 1, match_energy: bool = False,
    ) -> dict[str, Any]:
        if target not in self._row:
            return {"applied": False, "reason": "no lookup screen measured this target"}
        row = self._row[target]
        factor = self.energy_factor(shrinkage, tau, rank) if match_energy else 1.0
        used = factor * self.residual_for(target, shrinkage, tau, rank)
        return {
            "applied": scale > 0,
            "screen": self.screen[row],
            "n_cells": int(self.n_cells[row]),
            "reliability": float(self.reliability[row]),
            "weight": self.weight(target, weighting),
            "rms_raw": float(np.sqrt(np.mean(self.residual[row] ** 2))),
            "rms_used": float(np.sqrt(np.mean(used**2))),
            "genes_moved": int(np.count_nonzero(used)),
            "shrinkage": shrinkage,
            **({"snr_tau": tau} if shrinkage in denoise.NEEDS_Z else {}),
            **({"rank": rank} if shrinkage in denoise.NEEDS_BASIS else {}),
            **({"energy_factor": factor} if match_energy else {}),
        }

    def summary(self) -> dict[str, Any]:
        per_screen = {}
        for name in sorted(set(self.screen)):
            rows = [i for i, s in enumerate(self.screen) if s == name]
            per_screen[name] = {
                "n_targets": len(rows),
                "median_cells": float(np.median(self.n_cells[rows])),
                "median_reliability": float(np.median(self.reliability[rows])),
                "mean_response_rms": float(np.sqrt(np.mean(self.screen_means[name] ** 2))),
                "genes_kept_by_shrinkage": int(np.count_nonzero(self.gene_factor[name])),
                "median_gene_factor_kept": float(
                    np.median(self.gene_factor[name][self.gene_factor[name] > 0])
                ) if np.any(self.gene_factor[name] > 0) else 0.0,
            }
        return {
            "key": self.key,
            "has_z": self.z is not None,
            "n_covered": len(self.targets),
            "n_uncovered": len(self.uncovered),
            "uncovered": self.uncovered,
            "per_screen": per_screen,
            "median_residual_rms": float(
                np.median(np.sqrt(np.mean(self.residual**2, axis=1)))
            ) if len(self.targets) else None,
        }

    # ---- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            targets=np.array(self.targets, dtype=object),
            residual=self.residual.astype(np.float32),
            reliability=self.reliability.astype(np.float32),
            screen=np.array(self.screen, dtype=object),
            n_cells=self.n_cells.astype(np.int64),
            screen_names=np.array(sorted(self.screen_means), dtype=object),
            screen_means=np.stack([self.screen_means[s] for s in sorted(self.screen_means)])
            if self.screen_means else np.zeros((0, self.residual.shape[1]), np.float32),
            gene_factor=np.stack([self.gene_factor[s] for s in sorted(self.screen_means)])
            if self.screen_means else np.zeros((0, self.residual.shape[1]), np.float32),
            uncovered=np.array(self.uncovered, dtype=object),
            key=np.array(self.key),
            **({"z": self.z.astype(np.float32)} if self.z is not None else {}),
        )
        path.with_suffix(".json").write_text(json.dumps(self.summary(), indent=2))

    @classmethod
    def load(cls, path: Path) -> LookupTable:
        data = np.load(path, allow_pickle=True)
        names = [str(s) for s in data["screen_names"]]
        return cls(
            targets=[str(t) for t in data["targets"]],
            residual=data["residual"],
            reliability=data["reliability"],
            screen=[str(s) for s in data["screen"]],
            n_cells=data["n_cells"],
            key=str(data["key"]),
            screen_means={n: data["screen_means"][i] for i, n in enumerate(names)},
            gene_factor={n: data["gene_factor"][i] for i, n in enumerate(names)},
            uncovered=[str(t) for t in data["uncovered"]],
            z=data["z"] if "z" in data.files else None,
        )


# --------------------------------------------------------------------------- #
# building it
# --------------------------------------------------------------------------- #
def spearman_brown(a: np.ndarray, b: np.ndarray) -> float:
    """Split-half correlation corrected to full depth, clipped to [0, 1]."""
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.sqrt((a @ a) * (b @ b)))
    if denom <= 0:
        return 0.0
    r = float(a @ b) / denom
    if r <= 0:
        return 0.0
    return float(min(1.0, 2 * r / (1 + r)))


def lookup_key(cfg, screens: dict[str, Path], targets: list[str], n_genes: int) -> str:
    """Everything the table depends on, hashed."""
    material = {
        "screens": {name: str(path) for name, path in sorted(screens.items())},
        "targets": sorted(targets),
        "n_genes": n_genes,
        "seed": cfg.seed,
        "cpm_floor": cfg.predict.cpm_floor,
        "max_cells": cfg.predict.lookup_max_cells,
        "control_cells": cfg.predict.lookup_control_cells,
        "min_control_cp10k": cfg.predict.lookup_min_control_cp10k,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def chosen_screens(cfg, paths) -> dict[str, Path]:
    """The screens the lookup may read, by name."""
    found = paths.discover_phase2_contexts()
    wanted = tuple(cfg.predict.lookup_screens)
    if not wanted:
        return found
    missing = [name for name in wanted if name not in found]
    if missing:
        raise ValueError(
            f"predict.lookup_screens names {missing}, which are not among the Replogle "
            f"screens found under {paths.phase2_dir}: {sorted(found)}"
        )
    return {name: found[name] for name in wanted}


def build_lookup(cfg, paths, targets: list[str], gene_axis: list[str]) -> LookupTable:
    """Read every covered target's cells once and build the table."""
    from ..data.replogle import load_replogle_context
    from ..sanity.checks import mean_cp10k
    from .screen_stats import meansq_cp10k, residual_z

    log = get_logger()
    screens = chosen_screens(cfg, paths)
    n_genes = len(gene_axis)
    key = lookup_key(cfg, screens, targets, n_genes)
    contexts = {name: load_replogle_context(path) for name, path in screens.items()}
    gene_index = {g: i for i, g in enumerate(gene_axis)}

    # Each target from the screen with the most cells for it.
    best: dict[str, tuple[str, int]] = {}
    for name in sorted(contexts):
        counts = contexts[name].n_cells_per_target()
        for target in targets:
            n = int(counts.get(target, 0))
            if n >= MIN_CELLS and n > best.get(target, ("", 0))[1]:
                best[target] = (name, n)

    floor = cfg.predict.cpm_floor
    rows_out: dict[str, tuple[np.ndarray, float, str, int]] = {}
    screen_means: dict[str, np.ndarray] = {}
    gene_factor: dict[str, np.ndarray] = {}
    for name in sorted(contexts):
        mine = sorted(t for t, (s, _) in best.items() if s == name)
        if not mine:
            continue
        context = contexts[name]
        rng = np.random.default_rng(
            int(hashlib.sha256(f"{cfg.seed}:lookup:{name}".encode()).hexdigest()[:8], 16)
        )
        controls = context.cells.loc[context.cells["is_control"].astype(bool), "row"].to_numpy()
        if controls.size > cfg.predict.lookup_control_cells:
            controls = np.sort(rng.choice(controls, cfg.predict.lookup_control_cells, replace=False))
        control_block = context.raw_counts(controls)
        control_cp10k = mean_cp10k(control_block)
        control_meansq = meansq_cp10k(control_block)
        n_controls = control_block.shape[0]
        del control_block
        axis_idx = context.challenge_idx
        if axis_idx.size and int(axis_idx.max()) >= n_genes:
            raise ValueError(
                f"screen {name}: challenge_idx reaches {int(axis_idx.max())} but the gene "
                f"axis has {n_genes} genes; the processed data belong to another round"
            )
        expressed = control_cp10k >= cfg.predict.lookup_min_control_cp10k

        def on_axis(block, control_cp10k=control_cp10k, axis_idx=axis_idx, expressed=expressed):
            """One target's change, on the challenge gene axis, this screen bound."""
            lfc = np.log2(np.maximum(mean_cp10k(block), floor) / np.maximum(control_cp10k, floor))
            out = np.zeros(n_genes, dtype=np.float32)
            out[axis_idx[expressed]] = lfc[expressed]
            return out

        by_target = context.cells.loc[~context.cells["is_control"].astype(bool)]
        by_target = by_target.groupby(by_target["target_gene"].astype(str), observed=True)["row"]
        full, half_a, half_b, cells, moments = {}, {}, {}, {}, {}
        for start in range(0, len(mine), BATCH_TARGETS):
            batch = mine[start:start + BATCH_TARGETS]
            picked = {}
            for target in batch:
                rows = by_target.get_group(target).to_numpy()
                if rows.size > cfg.predict.lookup_max_cells:
                    rows = rng.choice(rows, cfg.predict.lookup_max_cells, replace=False)
                picked[target] = rng.permutation(rows)
            block = context.raw_counts(np.concatenate([picked[t] for t in batch]))
            offset = 0
            for target in batch:
                n = picked[target].size
                own = block[offset:offset + n]
                offset += n
                full[target] = on_axis(own)
                moments[target] = (mean_cp10k(own), meansq_cp10k(own))
                half_a[target] = on_axis(own[: n // 2])
                half_b[target] = on_axis(own[n // 2:])
                cells[target] = n
            del block
        mean = np.mean([full[t] for t in mine], axis=0).astype(np.float32)
        screen_means[name] = mean
        # James–Stein per gene: the share of the residuals' power that is not
        # sampling noise. The noise of a full-depth change is var(A - B) / 4.
        gene_factor[name] = denoise.gene_factor(
            np.stack([full[t] for t in mine]), mean,
            np.stack([half_a[t] for t in mine]), np.stack([half_b[t] for t in mine]),
        )
        # Each entry's z, against the screen's mean response on its own genes.
        shift = mean[axis_idx]
        for target in mine:
            residual = full[target] - mean
            own_gene = gene_index.get(target)
            if own_gene is not None:
                residual[own_gene] = 0.0
            reliability = spearman_brown(half_a[target] - mean, half_b[target] - mean)
            z = np.zeros(n_genes, dtype=np.float32)
            z_screen = residual_z(cells[target], *moments[target], n_controls, control_cp10k,
                                  control_meansq, shift)
            z[axis_idx[expressed]] = z_screen[expressed]
            if own_gene is not None:
                z[own_gene] = 0.0
            rows_out[target] = (residual.astype(np.float32), reliability, name, cells[target], z)
        log.info(
            "  lookup: %s gives %d targets (median %d cells), mean response rms %.3f, "
            "median split-half reliability %.2f, %d genes kept by shrinkage",
            name, len(mine), int(np.median([cells[t] for t in mine])),
            float(np.sqrt(np.mean(mean**2))),
            float(np.median([rows_out[t][1] for t in mine])),
            int(np.count_nonzero(gene_factor[name])),
        )

    covered = sorted(rows_out)
    residual = (
        np.stack([rows_out[t][0] for t in covered])
        if covered else np.zeros((0, n_genes), dtype=np.float32)
    )
    return LookupTable(
        targets=covered,
        residual=residual,
        reliability=np.array([rows_out[t][1] for t in covered], dtype=np.float32),
        screen=[rows_out[t][2] for t in covered],
        n_cells=np.array([rows_out[t][3] for t in covered], dtype=np.int64),
        key=key,
        screen_means=screen_means,
        gene_factor=gene_factor,
        uncovered=sorted(set(targets) - set(covered)),
        z=(
            np.stack([rows_out[t][4] for t in covered])
            if covered else np.zeros((0, n_genes), dtype=np.float32)
        ),
    )


def load_or_build(cfg, paths, run_paths, targets: list[str], gene_axis: list[str]):
    """The run's lookup table, or None when `predict.lookup_scale` is 0."""
    if cfg.predict.lookup_scale <= 0:
        return None
    log = get_logger()
    path = run_paths.root / "sources" / "lookup.npz"
    key = lookup_key(cfg, chosen_screens(cfg, paths), targets, len(gene_axis))
    shrinkage = cfg.predict.lookup_shrinkage
    table = None
    if path.is_file():
        cached = LookupTable.load(path)
        if cached.key != key:
            log.info("  lookup: cached table has key %s, this run needs %s; rebuilding",
                     cached.key, key)
        elif cached.z is None and shrinkage in denoise.NEEDS_Z:
            log.info("  lookup: cached table %s predates the per-entry z that '%s' needs; "
                     "rebuilding it from the same cells", path, shrinkage)
        else:
            log.info("  lookup: cached table %s (%d targets)", path, len(cached.targets))
            table = cached
    if table is None:
        table = build_lookup(cfg, paths, targets, gene_axis)
        table.save(path)
        log.info("  lookup: %d of %d targets covered -> %s",
                 len(table.targets), len(targets), path)
    if shrinkage in denoise.NEEDS_BASIS:
        table.bases = load_or_build_bases(cfg, paths, run_paths, sorted(set(table.screen)),
                                          gene_axis)
    return table


# --------------------------------------------------------------------------- #
# the low-rank basis: every target a screen measured
# --------------------------------------------------------------------------- #
def basis_key(cfg, screen: str, path: Path, n_genes: int) -> str:
    material = {
        "screen": screen,
        "path": str(path),
        "n_genes": n_genes,
        "seed": cfg.seed,
        "cpm_floor": cfg.predict.cpm_floor,
        "max_cells": cfg.predict.lookup_max_cells,
        "control_cells": cfg.predict.lookup_control_cells,
        "min_control_cp10k": cfg.predict.lookup_min_control_cp10k,
        "k_max": cfg.predict.lookup_basis_rank_max,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def screen_groups(context, max_cells: int, control_cells: int, rng) -> tuple[
        list[str], list[np.ndarray], np.ndarray]:
    """Every target with at least `MIN_CELLS` cells, its rows capped at
    `max_cells`, and the controls capped at `control_cells`, all drawn by `rng`."""
    cells = context.cells
    controls = cells.loc[cells["is_control"].astype(bool), "row"].to_numpy()
    if controls.size > control_cells:
        controls = rng.choice(controls, control_cells, replace=False)
    perturbed = cells.loc[~cells["is_control"].astype(bool)]
    grouped = perturbed.groupby(perturbed["target_gene"].astype(str), observed=True)["row"]
    targets, groups = [], []
    for target in sorted(grouped.groups):
        rows = grouped.get_group(target).to_numpy()
        if rows.size < MIN_CELLS:
            continue
        if rows.size > max_cells:
            rows = rng.choice(rows, max_cells, replace=False)
        targets.append(target)
        groups.append(np.sort(rows))
    return targets, groups, np.sort(controls)


def build_basis(cfg, name: str, context, gene_axis: list[str]) -> LowRankBasis:
    """The low-rank basis of one screen, on the challenge gene axis."""
    from .screen_stats import log2_change, null_log2_sd, stream_moments

    rng = np.random.default_rng(
        int(hashlib.sha256(f"{cfg.seed}:lookup-basis:{name}".encode()).hexdigest()[:8], 16)
    )
    targets, groups, controls = screen_groups(
        context, cfg.predict.lookup_max_cells, cfg.predict.lookup_control_cells, rng
    )
    moments = stream_moments(context, groups + [controls], cfg.resources.max_ram_gb,
                             label=f"lookup basis {name}")
    n_t = len(targets)
    n_c, mean_c, meansq_c = moments.merged([n_t])
    floor = cfg.predict.cpm_floor
    expressed = mean_c >= cfg.predict.lookup_min_control_cp10k
    lfc = log2_change(moments.mean[:n_t], mean_c, floor)
    noise = null_log2_sd(moments.n[:n_t], n_c, mean_c, meansq_c, floor)
    del moments
    lfc[:, ~expressed] = 0.0
    noise[:, ~expressed] = 0.0
    residual = lfc - lfc.mean(axis=0, keepdims=True)
    residual[:, ~expressed] = 0.0
    column = {g: i for i, g in enumerate(context.genes["gene_symbol"].astype(str))}
    for row, target in enumerate(targets):
        if target in column:
            residual[row, column[target]] = 0.0
    fitted = denoise.fit_basis(residual, noise, cfg.predict.lookup_basis_rank_max, cfg.seed)
    axis_idx = context.challenge_idx
    components = np.zeros((fitted.k_max, len(gene_axis)), dtype=np.float32)
    components[:, axis_idx] = fitted.components
    scale = np.zeros(len(gene_axis), dtype=np.float32)
    scale[axis_idx] = fitted.scale
    get_logger().info(
        "  lookup basis: %s, %d targets, %d components, top singular values %s",
        name, n_t, fitted.k_max, np.round(fitted.singular_values[:5], 1).tolist(),
    )
    return LowRankBasis(components=components, scale=scale,
                        singular_values=fitted.singular_values, n_targets=n_t)


def save_basis(basis: LowRankBasis, key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, components=basis.components, scale=basis.scale,
                        singular_values=basis.singular_values,
                        n_targets=np.array(basis.n_targets), key=np.array(key))


def load_basis(path: Path) -> tuple[LowRankBasis, str]:
    data = np.load(path, allow_pickle=False)
    basis = LowRankBasis(components=data["components"], scale=data["scale"],
                         singular_values=data["singular_values"],
                         n_targets=int(data["n_targets"]))
    return basis, str(data["key"])


def load_or_build_bases(cfg, paths, run_paths, screens: list[str],
                        gene_axis: list[str]) -> dict[str, LowRankBasis]:
    """Each named screen's basis, from `sources/` when its key matches."""
    from ..data.replogle import load_replogle_context

    log = get_logger()
    found = chosen_screens(cfg, paths)
    out = {}
    for name in screens:
        path = run_paths.root / "sources" / f"lookup_basis_{name}.npz"
        key = basis_key(cfg, name, found[name], len(gene_axis))
        if path.is_file():
            basis, cached_key = load_basis(path)
            if cached_key == key and basis.k_max >= cfg.predict.lookup_rank:
                log.info("  lookup basis: cached %s (%d components)", path, basis.k_max)
                out[name] = basis
                continue
            log.info("  lookup basis: cached %s does not fit this run; rebuilding", path)
        basis = build_basis(cfg, name, load_replogle_context(found[name]), gene_axis)
        if basis.k_max < cfg.predict.lookup_rank:
            raise ValueError(
                f"screen {name} gives a basis of {basis.k_max} components, fewer than "
                f"predict.lookup_rank ({cfg.predict.lookup_rank}); lower it"
            )
        save_basis(basis, key, path)
        out[name] = basis
    return out
