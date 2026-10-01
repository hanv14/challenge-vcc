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

**Per-gene shrinkage** (`predict.lookup_shrinkage: gene`, the default). At a
few hundred cells most of a residual is sampling noise, spread over thousands
of genes, and adding it to every target would make the DE test call noise.
For each gene and screen, the noise in a full-depth change is estimated from
the halves as mean_t (half_A − half_B)² / 4, and the residual's total power
as mean_t residual²; the gene's residuals are multiplied by
max(0, 1 − noise / total), the James–Stein factor. A gene whose residuals are
no larger than their noise goes to 0; one with real between-target variation
keeps most of it. The factors are stored with the table and applied when a
change is asked for, so the cache does not depend on the choice.

The table depends on the data, the round's targets, the gene axis and the
lookup settings, not on the model, so it is built once per run and cached
under `sources/`, keyed by all of them. A cache with another key is rebuilt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_utils import get_logger

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

    def __post_init__(self) -> None:
        self._row = {t: i for i, t in enumerate(self.targets)}

    def covers(self, target: str) -> bool:
        return target in self._row

    def weight(self, target: str, weighting: str) -> float:
        if weighting == RELIABILITY:
            return float(self.reliability[self._row[target]])
        return 1.0

    def residual_for(self, target: str, shrinkage: str) -> np.ndarray:
        row = self._row[target]
        if shrinkage == GENE:
            return self.residual[row] * self.gene_factor[self.screen[row]]
        return self.residual[row]

    def change(
        self, target: str, scale: float, weighting: str, shrinkage: str = NONE
    ) -> np.ndarray | None:
        """`scale x weight x (shrunk) residual`, or None for a target no screen measured."""
        if target not in self._row:
            return None
        return (
            scale * self.weight(target, weighting) * self.residual_for(target, shrinkage)
        ).astype(np.float32)

    def describe(
        self, target: str, scale: float, weighting: str, shrinkage: str = NONE
    ) -> dict[str, Any]:
        if target not in self._row:
            return {"applied": False, "reason": "no lookup screen measured this target"}
        row = self._row[target]
        used = self.residual_for(target, shrinkage)
        return {
            "applied": scale > 0,
            "screen": self.screen[row],
            "n_cells": int(self.n_cells[row]),
            "reliability": float(self.reliability[row]),
            "weight": self.weight(target, weighting),
            "rms_raw": float(np.sqrt(np.mean(self.residual[row] ** 2))),
            "rms_used": float(np.sqrt(np.mean(used**2))),
            "genes_moved": int(np.count_nonzero(used)),
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
        control_cp10k = mean_cp10k(context.raw_counts(controls))
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
        full, half_a, half_b, cells = {}, {}, {}, {}
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
                half_a[target] = on_axis(own[: n // 2])
                half_b[target] = on_axis(own[n // 2:])
                cells[target] = n
            del block
        mean = np.mean([full[t] for t in mine], axis=0).astype(np.float32)
        screen_means[name] = mean
        # James–Stein per gene: the share of the residuals' power that is not
        # sampling noise. The noise of a full-depth change is var(A - B) / 4.
        noise = np.mean([(half_a[t] - half_b[t]) ** 2 for t in mine], axis=0) / 4
        power = np.mean([(full[t] - mean) ** 2 for t in mine], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(power > 0, 1.0 - noise / power, 0.0)
        gene_factor[name] = np.clip(factor, 0.0, 1.0).astype(np.float32)
        for target in mine:
            residual = full[target] - mean
            own_gene = gene_index.get(target)
            if own_gene is not None:
                residual[own_gene] = 0.0
            reliability = spearman_brown(half_a[target] - mean, half_b[target] - mean)
            rows_out[target] = (residual.astype(np.float32), reliability, name, cells[target])
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
    )


def load_or_build(cfg, paths, run_paths, targets: list[str], gene_axis: list[str]):
    """The run's lookup table, or None when `predict.lookup_scale` is 0."""
    if cfg.predict.lookup_scale <= 0:
        return None
    log = get_logger()
    path = run_paths.root / "sources" / "lookup.npz"
    key = lookup_key(cfg, chosen_screens(cfg, paths), targets, len(gene_axis))
    if path.is_file():
        table = LookupTable.load(path)
        if table.key == key:
            log.info("  lookup: cached table %s (%d targets)", path, len(table.targets))
            return table
        log.info("  lookup: cached table has key %s, this run needs %s; rebuilding",
                 table.key, key)
    table = build_lookup(cfg, paths, targets, gene_axis)
    table.save(path)
    log.info("  lookup: %d of %d targets covered -> %s", len(table.targets), len(targets), path)
    return table
