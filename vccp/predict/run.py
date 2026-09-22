"""Phase 3's prediction: every (context, target) of the round (CLAUDE.md §4.7).

The chain, per target, is the one the rehearsal measured:

1. the context's control panel profile + the target's perturbation token
   → the predicted **perturbed panel** (the perturbation module, frozen);
2. that predicted panel → the predicted **perturbed rest** genes, through the
   mapping adapted on this context's own control cells;
3. the **knockdown prior** overwrites the target gene's own level, where the
   gene is detectably expressed in that context's controls;
4. the calibrated generator turns the per-gene fold changes into
   `cells_per_pert` distinct cells of **raw integer counts**.

Each target's block is written to `predictions/model/<context>/<target>.npz`
as it is produced and then freed, so nothing ever holds a
cells × all-genes matrix for more than one target (§7). The submission
writer reads those blocks back one at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..config import Config
from ..data import challenge as challenge_data
from ..data import manifest as manifest_mod
from ..data import reference
from ..logging_utils import get_logger
from ..models.adapters import _key as adapter_key
from ..models.core import as_index
from ..paths import DataPaths, RunPaths
from ..phases import adapt, phase1, phase2, phase3
from ..rehearsal import common
from . import generator as generator_mod
from . import knockdown as knockdown_mod

#: Fraction of `resources.max_ram_gb` a context's control-count block may take
#: when `predict.max_control_cells` is left at 0 (= "as many as fit").
CONTROL_BLOCK_RAM_FRACTION = 0.1
BYTES_PER_COUNT = 8  # int64, which is what the generator works in


@dataclass
class TargetBlock:
    """What one target's prediction produced, for the reports."""

    context: str
    target: str
    n_cells: int
    path: str
    generator: dict[str, Any]
    knockdown: dict[str, Any]
    perturbation_token: bool
    total_counts_median: float
    stored_entries: int
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "target_gene": self.target,
            "n_cells": self.n_cells,
            "path": self.path,
            "generator": self.generator,
            "knockdown": self.knockdown,
            "perturbation_token": self.perturbation_token,
            "total_counts_median": self.total_counts_median,
            "stored_entries": self.stored_entries,
            **self.detail,
        }


def control_cell_budget(cfg: Config, n_genes: int, n_available: int) -> int:
    """How many control cells to hold per context.

    `predict.max_control_cells` when it is set; otherwise a cap derived from
    `resources.max_ram_gb` (§1.2 — chunk sizes come from the RAM budget, not
    from what happens to fit the data in front of us).
    """
    if cfg.predict.max_control_cells > 0:
        return min(cfg.predict.max_control_cells, n_available)
    budget_bytes = cfg.resources.max_ram_gb * CONTROL_BLOCK_RAM_FRACTION * (1024**3)
    affordable = int(budget_bytes // max(n_genes * BYTES_PER_COUNT, 1))
    return max(1, min(affordable, n_available))


def mean_cp10k(counts: np.ndarray) -> np.ndarray:
    """Per-gene mean CP10K over a block of cells.

    The gate on the knockdown prior, and sanity check 3, are both stated in
    *mean CP10K* (§6.1) — not in `ctrl_mean`, which is the mean of
    log1p(CP10K) and is pulled far below it by the zeros. The sum is taken
    as a matrix-vector product so that no cells x genes float array is ever
    materialized.
    """
    library = np.maximum(counts.sum(axis=1), 1).astype(np.float64)
    per_gene = counts.T.astype(np.float64) @ (1.0 / library)
    return per_gene / counts.shape[0] * 1e4


def predicted_log2fc(
    model,
    context: phase3.ChallengeContext,
    target_idx: torch.Tensor | None,
    n_genes: int,
) -> tuple[np.ndarray, bool]:
    """Steps 1 and 2: the whole gene axis, as a log2 fold change vs control.

    Returns `(log2_fold_change, used_perturbation_token)`. A target with no
    embedding — one that is not a challenge gene, so the vocabulary has no
    token for it — gets no change rather than a guess (DECISIONS.md D31).
    """
    if target_idx is None:
        return np.zeros(n_genes, dtype=np.float32), False

    model.eval()
    with torch.no_grad():
        panel = phase2.predict_perturbed_panel(
            model, context, context.control_panel.unsqueeze(0), target_idx
        )
        rest = adapt.map_panel_to_rest(model, context, panel)

    full = np.zeros(n_genes, dtype=np.float32)
    full[context.panel_positions] = panel.squeeze(0).cpu().numpy()
    full[context.rest_positions] = rest.squeeze(0).cpu().numpy()
    return common.sd_units_to_log2fc(full, context.ctrl_mean, context.ctrl_std), True


def save_block(path, counts: np.ndarray) -> int:
    """Write one target's cells as a sparse CSR block; return stored entries.

    Sparse from the first moment they touch disk: the submission's cap is on
    *stored* entries (§8) and a dense block of the server's shape is 1.4 GB.
    """
    import scipy.sparse as sp

    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = sp.csr_matrix(np.asarray(counts, dtype=np.int32))
    matrix.eliminate_zeros()
    sp.save_npz(path, matrix, compressed=True)
    return int(matrix.nnz)


def run_predict(cfg: Config) -> dict[str, Any]:
    """The `predict` stage (checklist item 12)."""
    from ..models.checkpoint import load_core
    from ..models.core import build_model
    from ..priors.build import PriorFeatures
    from ..rehearsal.stage import load_policy, policy_settings
    from ..runtime import release_gpu_memory, resolve_device

    log = get_logger()
    run_paths = RunPaths(cfg)
    run_paths.ensure()
    paths = DataPaths(cfg)
    device = resolve_device(cfg.device)

    manifest = manifest_mod.load_manifest(paths.manifest)
    axis = reference.load_gene_names(paths.gene_names)
    gene_index = {symbol: i for i, symbol in enumerate(axis)}
    pert_counts = reference.load_pert_counts(paths.pert_counts, manifest.pert_col)
    targets_frame = challenge_data.load_targets(paths.phase3_targets)
    policy = load_policy(run_paths.phase3_policy)
    settings = policy_settings(policy)
    generate = generator_mod.get_generator(policy["generator"]["name"])
    prior = knockdown_mod.load_knockdown_prior(cfg, paths)

    wanted = _required_pairs(manifest, pert_counts)
    plan = _prediction_plan(targets_frame, manifest, pert_counts, wanted)

    features = PriorFeatures.load(run_paths.prior_features)
    model = build_model(cfg, features).to(device)
    contexts = phase3.load_challenge_contexts(cfg, device)
    for name in (phase1.ADAPTER, phase2.ADAPTER, phase3.ADAPTER):
        model.add_adapter(name)
    for name in contexts:
        model.add_adapter(adapt.context_adapter(name))
    payload = load_core(
        run_paths.core_checkpoint(phase3.PHASE),
        model,
        layout_hash=features.layout_hash(),
        strict=False,
    )
    # The checkpoint records adapters under their parameter-dict keys, which
    # are the adapter names with the characters a `ParameterDict` refuses
    # replaced (`models/adapters._key`).
    stored = set(payload.get("adapters", []))
    missing = [
        adapt.context_adapter(name)
        for name in contexts
        if adapter_key(adapt.context_adapter(name)) not in stored
    ]
    if missing:
        raise RuntimeError(
            f"{run_paths.core_checkpoint(phase3.PHASE)} has no adapter(s) {missing}; "
            "rerun the `phase3` stage so every context of this round is adapted"
        )

    log.info(
        "predict: %d contexts x %d perturbations, %s cells each, generator %r "
        "(threshold %.3g, scale %.3g)",
        len(contexts),
        len(pert_counts),
        manifest.cells_per_pert,
        policy["generator"]["name"],
        settings.confidence_threshold,
        settings.effect_scale,
    )

    blocks: list[TargetBlock] = []
    per_context: dict[str, Any] = {}

    for name in manifest.contexts:
        context = contexts[name]
        n_genes = len(axis)
        if context.n_genes != n_genes:
            raise ValueError(
                f"controls_{name}.h5ad has {context.n_genes} genes but "
                f"{paths.gene_names} lists {n_genes}"
            )
        model.use_adapters([adapt.context_adapter(name)])

        budget = control_cell_budget(cfg, n_genes, context.n_cells)
        pool_rng = np.random.default_rng(cfg.seed)
        pool_rows = (
            np.arange(context.n_cells)
            if budget >= context.n_cells
            else np.sort(pool_rng.choice(context.n_cells, budget, replace=False))
        )
        control_counts = context.controls.counts(pool_rows).astype(np.int64)
        control_cpm = mean_cp10k(control_counts)
        log.info(
            "  %s: %d control cells held (of %d), median library %.0f counts",
            name, control_counts.shape[0], context.n_cells,
            float(np.median(control_counts.sum(axis=1))),
        )

        knockdown_records = []
        for target in plan[name]:
            n_cells = _cells_for(target, manifest, pert_counts)
            target_position = gene_index.get(target)
            target_idx = (
                as_index([target_position], device) if target_position is not None else None
            )
            log2fc, used_token = predicted_log2fc(model, context, target_idx, n_genes)

            fold_expr, source = prior.resolve(target)
            if cfg.phase3.knockdown_source == "per_target_only" and source == knockdown_mod.POOLED:
                record = {
                    "applied": False,
                    "reason": "phase3.knockdown_source is 'per_target_only' and no "
                    "screen measured this target",
                    "control_mean_cpm": float(
                        control_cpm[target_position] if target_position is not None else np.nan
                    ),
                }
            else:
                log2fc, record = knockdown_mod.apply_to_log2fc(
                    log2fc,
                    target_position,
                    fold_expr,
                    float(control_cpm[target_position]) if target_position is not None else np.nan,
                    cfg.phase3.knockdown_min_control_cpm,
                )
            record = {
                "context": name,
                "target_gene": target,
                "fold_expr_source": source,
                "fold_expr_resolved": float(fold_expr),
                **record,
            }
            knockdown_records.append(record)

            rng = np.random.default_rng(
                common.arm_seed(f"predict:{name}", target, cfg.seed)
            )
            counts = generate(control_counts, log2fc, n_cells, rng, settings)
            path = run_paths.prediction_block(name, target)
            stored = save_block(path, counts)

            blocks.append(
                TargetBlock(
                    context=name,
                    target=target,
                    n_cells=int(counts.shape[0]),
                    path=str(path.relative_to(run_paths.root)),
                    generator=generator_mod.describe(settings, log2fc),
                    knockdown=record,
                    perturbation_token=used_token,
                    total_counts_median=float(np.median(counts.sum(axis=1))),
                    stored_entries=stored,
                )
            )
            del counts

        frame = knockdown_mod.records_to_frame(knockdown_records)
        run_paths.phase3_knockdown(name).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(run_paths.phase3_knockdown(name), index=False)

        context_blocks = [b for b in blocks if b.context == name]
        per_context[name] = {
            "n_targets": len(context_blocks),
            "n_cells": sum(b.n_cells for b in context_blocks),
            "stored_entries": sum(b.stored_entries for b in context_blocks),
            "control_cells_held": int(control_counts.shape[0]),
            "knockdown_applied": int(frame["applied"].sum()) if len(frame) else 0,
            "knockdown_table": str(
                run_paths.phase3_knockdown(name).relative_to(run_paths.root)
            ),
            "targets_without_token": [
                b.target for b in context_blocks if not b.perturbation_token
            ],
        }
        del control_counts
        release_gpu_memory()

    summary = {
        "stage": "predict",
        "device": device,
        "generator": {**policy["generator"], **settings.as_dict()},
        "gene_axis": {"n_genes": len(axis), "source": str(paths.gene_names)},
        "cells_per_pert": manifest.cells_per_pert,
        "cells_per_pert_source": (
            f"{paths.pert_counts} column {pert_counts.count_column!r}"
            if pert_counts.cells_per_pert
            else str(paths.manifest)
        ),
        "contexts": per_context,
        "n_blocks": len(blocks),
        "total_cells": sum(b.n_cells for b in blocks),
        "total_stored_entries": sum(b.stored_entries for b in blocks),
        "knockdown_prior": prior.as_dict(),
        "blocks": [b.as_dict() for b in blocks],
    }
    run_paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    run_paths.prediction_index.write_text(json.dumps(summary, indent=2, default=str))
    log.info(
        "predict done: %d blocks, %d cells, %d stored entries -> %s",
        summary["n_blocks"], summary["total_cells"], summary["total_stored_entries"],
        run_paths.predictions_dir,
    )
    return summary


# --------------------------------------------------------------------------- #
# what has to be predicted
# --------------------------------------------------------------------------- #
def _required_pairs(manifest, pert_counts) -> set[tuple[str, str]]:
    """Every context x every perturbation of `pert_counts.csv` (§8)."""
    return {(context, target) for context in manifest.contexts for target in pert_counts.targets}


def _prediction_plan(targets_frame, manifest, pert_counts, wanted) -> dict[str, list[str]]:
    """`targets.csv` (§4.7), checked against what §8 requires of the file.

    §4.7 says predict every row of `targets.csv`; §8 says the submission holds
    every context x every perturbation of `pert_counts.csv`. They agree on
    real data — and if they ever do not, that is a data problem to report, not
    one to resolve by quietly preferring one of them.
    """
    listed = {
        (str(row[manifest.context_col]), str(row[manifest.pert_col]))
        for _, row in targets_frame.iterrows()
    }
    if listed != wanted:
        only_targets = sorted(listed - wanted)[:5]
        only_wanted = sorted(wanted - listed)[:5]
        raise ValueError(
            "targets.csv and the submission requirement disagree: "
            f"{len(listed - wanted)} pair(s) only in targets.csv (e.g. {only_targets}), "
            f"{len(wanted - listed)} pair(s) only in "
            f"contexts x pert_counts.csv (e.g. {only_wanted})"
        )
    plan: dict[str, list[str]] = {context: [] for context in manifest.contexts}
    for context, target in sorted(wanted):
        plan[context].append(target)
    return plan


def _cells_for(target: str, manifest, pert_counts) -> int:
    """`cells_per_pert`, per target where the file carries one (§3.3)."""
    if pert_counts.cells_per_pert:
        return int(pert_counts.cells_per_pert[target])
    return int(manifest.cells_per_pert)
