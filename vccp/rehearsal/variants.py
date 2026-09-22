"""The three rehearsal variants (CLAUDE.md §4.6, checklist item 10).

Each runs Phase 3's procedure on Replogle data whose answers are hidden, and
each reports the method between two reference points: an **upper bound** that
was allowed to see what the method was not, and a **no-change floor**. Neither
is ever submitted; they exist so the method's number means something.

1. **Controls-only mapping.** Fit the panel→rest mapping on one screen's
   control cells alone, feed the *true* perturbed panel of held-out targets,
   and score the predicted rest genes. Upper bound: the mapping that also
   trained on perturbed cells. Floor: no change.
2. **Cross-context.** Learn perturbation behaviour in one screen, adapt to
   the other on its controls alone, and predict its perturbed cells for the
   targets they share. This produces whole cells, so it is scored with the six
   official metrics exactly as the challenge scores them — and it is the
   closest analogue of the challenge there is.
3. **Unseen genes.** Hide a gene set from everything except the rehearsal
   context's controls, then score its perturbed predictions — the position
   about 10,000 challenge genes are really in.

Variants 1 and 3 are fed true panel values, so their official metrics are
keyed `official_panel_assisted` and never mixed with variant 2's.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..logging_utils import get_logger
from ..models.checkpoint import load_core
from ..models.core import as_index, build_model
from ..phases import adapt, phase1, phase2
from ..predict.generator import GeneratorSettings
from ..priors.build import build_priors
from ..priors.scope import controls_only_scope, cross_context_scope, unseen_genes_scope
from ..train.loop import run_training
from . import common

log = get_logger()


@dataclass
class VariantResult:
    """One variant on one context: every arm, scored."""

    variant: str
    context: str
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"variant": self.variant, "context": self.context, "arms": self.arms, **self.detail}


# --------------------------------------------------------------------------- #
# helpers shared by the variants
# --------------------------------------------------------------------------- #
def restrict_genes(tensors: phase2.ContextTensors, hidden: set[int]) -> phase2.ContextTensors:
    """A copy of a context with `hidden` challenge indices removed from its
    outputs — how a gene is hidden from training (§4.6 variant 3)."""
    if not hidden:
        return tensors
    rest = tensors.rest_idx.cpu().numpy()
    keep = np.array([i for i, g in enumerate(rest) if int(g) not in hidden], dtype=np.int64)
    # The cells a screen hands back carry every rest gene it measures, so the
    # columns have to be narrowed too — not only the indices the model decodes.
    positions = (
        np.arange(rest.size) if tensors.rest_positions is None else tensors.rest_positions
    )
    return dataclasses.replace(
        tensors,
        rest_idx=tensors.rest_idx[torch.as_tensor(keep, device=tensors.rest_idx.device)],
        rest_mean=tensors.rest_mean[keep],
        rest_std=tensors.rest_std[keep],
        rest_positions=positions[keep],
    )


def build_and_load(cfg, features, device: str, checkpoint=None):
    """A model with every adapter registered, optionally warm-started."""
    model = build_model(cfg, features).to(device)
    for name in (phase1.ADAPTER, phase2.ADAPTER, "phase3"):
        model.add_adapter(name)
    if checkpoint is not None and checkpoint.is_file():
        load_core(checkpoint, model, layout_hash=features.layout_hash(), strict=False)
    return model


def train_phase2_arm(
    cfg, model, contexts, gene_index, device: str, steps: int, label: str
):
    """Phase 2's objective, under whatever restriction the caller applied."""
    from ..train.freeze import set_trainable

    model.use_adapters([phase2.ADAPTER])
    set_trainable(
        model, label, core=cfg.phase2.unfreeze_core, adapters=[phase2.ADAPTER],
        delta=True, projections=False, heads=True,
    )
    return run_training(
        model,
        phase2.make_step(model, contexts, cfg, device, gene_index),
        steps=steps,
        cfg=cfg,
        phase=label,
        rng=np.random.default_rng(cfg.seed),
        device=device,
    )


def held_out_targets(tensors: phase2.ContextTensors, cfg, pool: list[str] | None = None) -> list[str]:
    """The targets a variant scores, capped so DE scoring stays affordable."""
    candidates = pool if pool is not None else tensors.pert_val_targets
    usable = [t for t in candidates if t in tensors.pseudobulk]
    return usable[: cfg.rehearsal.max_targets]


def real_cells_for(
    tensors: phase2.ContextTensors, targets: list[str], cfg, rng
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """The truth: each target's real perturbed cells, in raw counts."""
    blocks, labels, counts = [], [], {}
    for target in targets:
        rows = tensors.context.cells.loc[
            tensors.context.cells["target_gene"].astype(str) == target, "row"
        ].to_numpy()
        if cfg.rehearsal.cells_per_target > 0:
            rows = rows[: cfg.rehearsal.cells_per_target]
        if rows.size == 0:
            continue
        blocks.append(tensors.context.raw_counts(rows))
        labels.extend([target] * rows.size)
        counts[target] = int(rows.size)
    if not blocks:
        raise RuntimeError("no held-out target has any real perturbed cell")
    return np.vstack(blocks), labels, counts


def control_counts_for(tensors: phase2.ContextTensors, cfg, rng) -> np.ndarray:
    rows = tensors.control_rows
    if cfg.rehearsal.max_control_cells > 0 and rows.size > cfg.rehearsal.max_control_cells:
        rows = np.sort(rng.choice(rows, cfg.rehearsal.max_control_cells, replace=False))
    return tensors.context.raw_counts(rows)


def gene_axis(tensors: phase2.ContextTensors) -> tuple[list[str], np.ndarray, np.ndarray]:
    """This screen's measured genes, and their control statistics.

    `genes.csv` order — the same order `raw_counts` returns, and the axis both
    sides of the scoring live on.
    """
    frame = tensors.context.genes
    return (
        frame["gene_symbol"].astype(str).tolist(),
        frame["ctrl_mean"].to_numpy(np.float64),
        np.maximum(frame["ctrl_std"].to_numpy(np.float64), 1e-3),
    )


def panel_rest_positions(tensors: phase2.ContextTensors) -> tuple[np.ndarray, np.ndarray]:
    """Where the panel and rest genes sit in `genes.csv` order."""
    is_panel = (tensors.context.genes["split"] == "panel").to_numpy()
    return np.flatnonzero(is_panel), np.flatnonzero(~is_panel)


def score_arms(
    cfg,
    variant: str,
    predictions_by_arm: dict[str, dict[str, common.TargetPrediction]],
    control_counts: np.ndarray,
    real_counts: np.ndarray,
    real_labels: list[str],
    cells_per_target: dict[str, int],
    gene_names: list[str],
    settings: GeneratorSettings,
    control_label: str,
    key: str,
    scale_reference=None,
) -> dict[str, dict[str, Any]]:
    """Generate and score every arm, sharing the control draw (D7)."""
    truth_counts = np.vstack([real_counts, control_counts])
    truth_labels = list(real_labels) + [control_label] * control_counts.shape[0]

    arms: dict[str, dict[str, Any]] = {}
    for arm, predictions in predictions_by_arm.items():
        counts, labels = common.build_cells(
            predictions, control_counts, cells_per_target, settings, variant, cfg.seed
        )
        try:
            arms[arm] = {
                key: common.score_arm(
                    cfg, counts, labels, truth_counts, truth_labels, gene_names,
                    "target_gene", control_label, scale_reference=scale_reference,
                )
            }
        except Exception as exc:  # noqa: BLE001 — a scorer failure is a result
            arms[arm] = {key: {"error": f"{type(exc).__name__}: {exc}"}}
            log.warning("  %s/%s: the scorer failed: %s", variant, arm, exc)
    return arms


def leaderboard_scale(
    cfg, run_paths, variant: str, context: str,
    real_counts: np.ndarray, real_labels: list[str],
    control_counts: np.ndarray, gene_names: list[str], control_label: str,
):
    """The 0 = baseline / 1 = replicate ends of this dataset's scale (§6).

    Built from the real data alone, on the same truth object the arms are
    scored against. `eval.build_scale` turns it off; a dataset too small for
    the pair gives a `ScaleReference` that says so.
    """
    from ..eval import scale as scale_mod

    if not cfg.eval.build_scale:
        return scale_mod.ScaleReference(
            False, "eval.build_scale is off in the config"
        )
    truth = common.to_anndata(
        np.vstack([real_counts, control_counts]),
        list(real_labels) + [control_label] * control_counts.shape[0],
        gene_names,
        "target_gene",
    )
    return scale_mod.build(
        cfg,
        truth,
        pert_col="target_gene",
        control_label=control_label,
        outdir=run_paths.rehearsal_scale(variant, context),
        bundle_id=f"{variant}_{context}",
    )


# --------------------------------------------------------------------------- #
# variant 1 — controls-only mapping
# --------------------------------------------------------------------------- #
def run_controls_only(
    cfg, features, contexts, gene_index, device: str, run_paths
) -> list[VariantResult]:
    """Fit the mapping on controls alone, feed the true perturbed panel."""
    variant = "controls_only"
    results = []

    for name, tensors in contexts.items():
        targets = held_out_targets(tensors, cfg)
        if len(targets) < 2:
            log.info("  %s/%s: skipped, %d held-out targets", variant, name, len(targets))
            continue
        log.info("  %s/%s: %d held-out targets", variant, name, len(targets))

        rng = np.random.default_rng(cfg.seed)
        gene_names, ctrl_mean, ctrl_std = gene_axis(tensors)
        panel_positions, rest_positions = panel_rest_positions(tensors)
        control_counts = control_counts_for(tensors, cfg, rng)
        real_counts, real_labels, cells_per_target = real_cells_for(tensors, targets, cfg, rng)

        # The method: a mapping that has only ever seen control cells. Warm
        # started from Phase 1, which never saw Replogle at all.
        method_model = build_and_load(
            cfg, features, device, run_paths.core_checkpoint(phase1.PHASE)
        )
        adaptation = adapt.adapt_on_controls(
            method_model, tensors, cfg, steps=cfg.rehearsal.adapt_steps,
            device=device, rng=np.random.default_rng(cfg.seed), phase="rehearsal",
        )
        # The upper bound: the mapping that also trained on perturbed cells.
        bound_model = build_and_load(
            cfg, features, device, run_paths.core_checkpoint(phase2.PHASE)
        )
        bound_model.use_adapters([phase2.ADAPTER])

        predictions: dict[str, dict[str, common.TargetPrediction]] = {
            common.METHOD: {}, common.UPPER_BOUND: {}, common.FLOOR: {},
        }
        per_gene: dict[str, list] = {common.METHOD: [], common.UPPER_BOUND: [], common.FLOOR: []}

        for target in targets:
            # The *true* perturbed panel, which is what this variant supplies.
            panel_truth, rest_truth = tensors.draw_perturbed(
                target, cfg.phase2.cells_per_draw, np.random.default_rng(
                    common.arm_seed(variant, target, cfg.seed)
                ), device,
            )
            panel_mean = panel_truth.mean(dim=0, keepdim=True)
            rest_mean = rest_truth.mean(dim=0).cpu().numpy()

            for arm, model, adapters in (
                (common.METHOD, method_model, [adapt.context_adapter(name)]),
                (common.UPPER_BOUND, bound_model, [phase2.ADAPTER]),
            ):
                model.use_adapters(adapters)
                model.eval()
                with torch.no_grad():
                    predicted_rest = adapt.map_panel_to_rest(model, tensors, panel_mean)
                predicted = predicted_rest.squeeze(0).cpu().numpy()
                per_gene[arm].append((predicted, rest_mean))

                full = np.zeros(len(gene_names), dtype=np.float32)
                full[panel_positions] = panel_truth.mean(dim=0).cpu().numpy()
                full[rest_positions] = predicted
                predictions[arm][target] = common.TargetPrediction(
                    target,
                    common.sd_units_to_log2fc(full, ctrl_mean, ctrl_std),
                    {"arm": arm},
                )

            per_gene[common.FLOOR].append((np.zeros_like(rest_mean), rest_mean))
            floor_full = np.zeros(len(gene_names), dtype=np.float32)
            floor_full[panel_positions] = panel_truth.mean(dim=0).cpu().numpy()
            predictions[common.FLOOR][target] = common.TargetPrediction(
                target, common.sd_units_to_log2fc(floor_full, ctrl_mean, ctrl_std), {"arm": "floor"}
            )

        settings = GeneratorSettings()
        arms = score_arms(
            cfg, variant, predictions, control_counts, real_counts, real_labels,
            cells_per_target, gene_names, settings, "non-targeting", common.PANEL_ASSISTED,
        )
        for arm, pairs in per_gene.items():
            predicted = np.stack([p for p, _ in pairs])
            truth = np.stack([t for _, t in pairs])
            arms[arm]["rest_genes"] = common.per_gene_scores(predicted, truth)

        results.append(
            VariantResult(
                variant=variant,
                context=name,
                arms=arms,
                detail={
                    "n_targets": len(targets),
                    "n_rest_genes": int(rest_positions.size),
                    "adaptation": adaptation.as_dict(),
                    "note": "the panel values fed in are the truth, so the official "
                    "metrics here are panel-assisted and are not end-to-end "
                    "performance; `rest_genes` is the reading that is about the model",
                },
            )
        )
    return results


# --------------------------------------------------------------------------- #
# variant 2 — cross-context
# --------------------------------------------------------------------------- #
def run_cross_context(
    cfg, features, contexts, gene_index, device: str, run_paths, paths
) -> list[VariantResult]:
    """Learn in one screen, adapt to the other on its controls alone."""
    variant = "cross_context"
    results = []
    names = sorted(contexts)
    if len(names) < 2:
        log.info("  %s: skipped, needs two Replogle screens", variant)
        return results

    for held_out in names:
        others = [n for n in names if n != held_out]
        tensors = contexts[held_out]
        shared = sorted(
            set(tensors.pert_train_targets + tensors.pert_val_targets).intersection(
                *[
                    set(contexts[o].pert_train_targets + contexts[o].pert_val_targets)
                    for o in others
                ]
            )
        )
        targets = held_out_targets(tensors, cfg, shared)
        if len(targets) < 2:
            log.info(
                "  %s/%s: skipped, only %d target(s) shared with %s",
                variant, held_out, len(targets), ", ".join(others),
            )
            continue
        log.info(
            "  %s/%s: %d targets shared with %s", variant, held_out, len(targets), ", ".join(others)
        )

        # Priors rebuilt without the held-out screen's responses; its control
        # cells stay visible, which is what "adapt using only its controls"
        # means (DECISIONS.md D21).
        scope = cross_context_scope(held_out, targets)
        # Rebuilt in the run's own layout, so the columns keep their meaning
        # and the Phase 1 core stays loadable; what the scope hides shows up
        # as a set missing flag rather than as a shift (`conform_results`).
        scoped_features = build_priors(cfg, scope, reference_layouts=features.layouts)
        if scoped_features.layout_hash() != features.layout_hash():
            raise RuntimeError(
                "the scoped priors did not land in the run's layout: "
                f"{scoped_features.layout_hash()} vs {features.layout_hash()}"
            )

        method_model = build_and_load(cfg, scoped_features, device)
        load_core(
            run_paths.core_checkpoint(phase1.PHASE), method_model,
            layout_hash=scoped_features.layout_hash(), strict=False,
        )
        train_phase2_arm(
            cfg, method_model, {n: contexts[n] for n in others}, gene_index, device,
            cfg.rehearsal.phase2_steps, f"rehearsal:{variant}:{held_out}",
        )
        adaptation = adapt.adapt_on_controls(
            method_model, tensors, cfg, steps=cfg.rehearsal.adapt_steps,
            device=device, rng=np.random.default_rng(cfg.seed), phase="rehearsal",
        )

        # The upper bound saw the held-out screen's perturbed cells.
        bound_model = build_and_load(cfg, features, device, run_paths.core_checkpoint(phase2.PHASE))
        adapt.adapt_on_controls(
            bound_model, tensors, cfg, steps=cfg.rehearsal.adapt_steps,
            device=device, rng=np.random.default_rng(cfg.seed), phase="rehearsal_bound",
        )

        rng = np.random.default_rng(cfg.seed)
        gene_names, ctrl_mean, ctrl_std = gene_axis(tensors)
        panel_positions, rest_positions = panel_rest_positions(tensors)
        control_counts = control_counts_for(tensors, cfg, rng)
        real_counts, real_labels, cells_per_target = real_cells_for(tensors, targets, cfg, rng)

        predictions = {common.METHOD: {}, common.UPPER_BOUND: {}, common.FLOOR: {}}
        for target in targets:
            target_idx = as_index([gene_index[target]], device)
            for arm, model in ((common.METHOD, method_model), (common.UPPER_BOUND, bound_model)):
                model.use_adapters([adapt.context_adapter(held_out)])
                model.eval()
                with torch.no_grad():
                    # Control profile + perturbation token -> perturbed panel,
                    # then the adapted mapping -> the rest. The whole chain,
                    # exactly as Phase 3 will run it.
                    panel = phase2.predict_perturbed_panel(
                        model, tensors, tensors.control_panel.unsqueeze(0), target_idx
                    )
                    rest = adapt.map_panel_to_rest(model, tensors, panel)
                full = np.zeros(len(gene_names), dtype=np.float32)
                full[panel_positions] = panel.squeeze(0).cpu().numpy()
                full[rest_positions] = rest.squeeze(0).cpu().numpy()
                predictions[arm][target] = common.TargetPrediction(
                    target, common.sd_units_to_log2fc(full, ctrl_mean, ctrl_std), {"arm": arm}
                )
            predictions[common.FLOOR][target] = common.zero_prediction(target, len(gene_names))

        settings = GeneratorSettings()
        scale_reference = leaderboard_scale(
            cfg, run_paths, variant, held_out, real_counts, real_labels,
            control_counts, gene_names, "non-targeting",
        )
        arms = score_arms(
            cfg, variant, predictions, control_counts, real_counts, real_labels,
            cells_per_target, gene_names, settings, "non-targeting", common.END_TO_END,
            scale_reference=scale_reference,
        )

        results.append(
            VariantResult(
                variant=variant,
                context=held_out,
                arms=arms,
                detail={
                    "n_targets": len(targets),
                    "learned_from": others,
                    "leaderboard_scale": scale_reference.as_dict(),
                    "adaptation": adaptation.as_dict(),
                    "prior_scope": scope.describe(),
                    "predictions": {
                        target: predictions[common.METHOD][target].log2_fold_change.tolist()
                        for target in targets[: cfg.rehearsal.n_saved_predictions]
                    },
                    "note": "whole predicted cells, scored exactly as the challenge "
                    "scores them — the closest analogue of the challenge available",
                },
            )
        )
    return results


# --------------------------------------------------------------------------- #
# variant 3 — unseen genes
# --------------------------------------------------------------------------- #
def run_unseen_genes(
    cfg, features, contexts, gene_index, device: str, run_paths, axis: list[str]
) -> list[VariantResult]:
    """Hide genes from everything but the rehearsal context's controls."""
    variant = "unseen_genes"
    results = []

    for name in sorted(contexts)[: cfg.rehearsal.unseen_genes_contexts]:
        tensors = contexts[name]
        targets = held_out_targets(tensors, cfg)
        rest = tensors.rest_idx.cpu().numpy()
        n_hidden = max(1, int(round(rest.size * cfg.rehearsal.unseen_gene_fraction)))
        if len(targets) < 2 or n_hidden < 1:
            log.info("  %s/%s: skipped", variant, name)
            continue

        rng = np.random.default_rng(cfg.seed)
        hidden_positions = np.sort(rng.choice(rest.size, n_hidden, replace=False))
        hidden_idx = set(int(g) for g in rest[hidden_positions])
        hidden_symbols = [axis[i] for i in sorted(hidden_idx)]
        log.info(
            "  %s/%s: %d genes hidden from everything but this screen's controls, "
            "%d held-out targets", variant, name, len(hidden_idx), len(targets),
        )

        scope = unseen_genes_scope(hidden_symbols, name)
        # Rebuilt in the run's own layout, so the columns keep their meaning
        # and the Phase 1 core stays loadable; what the scope hides shows up
        # as a set missing flag rather than as a shift (`conform_results`).
        scoped_features = build_priors(cfg, scope, reference_layouts=features.layouts)
        if scoped_features.layout_hash() != features.layout_hash():
            raise RuntimeError(
                "the scoped priors did not land in the run's layout: "
                f"{scoped_features.layout_hash()} vs {features.layout_hash()}"
            )

        method_model = build_and_load(cfg, scoped_features, device)
        load_core(
            run_paths.core_checkpoint(phase1.PHASE), method_model,
            layout_hash=scoped_features.layout_hash(), strict=False,
        )
        # Hidden from training: removed from every screen's output genes.
        restricted = {n: restrict_genes(t, hidden_idx) for n, t in contexts.items()}
        train_phase2_arm(
            cfg, method_model, restricted, gene_index, device,
            cfg.rehearsal.phase2_steps, f"rehearsal:{variant}:{name}",
        )
        # The controls of this screen *do* see them — that is the exception.
        adaptation = adapt.adapt_on_controls(
            method_model, tensors, cfg, steps=cfg.rehearsal.adapt_steps,
            device=device, rng=np.random.default_rng(cfg.seed), phase="rehearsal",
        )

        bound_model = build_and_load(cfg, features, device, run_paths.core_checkpoint(phase2.PHASE))
        bound_model.use_adapters([phase2.ADAPTER])

        gene_names, ctrl_mean, ctrl_std = gene_axis(tensors)
        panel_positions, rest_positions = panel_rest_positions(tensors)
        control_counts = control_counts_for(tensors, cfg, rng)
        real_counts, real_labels, cells_per_target = real_cells_for(tensors, targets, cfg, rng)

        predictions = {common.METHOD: {}, common.UPPER_BOUND: {}, common.FLOOR: {}}
        per_gene: dict[str, list] = {common.METHOD: [], common.UPPER_BOUND: [], common.FLOOR: []}

        for target in targets:
            panel_truth, rest_truth = tensors.draw_perturbed(
                target, cfg.phase2.cells_per_draw,
                np.random.default_rng(common.arm_seed(variant, target, cfg.seed)), device,
            )
            panel_mean = panel_truth.mean(dim=0, keepdim=True)
            rest_mean = rest_truth.mean(dim=0).cpu().numpy()

            for arm, model, adapters in (
                (common.METHOD, method_model, [adapt.context_adapter(name)]),
                (common.UPPER_BOUND, bound_model, [phase2.ADAPTER]),
            ):
                model.use_adapters(adapters)
                model.eval()
                with torch.no_grad():
                    predicted_rest = adapt.map_panel_to_rest(model, tensors, panel_mean)
                predicted = predicted_rest.squeeze(0).cpu().numpy()
                # Scored on the hidden genes alone — the question is what the
                # model can say about a gene it was never trained to output.
                per_gene[arm].append((predicted[hidden_positions], rest_mean[hidden_positions]))

                full = np.zeros(len(gene_names), dtype=np.float32)
                full[panel_positions] = panel_truth.mean(dim=0).cpu().numpy()
                full[rest_positions] = predicted
                predictions[arm][target] = common.TargetPrediction(
                    target, common.sd_units_to_log2fc(full, ctrl_mean, ctrl_std), {"arm": arm}
                )

            per_gene[common.FLOOR].append(
                (np.zeros_like(rest_mean[hidden_positions]), rest_mean[hidden_positions])
            )
            floor_full = np.zeros(len(gene_names), dtype=np.float32)
            floor_full[panel_positions] = panel_truth.mean(dim=0).cpu().numpy()
            predictions[common.FLOOR][target] = common.TargetPrediction(
                target, common.sd_units_to_log2fc(floor_full, ctrl_mean, ctrl_std), {"arm": "floor"}
            )

        settings = GeneratorSettings()
        arms = score_arms(
            cfg, variant, predictions, control_counts, real_counts, real_labels,
            cells_per_target, gene_names, settings, "non-targeting", common.PANEL_ASSISTED,
        )
        for arm, pairs in per_gene.items():
            arms[arm]["hidden_genes"] = common.per_gene_scores(
                np.stack([p for p, _ in pairs]), np.stack([t for _, t in pairs])
            )

        results.append(
            VariantResult(
                variant=variant,
                context=name,
                arms=arms,
                detail={
                    "n_targets": len(targets),
                    "n_hidden_genes": len(hidden_idx),
                    "hidden_gene_fraction_of_rest": round(len(hidden_idx) / rest.size, 4),
                    "adaptation": adaptation.as_dict(),
                    "prior_scope": scope.describe(),
                    "note": "`hidden_genes` is the reading that is about the model; "
                    "the official metrics here are panel-assisted",
                },
            )
        )
    return results
