"""Phase 2 — Replogle (CLAUDE.md §4.5, checklist item 9).

Two objectives, one network:

1. **The panel -> rest mapping, on single cells, Siamese.** Panel values in,
   rest values out, with the *same weights* for control cells and perturbed
   cells. That is the whole point: a mapping that only held in unperturbed
   cells would be useless at Phase 3, where it has to survive a perturbation
   it was never shown. Metrics are reported separately for the two states so
   that a mapping which quietly works on only one of them is visible.

2. **The perturbation module, adapted to Replogle.** Control profile plus
   perturbation token -> perturbed panel, in control-SD units (§3.2), scored
   against both forms of the truth the data offers: the stored pseudobulk,
   and the mean of a fresh draw of that target's cells.

Both losses are masked to the genes the context measures — on the server a
Replogle screen covers about 8,200 of the 18,533 challenge genes, and
averaging in the rest would be averaging in nothing.

Targets are held out **by gene**, as in Phase 1.

The core is trainable here by default (`phase2.unfreeze_core`), and a second
arm trains with it frozen so the choice can be read off a measurement rather
than taken on trust — see `phases/forgetting.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..config import Config
from ..data import manifest as manifest_mod
from ..data import replogle as replogle_data
from ..logging_utils import get_logger
from ..models.core import as_index, zscore_across_genes
from ..paths import DataPaths
from ..train.loop import masked_mse, pearson, sample_output_genes

PHASE = "phase2"
ADAPTER = "phase2"
CONTROL, PERTURBED = "control", "perturbed"


@dataclass
class ContextTensors:
    """One Replogle screen, ready for the model."""

    name: str
    context: replogle_data.ReplogleContext
    panel_idx: torch.Tensor
    rest_idx: torch.Tensor
    panel_mean: np.ndarray
    panel_std: np.ndarray
    rest_mean: np.ndarray
    rest_std: np.ndarray
    #: The context vector's source: the shape of this screen's control
    #: profile, z-scored across genes so it is comparable between phases.
    context_profile: torch.Tensor
    #: The screen's control pseudobulk over the panel, in control-SD units —
    #: the perturbation module's input and its baseline. It has to be this
    #: stable estimate and not a fresh draw of a few control cells: the mean
    #: of n cells in SD units carries noise of variance 1/n, which at the
    #: cell counts here is several times the size of the pseudobulk response
    #: the module is trying to predict, so the baseline alone would make the
    #: prediction worse than predicting no change.
    control_panel: torch.Tensor
    control_rows: np.ndarray
    #: Every target in the screen, split for the **mapping** objective. The
    #: mapping needs no perturbation token, so a target that is not itself a
    #: challenge gene still contributes perturbed cells to it.
    train_targets: list[str] = field(default_factory=list)
    val_targets: list[str] = field(default_factory=list)
    #: The subset that *is* a challenge gene, for the **perturbation**
    #: objective. Only those have a target-role embedding (§3.1), so only
    #: those can be conditioned on. On mini that is 51 of 177 in K562_gwps.
    pert_train_targets: list[str] = field(default_factory=list)
    pert_val_targets: list[str] = field(default_factory=list)
    #: target -> pseudobulk panel profile in control-SD units.
    pseudobulk: dict[str, torch.Tensor] = field(default_factory=dict)
    #: What predicting no change scores on each objective, measured on the
    #: training split. Each loss is divided by its own, so both read as a
    #: fraction of the variance there was to explain and the configured
    #: weights mean what they say. Without this the mapping loss (~1.0 in
    #: control-SD units, one cell at a time) is twenty times the
    #: perturbation loss (~0.04, a pseudobulk mean over cells) and simply
    #: drowns it.
    mapping_no_change: float = 1.0
    perturbation_no_change: float = 1.0
    #: Which columns of the screen's rest block this context uses. None means
    #: all of them; the rehearsal's "unseen genes" variant narrows it to hide
    #: a gene from training (§4.6 variant 3).
    rest_positions: np.ndarray | None = None

    @property
    def n_panel(self) -> int:
        return int(self.panel_idx.shape[0])

    @property
    def n_rest(self) -> int:
        return int(self.rest_idx.shape[0])

    def draw_controls(self, n: int, rng, device: str):
        """`(panel, rest)` for `n` control cells — the `ControlSource`
        interface the shared adaptation routine takes (`phases/adapt.py`)."""
        return draw_cells(self, None, n, rng, device)

    def draw_perturbed(self, target: str, n: int, rng, device: str):
        return draw_cells(self, target, n, rng, device)

    def raw_control_counts(self, rows) -> np.ndarray:
        """Raw counts for control rows, over this screen's measured genes."""
        return self.context.raw_counts(rows)


def load_contexts(cfg: Config, device: str, gene_index: dict | None = None) -> dict[str, ContextTensors]:
    """Prepare every Replogle screen the scope allows."""
    from ..data import reference

    paths = DataPaths(cfg)
    manifest = manifest_mod.load_manifest(paths.manifest)
    if gene_index is None:
        axis = reference.load_gene_names(paths.gene_names)
        gene_index = {symbol: i for i, symbol in enumerate(axis)}
    rng = np.random.default_rng(cfg.seed)
    out: dict[str, ContextTensors] = {}

    for name, directory in sorted(paths.discover_phase2_contexts().items()):
        ctx = replogle_data.load_replogle_context(directory)
        mean, std = ctx.control_stats()
        is_panel = (ctx.genes["split"] == replogle_data.PANEL).to_numpy()

        targets = ctx.targets
        if len(targets) < 2:
            continue
        shuffled = rng.permutation(np.array(sorted(targets)))
        n_val = max(1, int(round(len(shuffled) * cfg.phase2.val_fraction)))

        obs, matrix, _ = replogle_data.load_pseudobulk(directory, manifest.control_label)
        labels = obs.index.astype(str).tolist()
        control_row = labels.index(manifest.control_label)

        # The context vector comes from the *un-normalized* control profile:
        # in control-SD units a control mean is zero by construction, so it
        # would say nothing about which context this is.
        control_profile = matrix[control_row][is_panel]
        control_panel = (control_profile - mean[is_panel]) / std[is_panel]

        pseudobulk = {}
        for row, label in enumerate(labels):
            if label == manifest.control_label:
                continue
            panel_profile = (matrix[row][is_panel] - mean[is_panel]) / std[is_panel]
            pseudobulk[label] = torch.as_tensor(
                panel_profile, dtype=torch.float32, device=device
            )

        train_targets_list = [t for t in sorted(shuffled[n_val:])]
        pert_train = [t for t in train_targets_list if t in gene_index]
        no_change = 1.0
        if pert_train:
            stacked = np.stack(
                [pseudobulk[t].cpu().numpy() for t in pert_train if t in pseudobulk]
            )
            no_change = max(float((stacked**2).mean()), 1e-6)

        out[name] = ContextTensors(
            name=name,
            context=ctx,
            panel_idx=as_index(ctx.panel_challenge_idx, device),
            rest_idx=as_index(ctx.rest_challenge_idx, device),
            panel_mean=mean[is_panel],
            panel_std=std[is_panel],
            rest_mean=mean[~is_panel],
            rest_std=std[~is_panel],
            context_profile=torch.as_tensor(
                zscore_across_genes(control_profile), dtype=torch.float32, device=device
            ).unsqueeze(-1),
            control_panel=torch.as_tensor(control_panel, dtype=torch.float32, device=device),
            control_rows=ctx.cells.loc[ctx.cells["is_control"], "row"].to_numpy(),
            train_targets=sorted(shuffled[n_val:]),
            val_targets=sorted(shuffled[:n_val]),
            pert_train_targets=[t for t in sorted(shuffled[n_val:]) if t in gene_index],
            pert_val_targets=[t for t in sorted(shuffled[:n_val]) if t in gene_index],
            pseudobulk=pseudobulk,
            perturbation_no_change=no_change,
        )
    usable = [name for name, t in out.items() if t.pert_train_targets]
    if not out:
        raise RuntimeError("no Replogle screen has enough targets to train on")
    if not usable:
        raise RuntimeError(
            "no Replogle target is a challenge gene, so the perturbation module has "
            "nothing to condition on"
        )
    return out


def draw_cells(
    tensors: ContextTensors, target: str | None, n: int, rng: np.random.Generator, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(panel, rest)` for `n` cells, in control-SD units.

    Cells are read through `ReplogleCells`, which fetches the rows asked for
    and nothing else — the K562 genome-wide file is ~2 million cells on the
    server (§3.3).
    """
    if target is None:
        rows = rng.choice(tensors.control_rows, min(n, tensors.control_rows.size), replace=False)
        panel, rest = tensors.context.reader.get(rows)
    else:
        panel, rest = tensors.context.cells_for_target(target, n, rng)

    if tensors.rest_positions is not None:
        rest = rest[:, tensors.rest_positions]
    panel = (panel - tensors.panel_mean) / tensors.panel_std
    rest = (rest - tensors.rest_mean) / tensors.rest_std
    return (
        torch.as_tensor(panel, dtype=torch.float32, device=device),
        torch.as_tensor(rest, dtype=torch.float32, device=device),
    )


def map_panel_to_rest(
    model,
    tensors: ContextTensors,
    panel_values: torch.Tensor,
    output_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """The Siamese mapping.

    One function, called identically for control cells and for perturbed
    ones. No perturbation token: the mapping is a statement about a cell's
    transcriptome, not about what was done to it, and making it depend on the
    perturbation is exactly the failure Phase 3 would inherit.
    """
    context = tensors.context_profile.unsqueeze(0).expand(panel_values.shape[0], -1, -1)
    latents = model.encode(
        panel_values.unsqueeze(-1), tensors.panel_idx, context_profile=context
    )
    return model.decode(latents, tensors.rest_idx if output_idx is None else output_idx)


def predict_perturbed_panel(
    model,
    tensors: ContextTensors,
    control_panel: torch.Tensor,
    target_idx: torch.Tensor,
) -> torch.Tensor:
    """Control profile + perturbation token -> perturbed panel, in SD units.

    The head predicts the **change**; a perturbed profile is mostly its own
    control, and the latent bottleneck exists for the response (§4.2).
    """
    context = tensors.context_profile.unsqueeze(0).expand(control_panel.shape[0], -1, -1)
    latents = model.encode(
        control_panel.unsqueeze(-1),
        tensors.panel_idx,
        target_idx=target_idx,
        context_profile=context,
    )
    return control_panel + model.decode(latents, tensors.panel_idx)


def make_step(model, contexts: dict[str, ContextTensors], cfg: Config, device: str, gene_index):
    """One training step: the mapping on both cell states, plus the
    perturbation module."""
    names = sorted(contexts)

    def step(rng: np.random.Generator):
        tensors = contexts[names[int(rng.integers(len(names)))]]
        half = max(1, cfg.phase2.batch_size // 2)

        # --- the Siamese mapping, on both states, with the same weights ---
        control_panel, control_rest = draw_cells(tensors, None, half, rng, device)
        target = tensors.train_targets[int(rng.integers(len(tensors.train_targets)))]
        pert_panel, pert_rest = draw_cells(tensors, target, half, rng, device)

        output_idx, positions = sample_output_genes(
            tensors.rest_idx, cfg.train.output_genes_per_step, rng
        )
        control_loss = masked_mse(
            map_panel_to_rest(model, tensors, control_panel, output_idx),
            control_rest[:, positions],
        )
        perturbed_loss = masked_mse(
            map_panel_to_rest(model, tensors, pert_panel, output_idx),
            pert_rest[:, positions],
        )
        mapping_loss = 0.5 * (control_loss + perturbed_loss) / tensors.mapping_no_change

        # --- the perturbation module ---
        pool = tensors.pert_train_targets
        if not pool:
            # This screen has no target that is a challenge gene; the mapping
            # still trains on it, the perturbation module cannot.
            loss = cfg.phase2.loss_mapping * mapping_loss + model.vocabulary.delta_penalty()
            return loss, {
                "map_control": float(control_loss.detach()),
                "map_perturbed": float(perturbed_loss.detach()),
            }
        chosen = rng.choice(pool, min(cfg.phase2.batch_size, len(pool)), replace=False)
        target_idx = as_index([gene_index[t] for t in chosen], device)
        baseline = tensors.control_panel.unsqueeze(0).expand(len(chosen), -1)
        predicted = predict_perturbed_panel(model, tensors, baseline, target_idx)

        bulk_truth = torch.stack([tensors.pseudobulk[t] for t in chosen])
        bulk_loss = masked_mse(predicted, bulk_truth)

        # The same quantity as the cells actually deliver it, which is what
        # Phase 3 will be scored on.
        cell_means = torch.stack(
            [
                draw_cells(tensors, t, cfg.phase2.cells_per_draw, rng, device)[0].mean(dim=0)
                for t in chosen
            ]
        )
        cell_loss = masked_mse(predicted, cell_means)
        perturbation_loss = (
            0.5 * (bulk_loss + cell_loss) / tensors.perturbation_no_change
        )

        loss = (
            cfg.phase2.loss_mapping * mapping_loss
            + cfg.phase2.loss_perturbation * perturbation_loss
            + model.vocabulary.delta_penalty()
        )
        return loss, {
            "map_control": float(control_loss.detach()),
            "map_perturbed": float(perturbed_loss.detach()),
            "pert_bulk": float(bulk_loss.detach()),
            "pert_cells": float(cell_loss.detach()),
        }

    return step


def evaluate(
    model, contexts: dict[str, ContextTensors], cfg: Config, device: str, gene_index
) -> dict[str, float]:
    """Held-out-target metrics, control and perturbed reported apart."""
    rng = np.random.default_rng(cfg.seed + 1)
    per_context = evaluate_per_context(model, contexts, cfg, device, gene_index, rng)
    if not per_context:
        return {}

    scores: dict[str, float] = {}
    keys = set().union(*(set(v) for v in per_context.values()))
    for key in sorted(keys):
        values = [v[key] for v in per_context.values() if key in v and np.isfinite(v[key])]
        if values:
            scores[key] = float(np.mean(values))
    return scores


def evaluate_per_context(
    model,
    contexts: dict[str, ContextTensors],
    cfg: Config,
    device: str,
    gene_index,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, float]]:
    rng = rng or np.random.default_rng(cfg.seed + 1)
    model.eval()
    out: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for name, tensors in contexts.items():
            scores: dict[str, float] = {}
            n = cfg.phase2.batch_size

            # The mapping, on each state separately (checklist item 9).
            control_panel, control_rest = draw_cells(tensors, None, n, rng, device)
            predicted = map_panel_to_rest(model, tensors, control_panel)
            scores["map_control_mse"] = float(((predicted - control_rest) ** 2).mean())
            scores["map_control_pearson"] = pearson(
                predicted.cpu().numpy(), control_rest.cpu().numpy()
            )

            held_out = [t for t in tensors.val_targets if t in tensors.pseudobulk]
            if not held_out:
                out[name] = scores
                continue

            panels, rests = [], []
            for target in held_out[: cfg.phase2.batch_size]:
                panel, rest = draw_cells(tensors, target, cfg.phase2.cells_per_draw, rng, device)
                panels.append(panel)
                rests.append(rest)
            pert_panel = torch.cat(panels)
            pert_rest = torch.cat(rests)
            predicted = map_panel_to_rest(model, tensors, pert_panel)
            scores["map_perturbed_mse"] = float(((predicted - pert_rest) ** 2).mean())
            scores["map_perturbed_pearson"] = pearson(
                predicted.cpu().numpy(), pert_rest.cpu().numpy()
            )

            # The perturbation module, on unseen targets — and on seen ones,
            # because the gap between the two is what says whether a poor
            # held-out score is a model that cannot learn or one that has
            # learned too few targets to generalize from.
            for split, pool in (
                ("pert", tensors.pert_val_targets),
                ("pert_train", tensors.pert_train_targets),
            ):
                usable = [t for t in pool if t in tensors.pseudobulk]
                if not usable:
                    continue
                usable = usable[: cfg.phase2.batch_size]
                target_idx = as_index([gene_index[t] for t in usable], device)
                baseline = tensors.control_panel.unsqueeze(0).expand(len(usable), -1)
                predicted = predict_perturbed_panel(model, tensors, baseline, target_idx)
                truth = torch.stack([tensors.pseudobulk[t] for t in usable])

                mse = float(((predicted - truth) ** 2).mean())
                no_change = float((truth**2).mean())
                scores[f"{split}_mse"] = mse
                scores[f"{split}_pearson"] = pearson(
                    predicted.cpu().numpy(), truth.cpu().numpy()
                )
                # What predicting no change would score, so the number above
                # can be read against something. Below 1.0 means the module
                # beat predicting nothing at all — the only reading of a raw
                # MSE that means anything on its own.
                scores[f"{split}_mse_no_change"] = no_change
                scores[f"{split}_mse_ratio_to_no_change"] = mse / max(no_change, 1e-9)
                scores[f"n_{split}_targets_scored"] = float(len(usable))
            out[name] = scores
    return out


def run_phase2(cfg: Config) -> dict[str, Any]:
    """The `phase2` stage, including the core-freeze ablation."""
    from ..data import reference
    from ..models.checkpoint import load_core, save_core
    from ..models.core import build_model
    from ..paths import RunPaths
    from ..priors.build import PriorFeatures
    from ..runtime import release_gpu_memory, resolve_device
    from ..train.freeze import FrozenCheck, set_trainable
    from ..train.l2sp import L2SP
    from ..train.loop import run_training
    from ..train.replay import ReplayMixer, ReplayTask
    from . import phase1

    log = get_logger()
    run_paths = RunPaths(cfg)
    run_paths.ensure()
    device = resolve_device(cfg.device)

    features = PriorFeatures.load(run_paths.prior_features)
    axis = reference.load_gene_names(DataPaths(cfg).gene_names)
    gene_index = {symbol: i for i, symbol in enumerate(axis)}

    contexts = load_contexts(cfg, device, gene_index)
    for name, t in contexts.items():
        log.info(
            "phase 2 %s: %d panel -> %d rest; mapping %d train / %d val targets; "
            "perturbation %d train / %d val (targets that are challenge genes)",
            name,
            t.n_panel,
            t.n_rest,
            len(t.train_targets),
            len(t.val_targets),
            len(t.pert_train_targets),
            len(t.pert_val_targets),
        )

    frozen_check = FrozenCheck()
    arms: dict[str, Any] = {}

    def train_arm(arm: str, unfreeze_core: bool, steps: int) -> tuple[Any, Any, dict]:
        model = build_model(cfg, features).to(device)
        for name in (phase1.ADAPTER, ADAPTER):
            model.add_adapter(name)
        load_core(
            run_paths.core_checkpoint(phase1.PHASE),
            model,
            layout_hash=features.layout_hash(),
            strict=False,
        )
        model.use_adapters([ADAPTER])

        plan = set_trainable(
            model,
            f"{PHASE}:{arm}",
            core=unfreeze_core,
            adapters=[ADAPTER],
            delta=True,
            projections=False,
            heads=True,
        )
        before = frozen_check.begin(model, plan)

        # The guard against forgetting: replay Phase 1's objective on a
        # fraction of the steps, and pull the weights toward where Phase 1
        # left them (§4.3).
        phase1_tensors = phase1.load_phase1_tensors(cfg, device)
        replay = ReplayMixer(
            tasks=[
                ReplayTask(
                    phase1.PHASE, _phase1_replay_step(model, phase1_tensors, cfg)
                )
            ],
            fraction=cfg.train.replay_fraction,
        )
        l2sp = L2SP(model, cfg.train.l2sp_weight)

        result = run_training(
            model,
            make_step(model, contexts, cfg, device, gene_index),
            steps=steps,
            cfg=cfg,
            phase=f"{PHASE}:{arm}",
            rng=np.random.default_rng(cfg.seed),
            device=device,
            replay=replay,
            l2sp=l2sp,
            evaluate=lambda: evaluate(model, contexts, cfg, device, gene_index),
        )
        frozen = frozen_check.end(model, plan, before)
        return model, result, frozen

    main_arm = "core_unfrozen" if cfg.phase2.unfreeze_core else "core_frozen"
    model, result, frozen = train_arm(main_arm, cfg.phase2.unfreeze_core, cfg.phase2.steps)
    arms[main_arm] = {
        "unfreeze_core": cfg.phase2.unfreeze_core,
        "training": result.as_dict(),
        "validation": result.final_metrics,
        "trainable": frozen,
    }

    save_core(
        run_paths.core_checkpoint(PHASE),
        model,
        phase=PHASE,
        layout_hash=features.layout_hash(),
        extra={"arm": main_arm, "validation": result.final_metrics},
    )

    # The ablation: the same phase with the opposite core policy, so the
    # default rests on a measurement (PLAN.md §8.9).
    if cfg.train.core_freeze_ablation:
        other = "core_frozen" if cfg.phase2.unfreeze_core else "core_unfrozen"
        steps = max(1, int(round(cfg.phase2.steps * cfg.train.ablation_steps_fraction)))
        log.info("phase 2 ablation: %s for %d steps", other, steps)
        ablation_model, ablation_result, ablation_frozen = train_arm(
            other, not cfg.phase2.unfreeze_core, steps
        )
        arms[other] = {
            "unfreeze_core": not cfg.phase2.unfreeze_core,
            "training": ablation_result.as_dict(),
            "validation": ablation_result.final_metrics,
            "trainable": ablation_frozen,
        }
        save_core(
            run_paths.core_checkpoint(f"{PHASE}_{other}"),
            ablation_model,
            phase=f"{PHASE}:{other}",
            layout_hash=features.layout_hash(),
            extra={"arm": other, "validation": ablation_result.final_metrics},
        )
        del ablation_model
        release_gpu_memory()

    metrics = {
        "phase": PHASE,
        "device": device,
        "main_arm": main_arm,
        "data": {
            name: {
                "n_panel": t.n_panel,
                "n_rest": t.n_rest,
                "n_train_targets": len(t.train_targets),
                "n_val_targets": len(t.val_targets),
                "n_pert_train_targets": len(t.pert_train_targets),
                "n_pert_val_targets": len(t.pert_val_targets),
                "n_control_cells": int(t.control_rows.size),
                "perturbation_no_change_variance": t.perturbation_no_change,
            }
            for name, t in contexts.items()
        },
        "arms": arms,
        "validation_per_context": evaluate_per_context(
            model, contexts, cfg, device, gene_index
        ),
    }

    run_paths.phase_dir(PHASE).mkdir(parents=True, exist_ok=True)
    run_paths.phase_metrics(PHASE).write_text(json.dumps(metrics, indent=2, default=str))
    import pandas as pd

    pd.DataFrame(result.curve).to_csv(run_paths.phase_curves(PHASE), index=False)

    existing = (
        json.loads(run_paths.frozen_check.read_text())
        if run_paths.frozen_check.is_file()
        else {"phases": {}}
    )
    merged = frozen_check.report()
    merged["phases"] = {**existing.get("phases", {}), **merged["phases"]}
    merged["all_ok"] = all(p["frozen_verified"]["ok"] for p in merged["phases"].values())
    run_paths.frozen_check.write_text(json.dumps(merged, indent=2, default=str))

    log.info(
        "phase 2 done: %s  ->  %s",
        "  ".join(f"{k} {v:.4f}" for k, v in result.final_metrics.items()),
        run_paths.phase_metrics(PHASE),
    )
    return metrics


def _phase1_replay_step(model, tensors, cfg: Config):
    """Phase 1's own objective, under Phase 1's adapter.

    Replaying Phase 1's batches through Phase 2's loss would measure
    nothing, so the task carries its own loss *and* switches to the adapter
    the objective belongs to.
    """
    from . import phase1

    inner = phase1.make_step(model, tensors, cfg)

    def step(rng: np.random.Generator):
        model.use_adapters([phase1.ADAPTER])
        try:
            return inner(rng)
        finally:
            model.use_adapters([ADAPTER])

    return step
