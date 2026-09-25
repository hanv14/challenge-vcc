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

import hashlib
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
from ..runtime import seed_everything
from ..train.loop import masked_mse, pearson, sample_output_genes

PHASE = "phase2"
ADAPTER = "phase2"
#: The two ways the perturbation module can be supervised (§4.5 and
#: PLAN_PERCELL.md §3). `POOLED` is a setting, not a code fork: it stays in
#: the codebase as the control arm the per-cell mode is measured against.
POOLED, PERCELL = "pooled", "percell"
#: The arm trained with no Phase 1 input at all (PLAN_PERCELL.md §7.6).
SCRATCH_ARM = "core_scratch"
#: What Phase 2 may inherit from Phase 1 (DECISIONS.md D80).
FULL, WITHOUT_DELTA, NONE = "full", "without_delta", "none"


def initialization_fingerprint(model) -> str:
    """A short digest of a freshly built model's weights.

    The ablation arms are only comparable if they all start from the same
    random initialization. That is a property of the run, not of the code,
    so each arm records the digest of the weights it was built with and the
    contribution report says whether they agree — across arms and, since the
    digest is stable for a given seed and layout, across runs (D84).
    """
    digest = hashlib.blake2b(digest_size=8)
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def reset_delta(model) -> int:
    """Zero the gene vocabulary's free per-gene vectors, keeping everything else.

    `delta` is 4.74M of the model's 5.65M trainable parameters, and Phase 1
    only ever tokenizes the 955 panel genes — so a full warm start hands
    Phase 2 an embedding table where the panel has been moved by a bulk
    knockout objective and the other ~17,578 genes sit at `W . prior` alone.
    Phase 2's whole job is mapping the panel onto those rest genes, and it
    starts that job with the two halves on different footings.

    Zeroing `delta` is what §4.1 says it starts at, so this hands over the
    network Phase 1 learned without the per-gene memorization it learned it
    on. Returns how many tensors were reset, for the record.
    """
    import torch

    reset = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("vocabulary.") and "delta" in name:
                parameter.zero_()
                reset += 1
    return reset
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
    tensors: ContextTensors,
    target: str | None,
    n: int,
    rng: np.random.Generator,
    device: str,
    rows: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(panel, rest)` for `n` cells, in control-SD units.

    Cells are read through `ReplogleCells`, which fetches the rows asked for
    and nothing else — the K562 genome-wide file is ~2 million cells on the
    server (§3.3).

    `rows` names exactly which cells to read instead of sampling them, for
    either state. Two callers need it. `_score_delta` estimates its noise
    floor from two control draws, and two draws sampled independently overlap
    — two overlapping means differ by less than sampling noise, so the floor
    would come out too low and the mapping would look better than it is. The
    coupling diagnostic needs it because it has to know *which* cells it drew
    in order to look up their library sizes.
    """
    if rows is not None:
        panel, rest = tensors.context.reader.get(rows)
    elif target is None:
        rows = rng.choice(
            tensors.control_rows, min(n, tensors.control_rows.size), replace=False
        )
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
    pert_type: str | None = None,
) -> torch.Tensor:
    """Control profile + perturbation token -> perturbed panel, in SD units.

    The head predicts the **change**; a perturbed profile is mostly its own
    control, and the latent bottleneck exists for the response (§4.2).

    `pert_type` names the assay's perturbation modality. Every caller passes
    it from its own phase's config key rather than defaulting, so that the
    model is never asked to treat a knockout and a knockdown as the same
    thing by accident (PLAN_PERCELL.md §7.4).
    """
    context = tensors.context_profile.unsqueeze(0).expand(control_panel.shape[0], -1, -1)
    latents = model.encode(
        control_panel.unsqueeze(-1),
        tensors.panel_idx,
        target_idx=target_idx,
        context_profile=context,
        pert_type=pert_type,
    )
    return control_panel + model.decode(latents, tensors.panel_idx)


def _delta_loss(predicted_change: torch.Tensor, observed_change: torch.Tensor) -> torch.Tensor:
    """Squared error on the *change*, as a fraction of the change there was.

    Normalized by the batch's own observed change so it reads as "against
    predicting no change" and so `phase2.loss_delta` means what it says
    against the other two terms, which are normalized the same way.

    The observed change is a difference of two sampled means, so it carries
    sampling noise of its own — at these cell counts a good deal of it. That
    inflates the value this returns but does not move its minimum: the
    expected squared error against `true + noise` is minimized at `true`.
    What the noise does spoil is the *reading*, which is why
    `evaluate_per_context` measures the floor a perfect predictor would hit
    and reports it beside the ratio.
    """
    scale = float((observed_change**2).mean())
    return ((predicted_change - observed_change) ** 2).mean() / max(scale, 1e-6)


def _no_change_ratio(
    predicted: torch.Tensor, truth: torch.Tensor, baseline: torch.Tensor
) -> torch.Tensor:
    """Squared error against what predicting `baseline` would have cost.

    In per-cell mode "no change" means handing back the control cell
    untouched, so the baseline is the control cell rather than a pseudobulk
    of zeros. Below 1.0 means the module is doing something.
    """
    scale = float(((baseline - truth) ** 2).mean())
    return ((predicted - truth) ** 2).mean() / max(scale, 1e-6)


def percell_losses(
    model,
    tensors: ContextTensors,
    cfg: Config,
    rng: np.random.Generator,
    device: str,
    gene_index,
    output_idx: torch.Tensor,
    positions: np.ndarray,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
    """The per-cell perturbation loss and the per-pair delta, from one coupling.

    A cell in, that cell perturbed out (PLAN_PERCELL.md §3). Replogle cannot
    supply "that cell perturbed" — it is a destructive assay, so the control
    and perturbed cells are different cells — and the OT coupling is what
    constructs the missing target: for each control cell, the
    coupling-weighted average of the perturbed cells it was matched to.

    The same coupling pairs the cells, so it applies to their *rest* values
    too. That is what makes the delta term per-pair here rather than by group
    mean: recomputing a second coupling would both cost another Sinkhorn and
    risk pairing the two halves of one cell differently.

    Returns `(perturbation_loss, delta_loss, metrics)`, with `None` for the
    losses when the screen has no target the vocabulary can condition on.
    """
    from ..train import ot

    pool = tensors.pert_train_targets
    if not pool:
        return None, None, {}

    chosen = rng.choice(
        pool, min(cfg.phase2.ot_targets_per_step, len(pool)), replace=False
    )
    n_cells = cfg.phase2.ot_batch_cells

    perturbation_losses, delta_losses, partners, drifts = [], [], [], []
    for target in chosen:
        control_panel, control_rest = draw_cells(tensors, None, n_cells, rng, device)
        pert_panel, pert_rest = draw_cells(tensors, target, n_cells, rng, device)
        if control_panel.shape[0] < 2 or pert_panel.shape[0] < 2:
            continue

        paired_panel, log_coupling = ot.paired_target(
            control_panel,
            pert_panel,
            cfg.phase2.ot_epsilon,
            iterations=cfg.phase2.ot_iterations,
        )

        target_idx = as_index(
            [gene_index[target]] * int(control_panel.shape[0]), device
        )
        predicted_panel = predict_perturbed_panel(
            model, tensors, control_panel, target_idx, cfg.phase2.pert_type
        )
        perturbation_losses.append(
            _no_change_ratio(predicted_panel, paired_panel, control_panel)
        )

        # The delta term, per pair (§5). By default the decoder is fed the
        # OT-paired *true* panel, so the term trains the mapping on changes
        # without letting a bad perturbation prediction corrupt it;
        # `delta_through_prediction` trains the composition end to end
        # instead, which is what Phase 3 runs at inference.
        source_panel = (
            predicted_panel if cfg.phase2.delta_through_prediction else paired_panel
        )
        predicted_change = map_panel_to_rest(
            model, tensors, source_panel, output_idx
        ) - map_panel_to_rest(model, tensors, control_panel, output_idx)
        paired_rest = ot.barycentric_target(log_coupling, pert_rest[:, positions])
        delta_losses.append(
            _delta_loss(predicted_change, paired_rest - control_rest[:, positions])
        )

        diagnostics = ot.coupling_diagnostics(log_coupling)
        partners.append(diagnostics["effective_partners"])
        drifts.append(diagnostics["marginal_drift"])

    if not perturbation_losses:
        return None, None, {}

    perturbation_loss = torch.stack(perturbation_losses).mean()
    delta_loss = torch.stack(delta_losses).mean()
    return (
        perturbation_loss,
        delta_loss,
        {
            "pert_percell": float(perturbation_loss.detach()),
            "map_delta": float(delta_loss.detach()),
            # Where on the epsilon sweep the coupling actually landed, as
            # opposed to where the config asked for it to land. If this sits
            # at the batch size the coupling is uniform and the per-cell loss
            # is the pooled loss under another name (PLAN_PERCELL.md §11).
            "ot_effective_partners": float(np.mean(partners)),
            "ot_marginal_drift": float(np.mean(drifts)),
        },
    )


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
        control_predicted = map_panel_to_rest(model, tensors, control_panel, output_idx)
        perturbed_predicted = map_panel_to_rest(model, tensors, pert_panel, output_idx)
        control_loss = masked_mse(control_predicted, control_rest[:, positions])
        perturbed_loss = masked_mse(perturbed_predicted, pert_rest[:, positions])
        mapping_loss = 0.5 * (control_loss + perturbed_loss) / tensors.mapping_no_change

        # --- the perturbation module, and the delta term that rides with it ---
        #
        # The delta consistency term (PLAN_PERCELL.md §5): the two arms above
        # share weights, but nothing else constrains the *difference* between
        # them — and that difference is the only quantity the six official
        # metrics ever score. Each arm's MSE is dominated by the baseline
        # expression level, so a mapping can look healthy on both and still
        # get the rest-gene change badly wrong.
        #
        # Which pairing supplies it depends on the mode. `pooled` has two
        # independent cell draws and so can only pair by group mean;
        # `percell` has the OT coupling, which pairs the cells themselves and
        # therefore their rest values too.
        if cfg.phase2.perturbation_mode == PERCELL:
            perturbation_loss, delta_loss, extra = percell_losses(
                model, tensors, cfg, rng, device, gene_index, output_idx, positions
            )
            if delta_loss is None:
                delta_loss = _delta_loss(
                    perturbed_predicted.mean(dim=0) - control_predicted.mean(dim=0),
                    pert_rest[:, positions].mean(dim=0)
                    - control_rest[:, positions].mean(dim=0),
                )
        else:
            delta_loss = _delta_loss(
                perturbed_predicted.mean(dim=0) - control_predicted.mean(dim=0),
                pert_rest[:, positions].mean(dim=0)
                - control_rest[:, positions].mean(dim=0),
            )
            perturbation_loss, extra = _pooled_perturbation_loss(
                model, tensors, cfg, rng, device, gene_index
            )

        metrics = {
            "map_control": float(control_loss.detach()),
            "map_perturbed": float(perturbed_loss.detach()),
            "map_delta": float(delta_loss.detach()),
            **extra,
        }
        loss = (
            cfg.phase2.loss_mapping * mapping_loss
            + cfg.phase2.loss_delta * delta_loss
            + model.regularization()
        )
        if perturbation_loss is not None:
            loss = loss + cfg.phase2.loss_perturbation * perturbation_loss
        return loss, metrics

    return step


def _pooled_perturbation_loss(
    model,
    tensors: ContextTensors,
    cfg: Config,
    rng: np.random.Generator,
    device: str,
    gene_index,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """One mean effect per target: control pseudobulk in, pseudobulk out."""
    pool = tensors.pert_train_targets
    if not pool:
        # This screen has no target that is a challenge gene; the mapping
        # still trains on it, the perturbation module cannot.
        return None, {}

    chosen = rng.choice(pool, min(cfg.phase2.batch_size, len(pool)), replace=False)
    target_idx = as_index([gene_index[t] for t in chosen], device)
    baseline = tensors.control_panel.unsqueeze(0).expand(len(chosen), -1)
    predicted = predict_perturbed_panel(
        model, tensors, baseline, target_idx, cfg.phase2.pert_type
    )

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
    loss = 0.5 * (bulk_loss + cell_loss) / tensors.perturbation_no_change
    return loss, {
        "pert_bulk": float(bulk_loss.detach()),
        "pert_cells": float(cell_loss.detach()),
    }


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

            scores.update(_score_delta(model, tensors, cfg, device, rng, held_out))
            scores.update(
                _score_percell(model, tensors, cfg, device, rng, gene_index, held_out)
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
                usable = usable[: cfg.phase2.eval_targets or len(usable)]
                # In blocks, so the number of targets scored is a choice
                # about the metric's noise rather than about how much fits
                # in one forward pass.
                predicted_blocks, truth_blocks = [], []
                for start in range(0, len(usable), cfg.phase2.batch_size):
                    block = usable[start : start + cfg.phase2.batch_size]
                    target_idx = as_index([gene_index[t] for t in block], device)
                    baseline = tensors.control_panel.unsqueeze(0).expand(len(block), -1)
                    predicted_blocks.append(
                        predict_perturbed_panel(
                            model, tensors, baseline, target_idx, cfg.phase2.pert_type
                        )
                    )
                    truth_blocks.append(
                        torch.stack([tensors.pseudobulk[t] for t in block])
                    )
                predicted = torch.cat(predicted_blocks)
                truth = torch.cat(truth_blocks)

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


def _score_delta(
    model,
    tensors: ContextTensors,
    cfg: Config,
    device: str,
    rng: np.random.Generator,
    held_out: list[str],
) -> dict[str, float]:
    """How well the mapping predicts the *change*, and the floor it can reach.

    The two per-state MSEs above are dominated by the baseline expression
    level; this is the quantity the official metrics actually score
    (PLAN_PERCELL.md §5).

    Two of the four numbers exist because a difference of sampled means is
    noisy. `map_delta_ratio_to_no_change` is the ratio as measured, and
    `map_delta_noise_floor_ratio` is what a *perfect* predictor would score
    at these cell counts — estimated from two **disjoint** control draws,
    whose mean difference is pure sampling noise of the same size. A ratio at
    the floor means the mapping has learned everything the data can show it;
    a ratio at 1.0 means it has learned nothing about the change, whatever
    the per-state MSEs say.
    """
    targets = held_out[: cfg.phase2.delta_eval_targets]
    if not targets:
        return {}

    # Two **disjoint** halves of one draw: the first is the baseline every
    # target's change is measured against, the second exists only to estimate
    # the floor. Sampling them independently would let them share cells and
    # understate the noise.
    available = int(tensors.control_rows.size)
    n_cells = min(cfg.phase2.delta_eval_cells, available // 2)
    if n_cells < 2:
        return {}
    drawn = rng.choice(tensors.control_rows, 2 * n_cells, replace=False)

    control_panel, control_rest = draw_cells(
        tensors, None, n_cells, rng, device, rows=drawn[:n_cells]
    )
    control_predicted = map_panel_to_rest(model, tensors, control_panel)
    control_true_mean = control_rest.mean(dim=0)
    control_predicted_mean = control_predicted.mean(dim=0)

    errors, observed, scored = [], [], 0
    for target in targets:
        pert_panel, pert_rest = draw_cells(tensors, target, n_cells, rng, device)
        if pert_panel.shape[0] < 2:
            continue
        predicted_change = (
            map_panel_to_rest(model, tensors, pert_panel).mean(dim=0)
            - control_predicted_mean
        )
        observed_change = pert_rest.mean(dim=0) - control_true_mean
        errors.append(float(((predicted_change - observed_change) ** 2).mean()))
        observed.append(float((observed_change**2).mean()))
        scored += 1

    if not scored:
        return {}

    # The floor: two disjoint control draws differ by sampling noise alone,
    # and noise of that same size sits inside every observed change above.
    _, other_rest = draw_cells(
        tensors, None, n_cells, rng, device, rows=drawn[n_cells:]
    )
    floor = float(((other_rest.mean(dim=0) - control_true_mean) ** 2).mean())

    mse = float(np.mean(errors))
    no_change = float(np.mean(observed))
    return {
        "map_delta_mse": mse,
        "map_delta_no_change": no_change,
        "map_delta_ratio_to_no_change": mse / max(no_change, 1e-9),
        "map_delta_noise_floor_ratio": floor / max(no_change, 1e-9),
        "n_delta_targets_scored": float(scored),
        # The draw the two numbers above were measured at — the floor is a
        # function of it, so a run that had to shrink it (a screen with few
        # control cells) reports a higher floor for that reason alone.
        "n_delta_cells_per_draw": float(n_cells),
    }


def _score_percell(
    model,
    tensors: ContextTensors,
    cfg: Config,
    device: str,
    rng: np.random.Generator,
    gene_index,
    held_out: list[str],
) -> dict[str, float]:
    """The per-cell question, scored for **both** modes.

    A cell in, that cell perturbed out, against the OT-paired truth. It is
    measured whichever mode trained the model, because the A/B needs a number
    both arms can be read on: `pert_mse_ratio_to_no_change` beside it asks
    the pooled question of both arms, and this asks the per-cell question of
    both. An arm that wins one and loses the other is saying something; two
    arms each scored only on their own objective would say nothing.

    `ot_effective_partners` comes back with it because a coupling that turned
    out uniform makes this metric identical to the pooled one, and that has
    to be visible rather than inferred (PLAN_PERCELL.md §11).
    """
    from ..train import ot

    targets = [
        t for t in held_out if t in tensors.pseudobulk and t in gene_index
    ][: cfg.phase2.delta_eval_targets]
    if not targets:
        return {}

    n_cells = cfg.phase2.ot_batch_cells
    errors, baselines, partners, scored = [], [], [], 0

    with torch.no_grad():
        for target in targets:
            control_panel, _ = draw_cells(tensors, None, n_cells, rng, device)
            pert_panel, _ = draw_cells(tensors, target, n_cells, rng, device)
            if control_panel.shape[0] < 2 or pert_panel.shape[0] < 2:
                continue

            paired, log_coupling = ot.paired_target(
                control_panel,
                pert_panel,
                cfg.phase2.ot_epsilon,
                iterations=cfg.phase2.ot_iterations,
            )
            target_idx = as_index(
                [gene_index[target]] * int(control_panel.shape[0]), device
            )
            predicted = predict_perturbed_panel(
                model, tensors, control_panel, target_idx, cfg.phase2.pert_type
            )
            errors.append(float(((predicted - paired) ** 2).mean()))
            baselines.append(float(((control_panel - paired) ** 2).mean()))
            partners.append(ot.coupling_diagnostics(log_coupling)["effective_partners"])
            scored += 1

    if not scored:
        return {}

    mse = float(np.mean(errors))
    no_change = float(np.mean(baselines))
    return {
        "pert_percell_mse": mse,
        "pert_percell_no_change": no_change,
        "pert_percell_ratio_to_no_change": mse / max(no_change, 1e-9),
        "n_percell_targets_scored": float(scored),
        "ot_effective_partners": float(np.mean(partners)),
        "ot_batch_cells": float(n_cells),
    }


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

    def train_arm(
        arm: str, unfreeze_core: bool, steps: int, warm_start: str = FULL
    ) -> tuple[Any, Any, dict]:
        """One Phase 2 arm.

        `warm_start` is what the Phase 1 contribution ablation varies:
        `none` starts from the priors alone, with neither Phase 1's
        checkpoint nor its replayed objective, so the gap between it and a
        warm arm is what the first phase is worth (PLAN_PERCELL.md §7.6).
        `without_delta` takes the network but not the per-gene memorization.
        The priors stay in every case — this ablates Phase 1, not §4.1.

        Every arm is re-seeded here, immediately before its weights are
        drawn. The run is seeded once at start-up, so without this an arm's
        random initialization depends on how much of the global torch stream
        the *preceding* arms consumed — which differs whenever the arm list
        differs. That made the ablation's own control arm move between runs
        in which it could not legitimately move at all (DECISIONS.md D84).
        """
        seed_everything(cfg.seed)
        model = build_model(cfg, features).to(device)
        fingerprint = initialization_fingerprint(model)
        for name in (phase1.ADAPTER, ADAPTER):
            model.add_adapter(name)
        if warm_start != NONE:
            load_core(
                run_paths.core_checkpoint(phase1.PHASE),
                model,
                layout_hash=features.layout_hash(),
                strict=False,
            )
            if warm_start == WITHOUT_DELTA:
                reset_delta(model)
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
        # left them (§4.3). An arm with no warm start has no Phase 1 weights
        # to be pulled toward and no Phase 1 knowledge to forget, so it gets
        # neither — replaying Phase 1 there would put back through the side
        # door exactly what the ablation removes.
        if warm_start != NONE:
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
        else:
            replay = ReplayMixer(tasks=[], fraction=0.0)
            l2sp = None

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
            # The ablations are read off validation, so they have to be read
            # off the *best* validation rather than whatever the last step
            # left behind. A run that diverges late otherwise reports the
            # wreck (DECISIONS.md D74).
            select_on=SELECTION_METRIC,
            # Every selection metric here is a ratio against predicting no
            # change, so 1.0 is what the metric reads when nothing has been
            # learned. Telling the loop that is what makes `signal_retained`
            # readable — and `final / best` is not, on a metric anchored this
            # way (D85).
            select_anchor=1.0,
        )
        frozen = frozen_check.end(model, plan, before)
        return model, result, frozen, fingerprint

    def arm_record(
        arm: str,
        unfreeze_core: bool,
        result,
        frozen,
        steps: int,
        warm: str,
        fingerprint: str,
    ):
        replayed = sum((result.replay or {}).get("steps_replayed", {}).values())
        divergence, instability = result.divergence, result.instability
        # Two ways a run can be untrustworthy, and `final / best` only sees
        # one: an arm that swings to 2.46 mid-run and lands at 1.25 reads as
        # steady at 1.15x. The swing is the thing that says the optimisation
        # is not under control.
        retained = result.signal_retained
        diverged = (
            (divergence is not None and divergence > DIVERGENCE_RATIO)
            or (instability is not None and instability > DIVERGENCE_RATIO)
            # The reading `final / best` cannot give. An arm that went 0.817
            # to 1.001 kept none of what it learned and still read as 1.22x,
            # just under the 1.25 threshold (D85).
            or (retained is not None and retained < SIGNAL_RETAINED_FLOOR)
        )
        if diverged:
            log.warning(
                "  %s is unstable on %s: best %.4f at step %s, worst %.4f at step %s, "
                "ended %.4f (%.2fx best; swing %.2fx; %.0f%% of what it learned was "
                "still there at the end). The arm keeps its best weights, but an arm "
                "that swings this far is not converging — lower train.lr.",
                arm, SELECTION_METRIC,
                result.best_metrics.get(SELECTION_METRIC, float("nan")), result.best_step,
                result.worst_value if result.worst_value is not None else float("nan"),
                result.worst_step,
                result.final_metrics.get(SELECTION_METRIC, float("nan")),
                divergence if divergence is not None else float("nan"),
                instability if instability is not None else float("nan"),
                100 * retained if retained is not None else float("nan"),
            )
        return {
            "unfreeze_core": unfreeze_core,
            # What this arm took from Phase 1: "full", "without_delta" or
            # "none". `full` includes the gene vocabulary's delta, which is
            # 84% of the trainable parameters and which Phase 1 trained on
            # the 955 panel genes alone (D80).
            "warm_start": warm,
            "steps": steps,
            # What the arm actually spent on Phase 2's own objective. A warm
            # arm gives `replay_fraction` of its steps to Phase 1 and a
            # scratch arm gives none, so `train.match_phase2_steps` sets the
            # scratch arm's budget to compensate (D73).
            "n_phase2_steps": steps - replayed,
            "n_phase1_replay_steps": replayed,
            "training": result.as_dict(),
            # What the ablations compare: the best validation this arm
            # reached, not what the last step happened to leave (D74).
            "validation": result.best_metrics or result.final_metrics,
            "validation_final": result.final_metrics,
            "best_step": result.best_step,
            "divergence_final_over_best": divergence,
            "instability_worst_over_best": instability,
            # The share of what the arm learned that its last step still
            # held: 1.0 ended at its best, 0.0 ended knowing no more than
            # the no-change baseline, negative ended worse than that.
            "signal_retained": retained,
            "signal_retained_worst": result.signal_retained_worst,
            # Which weights this arm's checkpoint holds.
            "weights_kept": result.restored,
            "diverged": diverged,
            # The weights this arm was built with, before any warm start.
            # Equal across arms is the precondition for comparing them.
            "init_fingerprint": fingerprint,
            "trainable": frozen,
        }

    main_arm = "core_unfrozen" if cfg.phase2.unfreeze_core else "core_frozen"
    if cfg.phase2.warm_start == NONE:
        log.warning(
            "phase2.warm_start is 'none': the shipped model starts from the priors, "
            "not from Phase 1. That is a deviation from CLAUDE.md §4's three-phase "
            "design and is recorded as one."
        )
    elif cfg.phase2.warm_start == WITHOUT_DELTA:
        log.info(
            "phase2.warm_start is 'without_delta': taking Phase 1's network but "
            "resetting the gene vocabulary's per-gene vectors, which Phase 1 trained "
            "on the 955 panel genes alone"
        )
    model, result, frozen, fingerprint = train_arm(
        main_arm, cfg.phase2.unfreeze_core, cfg.phase2.steps,
        warm_start=cfg.phase2.warm_start,
    )
    arms[main_arm] = arm_record(
        main_arm, cfg.phase2.unfreeze_core, result, frozen, cfg.phase2.steps,
        cfg.phase2.warm_start, fingerprint,
    )

    save_core(
        run_paths.core_checkpoint(PHASE),
        model,
        phase=PHASE,
        layout_hash=features.layout_hash(),
        extra={
            "arm": main_arm,
            "weights_kept": result.restored,
            "validation": result.best_metrics if result.restored == "best"
            else result.final_metrics,
        },
    )

    # The ablation: the same phase with the opposite core policy, so the
    # default rests on a measurement (PLAN.md §8.9).
    if cfg.train.core_freeze_ablation:
        other = "core_frozen" if cfg.phase2.unfreeze_core else "core_unfrozen"
        steps = max(1, int(round(cfg.phase2.steps * cfg.train.ablation_steps_fraction)))
        log.info("phase 2 ablation: %s for %d steps", other, steps)
        ablation_model, ablation_result, ablation_frozen, ablation_fingerprint = train_arm(
            other, not cfg.phase2.unfreeze_core, steps
        )
        arms[other] = arm_record(
            other, not cfg.phase2.unfreeze_core, ablation_result, ablation_frozen, steps,
            FULL, ablation_fingerprint,
        )
        save_core(
            run_paths.core_checkpoint(f"{PHASE}_{other}"),
            ablation_model,
            phase=f"{PHASE}:{other}",
            layout_hash=features.layout_hash(),
            extra={
                "arm": other,
                "weights_kept": ablation_result.restored,
                "validation": ablation_result.best_metrics if ablation_result.restored == "best"
                else ablation_result.final_metrics,
            },
        )
        del ablation_model
        release_gpu_memory()

    # The Phase 1 contribution ablation: the same phase with no Phase 1 at
    # all. The gap between this arm and the warm ones is what the first phase
    # is worth, and until it was measured the three-phase design rested on an
    # assumption (PLAN_PERCELL.md §7.6).
    if cfg.train.phase1_contribution_ablation and cfg.phase2.warm_start == NONE:
        log.info(
            "phase 2: skipping the core_scratch ablation — the main arm already "
            "trains without Phase 1, so the arm would be a duplicate"
        )
    elif cfg.train.phase1_contribution_ablation:
        steps = scratch_arm_steps(cfg)
        log.info("phase 2 ablation: core_scratch (no Phase 1) for %d steps", steps)
        scratch_model, scratch_result, scratch_frozen, scratch_fingerprint = train_arm(
            SCRATCH_ARM, cfg.phase2.unfreeze_core, steps, warm_start=NONE
        )
        arms[SCRATCH_ARM] = arm_record(
            SCRATCH_ARM,
            cfg.phase2.unfreeze_core,
            scratch_result,
            scratch_frozen,
            steps,
            NONE,
            scratch_fingerprint,
        )
        save_core(
            run_paths.core_checkpoint(f"{PHASE}_{SCRATCH_ARM}"),
            scratch_model,
            phase=f"{PHASE}:{SCRATCH_ARM}",
            layout_hash=features.layout_hash(),
            extra={
                "arm": SCRATCH_ARM,
                "weights_kept": scratch_result.restored,
                "validation": scratch_result.best_metrics if scratch_result.restored == "best"
                else scratch_result.final_metrics,
            },
        )
        del scratch_model
        release_gpu_memory()

    metrics = {
        "phase": PHASE,
        "device": device,
        "main_arm": main_arm,
        "perturbation_mode": cfg.phase2.perturbation_mode,
        "warm_start": cfg.phase2.warm_start,
        "ot": {
            "epsilon": cfg.phase2.ot_epsilon,
            "batch_cells": cfg.phase2.ot_batch_cells,
            "targets_per_step": cfg.phase2.ot_targets_per_step,
            "iterations": cfg.phase2.ot_iterations,
            "delta_through_prediction": cfg.phase2.delta_through_prediction,
        },
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
        "phase1_contribution": _phase1_contribution(arms),
        "validation_per_context": evaluate_per_context(
            model, contexts, cfg, device, gene_index
        ),
        # Which model the block above describes. It is the main arm's
        # **final** state, because that is the checkpoint saved and the one
        # Phase 3 inherits — while `arms[...]["validation"]` is each arm's
        # *best* (D74). When `best_step` is not the last step those are two
        # different models, and a reader comparing them without knowing that
        # is comparing a model with itself at another moment.
        "validation_per_context_describes": {
            "arm": main_arm,
            # The model this block was measured on is the one in memory,
            # which is the one saved — the best step when the weights were
            # restored to it (D85), the last step otherwise.
            "step": result.best_step if result.restored == "best" else cfg.phase2.steps,
            "best_step": result.best_step,
            "is_the_best_step": result.restored == "best"
            or result.best_step == cfg.phase2.steps,
            "weights_kept": result.restored,
            "note": "the saved checkpoint, which is what Phase 3 inherits",
        },
    }

    contribution = metrics["phase1_contribution"]
    if contribution.get("arms_share_initialization") is False:
        log.warning(
            "phase 2: the arms did not start from the same weights (%s). The "
            "ablations compare arms, so any difference between them is partly "
            "the initialization — do not read phase1_contribution from this run.",
            contribution.get("init_fingerprints"),
        )

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


#: What "better" means for a Phase 2 arm, and so which step each arm is
#: judged at. The perturbation module is the thing the ablations are about,
#: and this is its headline ratio: below 1 beats predicting no change.
SELECTION_METRIC = "pert_mse_ratio_to_no_change"

#: A final validation this much worse than the run's own best is a
#: divergence, not a result. The first server Phase 2 ended at 2.86x.
DIVERGENCE_RATIO = 1.25
#: Below this share of what it learned still being there at the last step,
#: an arm is unstable however small `final / best` looks (D85).
SIGNAL_RETAINED_FLOOR = 0.5

#: The validation keys the Phase 1 contribution is read off. Each is a ratio
#: against predicting no change, so lower is better on all of them and they
#: are comparable between arms.
CONTRIBUTION_KEYS = (
    "pert_mse_ratio_to_no_change",
    "pert_train_mse_ratio_to_no_change",
    "map_delta_ratio_to_no_change",
)


#: Budgets within this of each other count as matched. Replay is sampled per
#: step rather than scheduled, so two arms meant to train equally land a few
#: steps apart; reporting that as a bias would be reporting noise.
BUDGET_TOLERANCE = 0.02


def scratch_arm_steps(cfg: Config) -> int:
    """How long the `core_scratch` arm trains.

    With `train.match_phase2_steps` it gets the warm arm's *Phase 2* steps
    rather than its total. The warm arm spends `replay_fraction` of its steps
    on Phase 1's objective and the scratch arm spends none, so equal totals
    mean unequal training on the objective the two are compared on — at the
    defaults, 36% more for the scratch arm. An advantage measured that way is
    partly just the extra steps, and the number cannot be read.
    """
    steps = cfg.phase2.steps * cfg.train.ablation_steps_fraction
    if cfg.train.match_phase2_steps:
        steps *= 1.0 - cfg.train.replay_fraction
    return max(1, int(round(steps)))


def _phase1_contribution(arms: dict[str, Any]) -> dict[str, Any]:
    """What warm-starting from Phase 1 bought, per metric.

    `scratch - warm` on a ratio where lower is better, so a **positive**
    number means Phase 1 helped by that much.

    Reported rather than judged, because the arms rarely spend equal effort
    on Phase 2's own objective. A warm arm gives `train.replay_fraction` of
    its steps to Phase 1, and the scratch arm's total budget is
    `train.ablation_steps_fraction` of the main arm's — so which way the
    inequality runs depends on both settings and cannot be stated once and
    for all. `n_phase2_steps` carries the two counts and `favours` names the
    arm the difference helps, so the reading is available rather than
    assumed.
    """
    scratch = arms.get(SCRATCH_ARM)
    warm = next(
        (record for name, record in arms.items() if name != SCRATCH_ARM), None
    )
    if scratch is None or warm is None:
        return {"measured": False, "reason": "both a warm arm and core_scratch are needed"}

    deltas = {}
    for key in CONTRIBUTION_KEYS:
        a, b = scratch["validation"].get(key), warm["validation"].get(key)
        if a is not None and b is not None:
            deltas[key] = float(a) - float(b)
    warm_steps = warm.get("n_phase2_steps")
    scratch_steps = scratch.get("n_phase2_steps")
    favours = None
    if warm_steps is not None and scratch_steps is not None:
        larger = max(warm_steps, scratch_steps, 1)
        if abs(warm_steps - scratch_steps) / larger >= BUDGET_TOLERANCE:
            favours = SCRATCH_ARM if scratch_steps > warm_steps else "warm"
    diverged = sorted(name for name, r in arms.items() if r.get("diverged"))
    fingerprints = {
        name: record.get("init_fingerprint")
        for name, record in arms.items()
        if record.get("init_fingerprint")
    }
    shared = len(set(fingerprints.values())) == 1 if fingerprints else None
    return {
        "measured": bool(deltas),
        "warm_arm": next(name for name in arms if name != SCRATCH_ARM),
        "read_at": "each arm's best validation, not its last step",
        # An arm that diverged is still comparable *at its best*, but the
        # checkpoint saved is the final one, so the number here and the model
        # on disk are no longer the same thing.
        "arms_that_diverged": diverged,
        "lower_is_better": True,
        "positive_means_phase1_helped": True,
        "improvement": deltas,
        "n_phase2_steps": {"warm": warm_steps, "scratch": scratch_steps},
        # Which arm the unequal budgets help, or `None` when they match to
        # within `BUDGET_TOLERANCE`. A result that survives a budget running
        # against it is stronger than the number alone suggests; a result the
        # budget runs *for* cannot be read at all.
        "budget_favours": favours,
        "budgets_matched": favours is None,
        # The other precondition, and the one that failed silently until it
        # was recorded: every arm must start from the same random weights.
        # The digest is also stable across runs for a given seed and layout,
        # so two runs' reports can be compared (D84).
        "init_fingerprints": fingerprints,
        "arms_share_initialization": shared,
    }


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
