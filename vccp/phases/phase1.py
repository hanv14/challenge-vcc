"""Phase 1 — LINCS (CLAUDE.md §4.4, checklist item 8).

    control + perturbation  ->  signature  ->  perturbed profile

Two steps, trained jointly: the first predicts the signature (`sig`, and the
GMT summary `gmt` where a row has one), the second predicts the perturbed
profile from the control *and the signature the first step just produced*. So
the gradient of the perturbed-profile loss reaches the signature head, which
is what "train jointly" buys over training them separately.

**Validation is by held-out target gene**, not by held-out row: the question
is whether the model generalizes to a perturbation it has not seen, and a
different cell line's copy of the same knockout would answer an easier one.
Results are reported per cell line too, because a model that works on one
line and not another is a different thing from one that works on average.

**Only rows whose target is a challenge gene are used.** The perturbation
token is the target's target-role embedding, and the vocabulary covers the
challenge genes (§3.1) — a LINCS target outside that axis has no embedding,
so its row would train the model to ignore the perturbation token. On
`mini_data` that keeps 177 of 688 rows; the count is reported in
`phase1/metrics.json` either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data import lincs as lincs_data
from ..data import reference
from ..logging_utils import get_logger
from ..models.core import as_index, zscore_across_genes
from ..paths import DataPaths
from ..train.loop import masked_mse, pearson, sample_output_genes

PHASE = "phase1"
ADAPTER = "phase1"
#: Standard deviations below which a gene is treated as flat.
MIN_STD = 1e-6


@dataclass
class Phase1Tensors:
    """Everything Phase 1 trains on, already on the device."""

    gene_idx: torch.Tensor
    ctrl: torch.Tensor
    pert: torch.Tensor
    sig: torch.Tensor
    gmt: torch.Tensor
    has_gmt: torch.Tensor
    target_idx: torch.Tensor
    cell_line: np.ndarray
    targets: np.ndarray
    #: Cell line -> pooled control profile, the context vector's source.
    context: dict[str, torch.Tensor] = field(default_factory=dict)
    train_rows: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    val_rows: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    n_rows_total: int = 0
    n_targets_total: int = 0

    @property
    def n_genes(self) -> int:
        return int(self.gene_idx.shape[0])


def load_phase1_tensors(cfg: Config, device: str) -> Phase1Tensors:
    """Read the Phase 1 file and prepare it for the model."""
    paths = DataPaths(cfg)
    axis = reference.load_gene_names(paths.gene_names)
    index = {symbol: i for i, symbol in enumerate(axis)}

    data = lincs_data.load_phase1(paths.phase1_lincs)
    labels = data.obs["target_gene"].astype(str).to_numpy()
    usable = np.array([label in index for label in labels])
    if usable.sum() < 2:
        raise RuntimeError(
            "no LINCS row has a target that is a challenge gene, so the perturbation "
            "token would carry no information"
        )

    rows = np.flatnonzero(usable)
    targets = labels[rows]
    cell_line = data.obs["cell_line"].astype(str).to_numpy()[rows]

    # Split by target gene: an unseen perturbation, not an unseen row.
    unique_targets = np.array(sorted(set(targets)))
    rng = np.random.default_rng(cfg.seed)
    shuffled = rng.permutation(unique_targets)
    n_val = max(1, int(round(len(shuffled) * cfg.phase1.val_fraction)))
    val_targets = set(shuffled[:n_val])

    is_val = np.array([t in val_targets for t in targets])
    train_rows = np.flatnonzero(~is_val)
    val_rows = np.flatnonzero(is_val)
    if train_rows.size < 1 or val_rows.size < 1:
        raise RuntimeError(
            f"the target split left {train_rows.size} training and {val_rows.size} "
            "validation rows; lower phase1.val_fraction"
        )

    # Standardize the two expression layers per gene, using training rows only.
    ctrl = data.layers["ctrl"][rows]
    pert = data.layers["pert"][rows]
    ctrl_std = _standardize(ctrl, train_rows)
    pert_std = _standardize(pert, train_rows)

    to_tensor = lambda array: torch.as_tensor(array, dtype=torch.float32, device=device)  # noqa: E731
    tensors = Phase1Tensors(
        gene_idx=as_index(data.challenge_idx, device),
        ctrl=to_tensor(ctrl_std),
        pert=to_tensor(pert_std),
        sig=to_tensor(data.layers["sig"][rows]),
        gmt=to_tensor(data.layers["gmt"][rows]),
        has_gmt=torch.as_tensor(
            data.obs["has_gmt"].to_numpy(bool)[rows], dtype=torch.bool, device=device
        ),
        target_idx=as_index([index[t] for t in targets], device),
        cell_line=cell_line,
        targets=targets,
        train_rows=train_rows,
        val_rows=val_rows,
        n_rows_total=int(data.n_rows),
        n_targets_total=len(set(labels)),
    )

    # The context vector is the cell line's pooled control profile, computed
    # from training rows so validation cannot leak through it.
    for line in sorted(set(cell_line)):
        in_line = np.flatnonzero(cell_line == line)
        pooled_from = np.intersect1d(in_line, train_rows)
        if pooled_from.size == 0:
            pooled_from = in_line
        pooled = tensors.ctrl[pooled_from].mean(dim=0)
        tensors.context[line] = torch.as_tensor(
            zscore_across_genes(pooled.cpu().numpy()), dtype=torch.float32, device=device
        )
    return tensors


def _standardize(values: np.ndarray, reference_rows: np.ndarray) -> np.ndarray:
    mean = values[reference_rows].mean(axis=0)
    std = values[reference_rows].std(axis=0)
    return ((values - mean) / np.where(std < MIN_STD, 1.0, std)).astype(np.float32)


def _context_for(tensors: Phase1Tensors, rows: np.ndarray) -> torch.Tensor:
    """(B, G, 1) pooled control profiles, one per row's cell line."""
    stacked = torch.stack([tensors.context[line] for line in tensors.cell_line[rows]])
    return stacked.unsqueeze(-1)


def predict(
    model,
    tensors: Phase1Tensors,
    rows: np.ndarray,
    cfg: Config,
    output_idx: torch.Tensor | None = None,
    output_positions: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The two steps. Returns `(sig_hat, gmt_hat, pert_hat)`.

    `output_idx` restricts which genes are decoded; the whole panel is fed
    in either way, since the input is what the latents are built from.
    """
    index = torch.as_tensor(rows, dtype=torch.long, device=tensors.ctrl.device)
    if output_idx is None:
        output_idx, output_positions = tensors.gene_idx, np.arange(tensors.n_genes)
    control = tensors.ctrl.index_select(0, index)
    target_idx = tensors.target_idx.index_select(0, index)
    context = _context_for(tensors, rows)

    # Step 1: control + perturbation -> signature.
    blank = torch.zeros_like(control)
    latents = model.encode(
        torch.stack([control, blank], dim=-1),
        tensors.gene_idx,
        target_idx=target_idx,
        context_profile=context,
    )
    # The signature is decoded over the **whole** input set even when the
    # loss only scores a subsample, because step 2 consumes it as a per-gene
    # input channel — a partial signature would change what step 2 sees.
    sig_full = model.decode(latents, tensors.gene_idx, head="sig")
    sig_hat = sig_full[:, output_positions]
    gmt_hat = model.decode(latents, output_idx, head="gmt")

    # Step 2: control + signature -> perturbed profile. Feeding the *predicted*
    # signature is what makes the two steps train jointly (§4.4).
    signature = tensors.sig.index_select(0, index) if cfg.phase1.teacher_forcing else sig_full
    latents = model.encode(
        torch.stack([control, signature], dim=-1),
        tensors.gene_idx,
        target_idx=target_idx,
        context_profile=context,
    )
    # The head predicts the **change** from the control, not the perturbed
    # level. A perturbed profile is mostly its own control, and pushing that
    # through a 64-latent bottleneck would spend the whole model on copying
    # the input; the bottleneck is there for the response (§4.2, §3.2).
    baseline = control[:, output_positions]
    pert_hat = baseline + model.decode(latents, output_idx, head="value")
    return sig_hat, gmt_hat, pert_hat


def make_step(model, tensors: Phase1Tensors, cfg: Config):
    """A training step over a random batch of training rows."""

    def step(rng: np.random.Generator):
        size = min(cfg.phase1.batch_size, tensors.train_rows.size)
        rows = rng.choice(tensors.train_rows, size, replace=False)
        index = torch.as_tensor(rows, dtype=torch.long, device=tensors.ctrl.device)
        output_idx, positions = sample_output_genes(
            tensors.gene_idx, cfg.train.output_genes_per_step, rng
        )

        sig_hat, gmt_hat, pert_hat = predict(
            model, tensors, rows, cfg, output_idx, positions
        )
        truth = lambda layer: layer.index_select(0, index)[:, positions]  # noqa: E731
        sig_loss = masked_mse(sig_hat, truth(tensors.sig))
        # `gmt` is masked to the rows that have one: `has_gmt` is false for a
        # good fraction of the Phase 1 rows (§3.3).
        gmt_mask = tensors.has_gmt.index_select(0, index).unsqueeze(-1)
        gmt_loss = masked_mse(gmt_hat, truth(tensors.gmt), gmt_mask)
        pert_loss = masked_mse(pert_hat, truth(tensors.pert))

        loss = (
            cfg.phase1.loss_sig * sig_loss
            + cfg.phase1.loss_gmt * gmt_loss
            + cfg.phase1.loss_pert * pert_loss
        )
        loss = loss + model.vocabulary.delta_penalty()
        return loss, {
            "sig": float(sig_loss.detach()),
            "gmt": float(gmt_loss.detach()),
            "pert": float(pert_loss.detach()),
        }

    return step


def evaluate(model, tensors: Phase1Tensors, cfg: Config, rows: np.ndarray | None = None) -> dict:
    """Held-out-target metrics, overall and per cell line."""
    rows = tensors.val_rows if rows is None else rows
    if rows.size == 0:
        return {}

    model.eval()
    with torch.no_grad():
        sig_hat, gmt_hat, pert_hat = predict(model, tensors, rows, cfg)
        index = torch.as_tensor(rows, dtype=torch.long, device=tensors.ctrl.device)
        truth = {
            "sig": tensors.sig.index_select(0, index),
            "gmt": tensors.gmt.index_select(0, index),
            "pert": tensors.pert.index_select(0, index),
        }
        predicted = {"sig": sig_hat, "gmt": gmt_hat, "pert": pert_hat}
        gmt_rows = tensors.has_gmt.index_select(0, index).cpu().numpy()

    scores: dict[str, float] = {}
    for name in ("sig", "gmt", "pert"):
        a = predicted[name].cpu().numpy()
        b = truth[name].cpu().numpy()
        if name == "gmt":
            a, b = a[gmt_rows], b[gmt_rows]
            if a.size == 0:
                continue
        scores[f"{name}_mse"] = float(np.mean((a - b) ** 2))
        scores[f"{name}_pearson"] = pearson(a, b)
    return scores


def evaluate_per_cell_line(model, tensors: Phase1Tensors, cfg: Config) -> dict[str, dict]:
    """The same metrics, split by cell line (§4.4)."""
    out = {}
    for line in sorted(set(tensors.cell_line[tensors.val_rows])):
        rows = tensors.val_rows[tensors.cell_line[tensors.val_rows] == line]
        if rows.size == 0:
            continue
        out[line] = {"n_rows": int(rows.size), **evaluate(model, tensors, cfg, rows)}
    return out


def run_phase1(cfg: Config) -> dict[str, Any]:
    """The `phase1` stage."""
    from ..models.checkpoint import save_core
    from ..models.core import build_model
    from ..paths import RunPaths
    from ..priors.build import PriorFeatures
    from ..runtime import resolve_device
    from ..train.freeze import FrozenCheck, set_trainable
    from ..train.loop import run_training

    log = get_logger()
    run_paths = RunPaths(cfg)
    run_paths.ensure()
    device = resolve_device(cfg.device)

    features = PriorFeatures.load(run_paths.prior_features)
    model = build_model(cfg, features).to(device)
    model.add_adapter(ADAPTER)
    model.use_adapters([ADAPTER])

    tensors = load_phase1_tensors(cfg, device)
    log.info(
        "phase 1: %d of %d rows usable (%d of %d targets are challenge genes), "
        "%d train / %d val rows over %d cell lines",
        tensors.train_rows.size + tensors.val_rows.size,
        tensors.n_rows_total,
        len(set(tensors.targets)),
        tensors.n_targets_total,
        tensors.train_rows.size,
        tensors.val_rows.size,
        len(set(tensors.cell_line)),
    )

    # Phase 1 is where the core is learned, so everything trains here.
    plan = set_trainable(
        model, PHASE, core=True, adapters=[ADAPTER], delta=True, projections=True, heads=True
    )
    frozen_check = FrozenCheck()
    before = frozen_check.begin(model, plan)

    result = run_training(
        model,
        make_step(model, tensors, cfg),
        steps=cfg.phase1.steps,
        cfg=cfg,
        phase=PHASE,
        rng=np.random.default_rng(cfg.seed),
        device=device,
        evaluate=lambda: evaluate(model, tensors, cfg),
    )
    frozen = frozen_check.end(model, plan, before)

    per_cell_line = evaluate_per_cell_line(model, tensors, cfg)
    metrics = {
        "phase": PHASE,
        "device": device,
        "data": {
            "n_rows_usable": int(tensors.train_rows.size + tensors.val_rows.size),
            "n_rows_total": tensors.n_rows_total,
            "n_targets_usable": len(set(tensors.targets)),
            "n_targets_total": tensors.n_targets_total,
            "n_train_rows": int(tensors.train_rows.size),
            "n_val_rows": int(tensors.val_rows.size),
            "n_genes": tensors.n_genes,
            "n_cell_lines": len(set(tensors.cell_line)),
            "note": "only rows whose target is a challenge gene are used; the "
            "perturbation token has no embedding for any other target",
        },
        "training": result.as_dict(),
        "validation": result.final_metrics,
        "validation_per_cell_line": per_cell_line,
        "trainable": frozen,
    }

    run_paths.phase_dir(PHASE).mkdir(parents=True, exist_ok=True)
    run_paths.phase_metrics(PHASE).write_text(json.dumps(metrics, indent=2, default=str))
    pd.DataFrame(result.curve).to_csv(run_paths.phase_curves(PHASE), index=False)
    save_core(
        run_paths.core_checkpoint(PHASE),
        model,
        phase=PHASE,
        layout_hash=features.layout_hash(),
        extra={"validation": result.final_metrics},
    )
    run_paths.frozen_check.write_text(json.dumps(frozen_check.report(), indent=2, default=str))

    log.info(
        "phase 1 done: %s  ->  %s",
        "  ".join(f"{k} {v:.4f}" for k, v in result.final_metrics.items()),
        run_paths.phase_metrics(PHASE),
    )
    return metrics
