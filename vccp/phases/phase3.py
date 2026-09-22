"""Phase 3 — the challenge (CLAUDE.md §4.7, checklist item 11).

A challenge context arrives as **control cells and nothing else**. So the only
thing that can be fitted on it is the panel→rest mapping, where both halves
are observed in a control cell; what a knockdown *does* was learned in Phases
1 and 2 from data this context does not have, and the perturbation module
stays frozen.

What adapts is not decided here — it is read from `phase3_policy.json`, which
the rehearsal wrote from measurements (§4.6). This module obeys it and records
the loss before and after, per context, in `phase3/adapt_<context>.json`.
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
from ..models.core import as_index, zscore_across_genes
from ..paths import DataPaths, RunPaths
from . import adapt, phase1, phase2

PHASE = "phase3"
ADAPTER = "phase3"
MIN_CTRL_STD = 1e-3


@dataclass
class ChallengeContext:
    """One challenge context, in the shape the model and generator need.

    Satisfies the `ControlSource` protocol, so it goes through exactly the
    same adaptation routine the rehearsal used.
    """

    name: str
    controls: challenge_data.ChallengeControls
    panel_idx: torch.Tensor
    rest_idx: torch.Tensor
    panel_positions: np.ndarray
    rest_positions: np.ndarray
    ctrl_mean: np.ndarray
    ctrl_std: np.ndarray
    context_profile: torch.Tensor
    control_panel: torch.Tensor
    device: str = "cpu"
    _lognorm: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_genes(self) -> int:
        return self.controls.n_genes

    @property
    def n_cells(self) -> int:
        return self.controls.n_cells

    def lognorm(self) -> np.ndarray:
        """All control cells in log1p(CP10K), cached for the adaptation.

        A challenge context has a few hundred to a few thousand control cells
        over the challenge gene axis, so this is far smaller than a Replogle
        screen and holding it is what makes the adaptation quick.
        """
        if self._lognorm is None:
            self._lognorm = self.controls.lognorm()
        return self._lognorm

    def standardized(self) -> np.ndarray:
        return (self.lognorm() - self.ctrl_mean) / self.ctrl_std

    def draw_controls(self, n: int, rng, device: str):
        values = self.standardized()
        rows = rng.choice(values.shape[0], min(n, values.shape[0]), replace=False)
        block = torch.as_tensor(values[rows], dtype=torch.float32, device=device)
        return block[:, self.panel_positions], block[:, self.rest_positions]


def load_challenge_contexts(cfg: Config, device: str) -> dict[str, ChallengeContext]:
    """Every context of this round, from its control file."""
    from ..data import gene_split as gene_split_mod

    paths = DataPaths(cfg)
    manifest = manifest_mod.load_manifest(paths.manifest)
    split = gene_split_mod.load_gene_split(paths.gene_split)
    panel_positions = split.panel_idx
    rest_positions = split.rest_idx

    out: dict[str, ChallengeContext] = {}
    for context in manifest.contexts:
        controls = challenge_data.load_challenge_controls(
            paths.phase3_controls(context), context
        )
        ctrl_mean, ctrl_std = controls.control_stats()
        ctrl_mean = ctrl_mean.astype(np.float64)
        ctrl_std = np.maximum(ctrl_std.astype(np.float64), MIN_CTRL_STD)

        # The context vector: the shape of this context's control profile,
        # z-scored across genes so one encoder serves all three phases.
        profile = zscore_across_genes(ctrl_mean[panel_positions])
        out[context] = ChallengeContext(
            name=context,
            controls=controls,
            panel_idx=as_index(panel_positions, device),
            rest_idx=as_index(rest_positions, device),
            panel_positions=panel_positions,
            rest_positions=rest_positions,
            ctrl_mean=ctrl_mean,
            ctrl_std=ctrl_std,
            context_profile=torch.as_tensor(
                profile, dtype=torch.float32, device=device
            ).unsqueeze(-1),
            # In control-SD units the control mean is zero by construction —
            # which is exactly the stable baseline the perturbation module
            # needs (DECISIONS.md D27).
            control_panel=torch.zeros(
                panel_positions.size, dtype=torch.float32, device=device
            ),
            device=device,
        )
    return out


def run_phase3(cfg: Config) -> dict[str, Any]:
    """The `phase3` stage: adapt on each context's controls, per the policy."""
    from ..models.checkpoint import load_core, save_core
    from ..models.core import build_model
    from ..priors.build import PriorFeatures
    from ..rehearsal.stage import load_policy
    from ..runtime import release_gpu_memory, resolve_device
    from ..train.freeze import FrozenCheck

    log = get_logger()
    run_paths = RunPaths(cfg)
    run_paths.ensure()
    device = resolve_device(cfg.device)

    policy = load_policy(run_paths.phase3_policy)
    features = PriorFeatures.load(run_paths.prior_features)

    model = build_model(cfg, features).to(device)
    for name in (phase1.ADAPTER, phase2.ADAPTER, ADAPTER):
        model.add_adapter(name)
    load_core(
        run_paths.core_checkpoint(phase2.PHASE),
        model,
        layout_hash=features.layout_hash(),
        strict=False,
    )

    contexts = load_challenge_contexts(cfg, device)
    log.info(
        "phase 3: %d contexts, %d panel -> %d rest genes each",
        len(contexts),
        next(iter(contexts.values())).panel_idx.shape[0],
        next(iter(contexts.values())).rest_idx.shape[0],
    )
    if policy["adapt"]["core"] or policy["adapt"]["perturbation_module"]:
        raise RuntimeError(
            f"{run_paths.phase3_policy} permits adapting the core or the perturbation "
            "module in Phase 3; §4.7 requires the perturbation module to stay frozen"
        )

    frozen_check = FrozenCheck()
    per_context = {}
    run_paths.phase_dir(PHASE).mkdir(parents=True, exist_ok=True)

    for name, context in contexts.items():
        result = adapt.adapt_on_controls(
            model,
            context,
            cfg,
            steps=cfg.phase3.adapt_steps,
            device=device,
            rng=np.random.default_rng(cfg.seed),
            phase=PHASE,
            train_delta=policy["adapt"]["delta"],
            frozen_check=frozen_check,
            batch_size=cfg.phase3.batch_size,
        )
        record = {
            **result.as_dict(),
            "policy": policy["adapt"],
            "n_control_cells": context.n_cells,
        }
        run_paths.phase3_adapt(name).write_text(json.dumps(record, indent=2, default=str))
        per_context[name] = {
            key: record[key]
            for key in ("loss_before", "loss_after", "loss_improvement",
                        "pearson_before", "pearson_after", "steps", "adapter")
        }
        release_gpu_memory()

    save_core(
        run_paths.core_checkpoint(PHASE),
        model,
        phase=PHASE,
        layout_hash=features.layout_hash(),
        extra={"contexts": sorted(contexts), "policy": policy["adapt"]},
    )

    existing = (
        json.loads(run_paths.frozen_check.read_text())
        if run_paths.frozen_check.is_file()
        else {"phases": {}}
    )
    merged = frozen_check.report()
    merged["phases"] = {**existing.get("phases", {}), **merged["phases"]}
    merged["all_ok"] = all(p["frozen_verified"]["ok"] for p in merged["phases"].values())
    run_paths.frozen_check.write_text(json.dumps(merged, indent=2, default=str))

    metrics = {
        "phase": PHASE,
        "device": device,
        "policy": policy["adapt"],
        "contexts": per_context,
        "n_genes": next(iter(contexts.values())).n_genes,
    }
    run_paths.phase_metrics(PHASE).write_text(json.dumps(metrics, indent=2, default=str))
    log.info("phase 3 done -> %s", run_paths.phase_metrics(PHASE))
    return metrics
