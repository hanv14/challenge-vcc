"""The official scorer, wrapped (CLAUDE.md §6).

The six `vcc2026` metrics are computed by `cell-eval2`, never reimplemented
here. This module does four things and nothing else:

1. builds the `vcc2026` preset with `pert_col` set to this challenge's column;
2. **pins what the preset leaves loose** — the DE backend (the preset says
   `auto`, and which engine a host happens to have would otherwise change the
   numbers), the device (CPU: §1 says the scorer runs on CPU), and the thread
   counts (the preset ships `-1`, which on the shared server would start one
   thread per core — §1.2 forbids that);
3. asks for the two `nsig` diagnostics alongside the six, since they are not
   in the profile and sanity check 7 needs them;
4. adds the real control cells to an **in-memory copy** of the prediction when
   the scorer asks for a control group on both sides — which it does. The
   submission itself never carries control rows (§8).

`cell_eval2.__version__` goes into every result, because the package's own
documentation records metric semantics moving between releases.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The six that enter the leaderboard's average (§6).
SCORED = (
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
)

#: Diagnostics sanity check 7 reads. Not in the `vcc2026` profile, so they
#: are requested explicitly.
DIAGNOSTICS = ("de_wilcoxon_nsig_counts_real", "de_wilcoxon_nsig_counts_pred")


class ScorerUnavailable(RuntimeError):
    """`cell-eval2` could not be imported or its preset is missing."""


@dataclass
class OfficialScore:
    """One scoring run."""

    metrics: dict[str, float]
    per_perturbation: Any = None
    n_perturbations: int = 0
    scorer_version: str = "unknown"
    config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: The wide aggregate and the run's identity record, which are what
    #: `eval/scale.py` enrols on the leaderboard's scale. Kept out of
    #: `as_dict`: one is a polars frame, the other is provenance.
    aggregate_wide: Any = None
    run_meta: dict[str, Any] | None = None

    @property
    def scored(self) -> dict[str, float]:
        return {name: self.metrics[name] for name in SCORED if name in self.metrics}

    @property
    def complete(self) -> bool:
        return all(name in self.metrics for name in SCORED)

    def per_target(self, metric: str) -> dict[str, float]:
        """One metric's per-perturbation values, where the scorer emits them.

        Sanity check 7 asks for the *median over targets* of a ratio of two
        diagnostics, which the aggregate means cannot give.
        """
        if self.per_perturbation is None:
            return {}
        frame = self.per_perturbation.filter(self.per_perturbation["metric"] == metric)
        return {
            str(row["perturbation"]): float(row["value"])
            for row in frame.iter_rows(named=True)
            if row["value"] is not None and np.isfinite(row["value"])
        }

    def nsig_counts(self) -> dict[str, dict[str, float]]:
        """`{target: {"real": n, "pred": n}}` from the scorer's diagnostics."""
        real = self.per_target("de_wilcoxon_nsig_counts_real")
        pred = self.per_target("de_wilcoxon_nsig_counts_pred")
        return {
            target: {"real": real[target], "pred": pred.get(target, 0.0)}
            for target in sorted(real)
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "scored": self.scored,
            "complete": self.complete,
            "n_perturbations": self.n_perturbations,
            "scorer_version": self.scorer_version,
            "config": self.config,
            "warnings": self.warnings,
        }


def scorer_version() -> str:
    try:
        import importlib.metadata as metadata

        return metadata.version("cell-eval2")
    except Exception:  # noqa: BLE001 - version reporting must never fail a run
        return "unknown"


def available() -> bool:
    try:
        import cell_eval2  # noqa: F401
    except ImportError:
        return False
    return True


def build_config(cfg, pert_col: str, control_label: str):
    """The `vcc2026` preset, with everything the preset leaves loose pinned."""
    try:
        from cell_eval2 import EvalConfig, catalog
    except ImportError as exc:  # pragma: no cover - exercised by the skip path
        raise ScorerUnavailable(
            "cell-eval2 is not installed; `pip install cell-eval2` (CPU base "
            "install, no [gpu] or [gpudge] extras — see CLAUDE.md §1)"
        ) from exc

    try:
        preset = EvalConfig.from_preset("vcc2026")
    except Exception as exc:  # noqa: BLE001
        raise ScorerUnavailable(
            f"cell-eval2 {scorer_version()} has no 'vcc2026' preset: {exc}"
        ) from exc

    names, _ = catalog.resolve_metrics("vcc2026")
    metrics = list(names) + [d for d in DIAGNOSTICS if d not in names]

    threads = scorer_threads(cfg)
    resolved = dataclasses.replace(
        preset,
        pert_col=pert_col,
        control=control_label,
        metrics=metrics,
        # The preset ships -1 for both. On a 192-core shared machine that is
        # one thread per core (CLAUDE.md §1.2).
        num_threads=threads,
        gather_threads=threads,
        # The scorer runs on CPU; the GPU belongs to the model (§1).
        device="cpu",
    )
    # `auto` would let whichever DE engine is installed decide the numbers.
    return dataclasses.replace(
        resolved, de=dataclasses.replace(resolved.de, backend=cfg.eval.de_backend)
    )


def scorer_threads(cfg) -> int:
    """`resources.cpu_threads`, but never more than polars actually has.

    `cell-eval2` gathers with polars, whose pool size was fixed from
    `POLARS_MAX_THREADS` when it was imported — by `scripts/server_env.sh` on
    the server, or by the conservative default `vccp/__main__.py` sets when
    the environment is silent (§1.2). Asking it for more threads than the
    pool holds is an error, and raising the pool would override what the
    environment said, which §1.2 forbids. So take the smaller of the two.
    """
    threads = max(1, int(cfg.resources.cpu_threads))
    try:
        import polars as pl

        available = int(pl.thread_pool_size())
    except Exception:  # noqa: BLE001 — a missing polars is the scorer's problem
        return threads
    return max(1, min(threads, available))


def describe_config(config) -> dict[str, Any]:
    """The settings that decide the numbers, for the report."""
    return {
        "preset": "vcc2026",
        "pert_col": config.pert_col,
        "control": config.control,
        "input_type": config.input_type,
        "control_source": config.control_source,
        "de_backend": config.de.backend,
        "de_method": config.de.method,
        "p_adj_threshold": config.de.p_adj_threshold,
        "filter_gene_min_cpm_cell": config.filter.filter_gene_min_cpm_cell,
        "bulk_target_sum": config.bulk_target_sum,
        "target_sum": config.target_sum,
        "num_threads": config.num_threads,
        "device": config.device,
    }


def with_controls(prediction, real, pert_col: str, control_label: str):
    """Add the real control cells to a copy of the prediction, for scoring.

    The submission has no control rows (§8), and the scorer needs a control
    group on both sides. This is the in-memory copy §8 permits — nothing here
    is ever written to a file that is submitted.
    """
    import anndata as ad

    if control_label in set(prediction.obs[pert_col].astype(str)):
        return prediction

    controls = real[real.obs[pert_col].astype(str) == control_label]
    if controls.n_obs == 0:
        raise ValueError(
            f"the real data has no {control_label!r} cells, so the scorer has no "
            "control group to measure effects against"
        )
    merged = ad.concat([prediction, controls.copy()], axis=0, join="outer", merge="first")
    merged.obs_names_make_unique()
    merged.var = prediction.var.copy()
    return merged


def score(cfg, prediction, real, *, pert_col: str, control_label: str) -> OfficialScore:
    """Score a prediction against the real data with the official metrics."""
    from cell_eval2 import aggregate_metrics, aggregate_metrics_wide, compute_metrics
    from cell_eval2.baseline import build_run_meta

    config = build_config(cfg, pert_col, control_label)
    scoring_prediction = with_controls(prediction, real, pert_col, control_label)

    result = compute_metrics(scoring_prediction, real, config=config)
    aggregate = aggregate_metrics(result)

    metrics = {
        str(row["metric"]): float(row["mean"])
        for row in aggregate.iter_rows(named=True)
        if row["mean"] is not None and np.isfinite(row["mean"])
    }
    missing = [name for name in SCORED if name not in metrics]

    labels = set(real.obs[pert_col].astype(str)) - {control_label}
    return OfficialScore(
        aggregate_wide=aggregate_metrics_wide(result),
        run_meta=build_run_meta(config, real, scoring_prediction),
        metrics=metrics,
        per_perturbation=result,
        n_perturbations=len(labels),
        scorer_version=scorer_version(),
        config=describe_config(config),
        warnings=(
            [
                f"the scorer returned no value for {missing}; on small panels "
                "de_wilcoxon_lfc_nmae drops perturbations whose reference gate holds "
                "fewer than min_gate_size genes, which is a property of the data, "
                "not of the prediction"
            ]
            if missing
            else []
        ),
    )
