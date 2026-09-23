"""The `sanity` stage: measure the predictions, then judge them (§6.1).

Every block is read **once**. The same pass feeds §8's rule audit (check 1)
and the per-context statistics the other checks need, so nothing is read
twice and no cells × all-genes matrix for more than one target is ever held.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Config
from ..logging_utils import get_logger
from ..paths import DataPaths, RunPaths
from ..submit import validator
from ..submit.spec import SubmissionSpec, load_spec
from . import checks as check_mod
from .checks import CheckResult, cp10k, lognorm, mean_cp10k

#: Guards a division by a control level of zero. The *log-ratio* floor is
#: `predict.cpm_floor`, so that a measured change sits on the same scale as
#: the change the generator was asked to make.
EPSILON = 1e-12


@dataclass
class ContextStats:
    """What one context's predictions look like, measured."""

    name: str
    median_control_library: float
    median_predicted_library: float = 0.0
    median_library_ratio: float = float("nan")
    unmoved_outside_control_range: int = 0
    unmoved_fraction_outside: float = 0.0
    worst_unmoved: list[dict[str, Any]] = field(default_factory=list)
    moved_over_max_abs_log2fc: int = 0
    moved_fraction_over: float = 0.0
    worst_moved: list[dict[str, Any]] = field(default_factory=list)
    max_abs_log2fc: float = 0.0
    median_target_correlation: float = float("nan")
    identical_target_pairs: list[list[str]] = field(default_factory=list)
    variance_ratios: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "median_control_library": self.median_control_library,
            "median_predicted_library": self.median_predicted_library,
            "median_library_ratio": self.median_library_ratio,
            "unmoved_outside_control_range": self.unmoved_outside_control_range,
            "unmoved_fraction_outside": self.unmoved_fraction_outside,
            "worst_unmoved": self.worst_unmoved,
            "moved_over_max_abs_log2fc": self.moved_over_max_abs_log2fc,
            "moved_fraction_over": self.moved_fraction_over,
            "worst_moved": self.worst_moved,
            "max_abs_log2fc": self.max_abs_log2fc,
            "median_target_correlation": self.median_target_correlation,
            "identical_target_pairs": self.identical_target_pairs,
            "variance_ratios": self.variance_ratios,
        }


def load_fold_changes(path) -> dict[str, np.ndarray]:
    """The applied per-gene log2 fold changes the cells were built from."""
    if not path.is_file():
        return {}
    payload = np.load(path, allow_pickle=True)
    return {
        str(target): payload["log2_fold_change"][i]
        for i, target in enumerate(payload["targets"])
    }


def measure_context(
    cfg: Config,
    spec: SubmissionSpec,
    run_paths: RunPaths,
    context,
    axis_position: dict[str, int],
    audit: validator.MatrixAudit,
    pairs: Counter,
) -> tuple[ContextStats, list[dict[str, Any]]]:
    """One context: read every target's block once and measure it."""
    import scipy.sparse as sp

    log = get_logger()
    name = context.name
    control_counts = np.asarray(context.controls.counts())
    control_library = control_counts.sum(axis=1)
    control_lognorm = lognorm(control_counts)
    control_mean_cpm = mean_cp10k(control_counts)
    median_control_variance = float(np.median(control_lognorm.var(axis=0)))
    del control_lognorm
    # The first half of check 2 is judged in **count space**. A gene the
    # generator was told to leave alone comes back as the control cells'
    # own counts, so its mean deviates from the control pool's only by which
    # cells were drawn. In normalized space it would also shift with the
    # library size that the *other* genes' changes moved — a compositional
    # artifact, not a magnitude problem.
    control_mean_counts = control_counts.mean(axis=0)
    control_std_counts = control_counts.std(axis=0)

    stats = ContextStats(
        name=name, median_control_library=float(np.median(control_library))
    )
    applied = load_fold_changes(run_paths.fold_changes(name))
    knockdown: list[dict[str, Any]] = []
    changes, targets_seen = [], []
    predicted_libraries: list[float] = []

    unmoved_offenders: list[dict[str, Any]] = []
    moved_offenders: list[dict[str, Any]] = []
    n_unmoved = n_unmoved_bad = n_moved = n_moved_bad = 0

    for target in spec.targets:
        block = sp.load_npz(run_paths.prediction_block(name, target)).tocsr()
        audit.add(block)
        audit.count_over_cap(block, cfg.submission.max_counts_per_cell)
        pairs[(name, target)] += int(block.shape[0])

        dense = block.toarray()
        del block
        n_predicted = dense.shape[0]
        predicted_libraries.extend(dense.sum(axis=1).tolist())
        predicted_lognorm = lognorm(dense)
        predicted_mean_cpm = mean_cp10k(dense)
        predicted_mean_counts = dense.mean(axis=0)
        del dense

        # check 5 — the spread of the predicted cells, in log1p(CP10K).
        stats.variance_ratios[target] = float(
            np.median(predicted_lognorm.var(axis=0)) / max(median_control_variance, 1e-12)
        )
        del predicted_lognorm

        position = axis_position.get(target)
        floor = cfg.predict.cpm_floor
        change = np.log2(
            np.maximum(predicted_mean_cpm, floor) / np.maximum(control_mean_cpm, floor)
        )

        # check 3 — the knockdown is visible.
        if position is not None:
            control_level = float(control_mean_cpm[position])
            predicted_level = float(predicted_mean_cpm[position])
            knockdown.append(
                {
                    "context": name,
                    "target_gene": target,
                    "control_mean_cpm": control_level,
                    "predicted_mean_cpm": predicted_level,
                    "ratio": predicted_level / max(control_level, EPSILON),
                    "eligible": control_level >= cfg.sanity.knockdown_min_control_cpm,
                }
            )

        # check 2 — both halves, split by what the generator was allowed to move.
        applied_change = np.asarray(
            applied.get(target, np.zeros(spec.n_genes)), dtype=np.float64
        )
        moved_mask = np.abs(applied_change) > 0
        if position is not None:
            moved_mask[position] = False  # the target's own gene is excluded
        unmoved_mask = ~moved_mask
        if position is not None:
            unmoved_mask[position] = False

        tolerance = cfg.sanity.control_mean_sigmas * control_std_counts / np.sqrt(
            max(n_predicted, 1)
        )
        deviation = np.abs(predicted_mean_counts - control_mean_counts) - tolerance
        outside = np.flatnonzero(unmoved_mask & (deviation > 0))
        n_unmoved += int(unmoved_mask.sum())
        n_unmoved_bad += outside.size
        for gene in outside[: cfg.sanity.max_reported]:
            unmoved_offenders.append(
                {"context": name, "target_gene": target, "gene": spec.axis[gene],
                 "predicted_mean_counts": float(predicted_mean_counts[gene]),
                 "control_mean_counts": float(control_mean_counts[gene]),
                 "sampling_tolerance": float(tolerance[gene])}
            )

        # The bound is on the change the generator was **asked** to make, not
        # on one re-measured from the cells: at 0.02 CP10K a gene that simply
        # draws no count in 50 cells reads as a 14-log2 drop, which measures
        # the sparsity of the data rather than the size of the prediction.
        over = np.flatnonzero(
            moved_mask & (np.abs(applied_change) > cfg.sanity.max_abs_log2fc)
        )
        n_moved += int(moved_mask.sum())
        n_moved_bad += over.size
        for gene in over[: cfg.sanity.max_reported]:
            moved_offenders.append(
                {"context": name, "target_gene": target, "gene": spec.axis[gene],
                 "applied_log2_fold_change": float(applied_change[gene]),
                 "measured_log2_fold_change": float(change[gene])}
            )
        if moved_mask.any():
            stats.max_abs_log2fc = max(
                stats.max_abs_log2fc, float(np.abs(applied_change[moved_mask]).max())
            )

        changes.append(change.astype(np.float32))
        targets_seen.append(target)

    stats.median_predicted_library = float(np.median(predicted_libraries))
    stats.median_library_ratio = stats.median_predicted_library / max(
        stats.median_control_library, 1e-12
    )
    stats.unmoved_outside_control_range = n_unmoved_bad
    stats.unmoved_fraction_outside = n_unmoved_bad / max(n_unmoved, 1)
    stats.worst_unmoved = unmoved_offenders[: cfg.sanity.max_reported]
    stats.moved_over_max_abs_log2fc = n_moved_bad
    stats.moved_fraction_over = n_moved_bad / max(n_moved, 1)
    stats.worst_moved = moved_offenders[: cfg.sanity.max_reported]

    # Check 4 is judged with **every** target gene's column removed, not just
    # each row's own. Two predictions that agree everywhere except at the
    # genes being knocked down are the same profile, and leaving those columns
    # in would let a row differ from its twin purely by which gene it targets.
    # It is also what the discrimination metric does: it excludes every target
    # gene of the panel (§6).
    matrix = np.vstack(changes)
    keep = np.setdiff1d(
        np.arange(matrix.shape[1]),
        np.array([axis_position[t] for t in targets_seen if t in axis_position], dtype=int),
    )
    comparable = matrix[:, keep] if keep.size else matrix
    stats.median_target_correlation = check_mod.median_pairwise_correlation(comparable)
    stats.identical_target_pairs = identical_pairs(comparable, targets_seen)
    log.info(
        "  %s: library ratio %.2f, median target correlation %.3f",
        name, stats.median_library_ratio, stats.median_target_correlation,
    )
    return stats, knockdown


def identical_pairs(matrix: np.ndarray, targets: list[str]) -> list[list[str]]:
    """Targets whose predicted changes are bit-for-bit the same (§6.1 check 4)."""
    seen: dict[bytes, str] = {}
    duplicates = []
    for row, target in zip(matrix, targets):
        key = row.tobytes()
        if key in seen:
            duplicates.append([seen[key], target])
        else:
            seen[key] = target
    return duplicates


def run_sanity(cfg: Config) -> dict[str, Any]:
    """The `sanity` stage (checklist item 14)."""
    from ..phases import phase3
    from ..runtime import release_gpu_memory

    log = get_logger()
    run_paths = RunPaths(cfg)
    run_paths.ensure()
    run_paths.sanity_dir.mkdir(parents=True, exist_ok=True)
    paths = DataPaths(cfg)
    spec = load_spec(cfg)
    axis_position = {symbol: i for i, symbol in enumerate(spec.axis)}

    contexts = phase3.load_challenge_contexts(cfg, "cpu")
    audit = validator.MatrixAudit()
    pairs: Counter = Counter()
    per_context: dict[str, Any] = {}
    knockdown_records: list[dict[str, Any]] = []

    for name in spec.contexts:
        stats, records = measure_context(
            cfg, spec, run_paths, contexts[name], axis_position, audit, pairs
        )
        per_context[name] = stats.as_dict()
        knockdown_records.extend(records)
        release_gpu_memory()

    format_result = validator.result_from_audit(
        cfg, spec, audit, pairs, source="predictions/model"
    )
    rehearsal = (
        json.loads(run_paths.rehearsal_report.read_text())
        if run_paths.rehearsal_report.is_file()
        else {}
    )

    results: list[CheckResult] = [
        CheckResult(
            number=1,
            name="complete_and_valid",
            status=check_mod.verdict(format_result.ok, hard=True),
            message="every rule of §8 holds on the predicted blocks"
            if format_result.ok
            else f"{len(format_result.failures)} rule(s) of §8 are violated: "
                 f"{[r.rule for r in format_result.failures]}",
            value=len(format_result.failures),
            threshold=0,
            detail={"rules": [r.as_dict() for r in format_result.rules],
                    "totals": format_result.totals},
        ),
        check_mod.check_magnitudes(cfg, per_context),
        check_mod.check_knockdown(cfg, knockdown_records),
        check_mod.check_targets_differ(cfg, per_context),
        check_mod.check_cells_vary(cfg, per_context),
        check_mod.check_beats_no_change(cfg, rehearsal),
        check_mod.check_de_call_count(cfg, rehearsal),
    ]

    on_mini = "mini" in str(cfg.data_root)
    report = check_mod.report_dict(cfg, results, on_mini)
    report["contexts"] = per_context
    report["knockdown"] = knockdown_records
    run_paths.sanity_report.write_text(json.dumps(report, indent=2, default=str))
    summary = check_mod.summarize(results, on_mini)
    run_paths.sanity_summary.write_text(summary)
    log.info("\n%s", summary)

    if not report["ok"]:
        # §6.1: a failure of check 1 stops the pipeline before the submission
        # is written. Nothing else here stops it.
        raise RuntimeError(
            "sanity check 1 failed — the predictions violate §8, so no submission "
            f"was written. See {run_paths.sanity_report}"
        )
    return report
