"""The seven sanity checks of CLAUDE.md §6.1.

Each reports **pass / warn / fail** with the measured value and the threshold
it was measured against, and every threshold comes from `sanity:` in the
config. Check 1 — "every rule of §8" — runs on the prediction blocks *before*
the submission is written, so a violation stops the pipeline rather than
producing a file that cannot be submitted. The others do not stop the run;
they are printed prominently and make `validate` refuse the submission unless
it is run with `--allow-warnings`.

Checks 6 and 7 are about the rehearsal, where the truth is known: the
official metrics must beat their no-change values, and the number of genes
the prediction calls differentially expressed must be neither near zero (too
timid for the DE metrics to score) nor far above the reference (flooding
them).

On `mini_data` these statistics are noisy and warnings are expected; the
checks still run and still report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Config
from ..logging_utils import get_logger

PASS, WARN, FAIL = "pass", "warn", "fail"

#: log1p(CP10K), the units checks 2 and 5 compare in.
TARGET_SUM = 1e4


@dataclass
class CheckResult:
    """One check of §6.1."""

    number: int
    name: str
    status: str
    message: str
    value: Any = None
    threshold: Any = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == FAIL

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.number,
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            **self.detail,
        }


def verdict(ok: bool, *, hard: bool) -> str:
    """A failure is a `fail` only where §6.1 says the run must stop."""
    if ok:
        return PASS
    return FAIL if hard else WARN


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #
def cp10k(counts: np.ndarray) -> np.ndarray:
    """Counts per 10,000, per cell."""
    library = np.maximum(np.asarray(counts).sum(axis=1, keepdims=True), 1.0)
    return np.asarray(counts) / library * TARGET_SUM


def lognorm(counts: np.ndarray) -> np.ndarray:
    """log1p(CP10K) — the normalization of §3.2."""
    return np.log1p(cp10k(counts))


def mean_cp10k(counts: np.ndarray) -> np.ndarray:
    """Per-gene mean CP10K, without materializing a float copy of `counts`."""
    counts = np.asarray(counts)
    library = np.maximum(counts.sum(axis=1), 1).astype(np.float64)
    return counts.T.astype(np.float64) @ (1.0 / library) / counts.shape[0] * TARGET_SUM


# --------------------------------------------------------------------------- #
# check 1 — complete and valid
# --------------------------------------------------------------------------- #
def check_complete_and_valid(cfg, spec, blocks, axis) -> tuple[CheckResult, Any]:
    """§8's every rule, on the blocks, before anything is written."""
    from ..submit import validator

    result = validator.validate_blocks(
        cfg, spec, blocks, source="predictions/model", axis=axis
    )
    failures = [rule.rule for rule in result.failures]
    return (
        CheckResult(
            number=1,
            name="complete_and_valid",
            status=verdict(result.ok, hard=True),
            message="every rule of §8 holds on the predicted blocks"
            if result.ok
            else f"{len(failures)} rule(s) of §8 are violated: {failures}",
            value=len(failures),
            threshold=0,
            detail={"rules": [rule.as_dict() for rule in result.rules],
                    "totals": result.totals},
        ),
        result,
    )


# --------------------------------------------------------------------------- #
# check 2 — plausible magnitudes
# --------------------------------------------------------------------------- #
def check_magnitudes(cfg, per_context: dict[str, Any]) -> CheckResult:
    """Library sizes in range, and both halves of the per-gene bound.

    The second half is the M2-review addition (PLAN.md §8.5): genes below the
    confidence threshold are copies of control cells by construction, so a
    check restricted to them leaves every gene the model *did* move unbounded.
    """
    sanity = cfg.sanity
    contexts: dict[str, Any] = {}
    ok = True

    for name, stats in per_context.items():
        ratio = stats["median_library_ratio"]
        in_range = sanity.library_ratio_low <= ratio <= sanity.library_ratio_high
        unmoved_bad = stats["unmoved_outside_control_range"]
        moved_bad = stats["moved_over_max_abs_log2fc"]
        context_ok = in_range and unmoved_bad == 0 and moved_bad == 0
        ok &= context_ok
        contexts[name] = {
            "median_library_ratio": ratio,
            "median_predicted_library": stats["median_predicted_library"],
            "median_control_library": stats["median_control_library"],
            "library_in_range": in_range,
            "unmoved_genes_outside_control_range": unmoved_bad,
            "unmoved_fraction": stats["unmoved_fraction_outside"],
            "worst_unmoved": stats["worst_unmoved"],
            "moved_genes_over_max_abs_log2fc": moved_bad,
            "moved_fraction_over": stats["moved_fraction_over"],
            "worst_moved": stats["worst_moved"],
            "max_abs_log2fc": stats["max_abs_log2fc"],
        }

    worst = max((c["median_library_ratio"] for c in contexts.values()), default=float("nan"))
    return CheckResult(
        number=2,
        name="plausible_magnitudes",
        status=verdict(ok, hard=False),
        message="library sizes and per-gene means are plausible in every context"
        if ok
        else "at least one context has an implausible library size or per-gene mean",
        value={name: c["median_library_ratio"] for name, c in contexts.items()},
        threshold={
            "library_ratio": [sanity.library_ratio_low, sanity.library_ratio_high],
            "control_mean_sigmas": sanity.control_mean_sigmas,
            "max_abs_log2fc": sanity.max_abs_log2fc,
        },
        detail={"contexts": contexts, "worst_library_ratio": worst},
    )


# --------------------------------------------------------------------------- #
# check 3 — the knockdown is visible
# --------------------------------------------------------------------------- #
def check_knockdown(cfg, records: list[dict[str, Any]]) -> CheckResult:
    """Each target's own gene must actually go down where it is expressed."""
    sanity = cfg.sanity
    eligible = [r for r in records if r["eligible"]]
    if not eligible:
        return CheckResult(
            number=3, name="knockdown_visible", status=WARN,
            message="no target gene is detectably expressed in its context's controls, "
            f"so nothing could be checked (floor {sanity.knockdown_min_control_cpm} CP10K)",
            value=0, threshold=sanity.knockdown_min_fraction,
            detail={"n_eligible": 0, "n_targets": len(records)},
        )

    good = [r for r in eligible if r["ratio"] <= sanity.knockdown_max_ratio]
    fraction = len(good) / len(eligible)
    failing = sorted(
        (r for r in eligible if r["ratio"] > sanity.knockdown_max_ratio),
        key=lambda r: -r["ratio"],
    )
    return CheckResult(
        number=3,
        name="knockdown_visible",
        status=verdict(fraction >= sanity.knockdown_min_fraction, hard=False),
        message=f"{len(good)}/{len(eligible)} expressed target genes are at most "
        f"{sanity.knockdown_max_ratio:g}x their control level",
        value=round(fraction, 4),
        threshold=sanity.knockdown_min_fraction,
        detail={
            "n_eligible": len(eligible),
            "n_targets": len(records),
            "max_ratio": sanity.knockdown_max_ratio,
            "failing": [
                {k: r[k] for k in ("context", "target_gene", "ratio", "control_mean_cpm")}
                for r in failing[: sanity.max_reported]
            ],
        },
    )


# --------------------------------------------------------------------------- #
# check 4 — targets differ
# --------------------------------------------------------------------------- #
def median_pairwise_correlation(changes: np.ndarray) -> float:
    """Median off-diagonal Pearson correlation between targets' changes."""
    if changes.shape[0] < 2:
        return float("nan")
    centered = changes - changes.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    norms[norms == 0] = 1.0
    matrix = (centered / norms[:, None]) @ (centered / norms[:, None]).T
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    return float(np.median(upper)) if upper.size else float("nan")


def check_targets_differ(cfg, per_context: dict[str, Any]) -> CheckResult:
    sanity = cfg.sanity
    contexts = {}
    ok = True
    for name, stats in per_context.items():
        correlation = stats["median_target_correlation"]
        duplicates = stats["identical_target_pairs"]
        context_ok = (
            (np.isnan(correlation) or correlation < sanity.max_median_target_correlation)
            and not duplicates
        )
        ok &= context_ok
        contexts[name] = {
            "median_pairwise_correlation": correlation,
            "identical_pairs": duplicates[: sanity.max_reported],
            "n_identical_pairs": len(duplicates),
        }
    return CheckResult(
        number=4,
        name="targets_differ",
        status=verdict(ok, hard=False),
        message="predicted changes differ between targets"
        if ok
        else "targets are too alike, or two of them are identical",
        value={n: c["median_pairwise_correlation"] for n, c in contexts.items()},
        threshold=sanity.max_median_target_correlation,
        detail={"contexts": contexts,
                "note": "every target gene's column is excluded from every change "
                        "vector, as the discrimination metric does (§6)"},
    )


# --------------------------------------------------------------------------- #
# check 5 — cells vary
# --------------------------------------------------------------------------- #
def check_cells_vary(cfg, per_context: dict[str, Any]) -> CheckResult:
    sanity = cfg.sanity
    contexts = {}
    ok = True
    for name, stats in per_context.items():
        ratios = stats["variance_ratios"]
        values = np.array(list(ratios.values()), dtype=float)
        in_range = (values >= sanity.variance_ratio_low) & (values <= sanity.variance_ratio_high)
        context_ok = bool(in_range.all()) and values.size > 0
        ok &= context_ok
        worst = sorted(ratios.items(), key=lambda kv: abs(np.log2(max(kv[1], 1e-12))))[::-1]
        contexts[name] = {
            "median_variance_ratio": float(np.median(values)) if values.size else float("nan"),
            "n_targets_out_of_range": int((~in_range).sum()),
            "worst": dict(worst[: sanity.max_reported]),
        }
    return CheckResult(
        number=5,
        name="cells_vary",
        status=verdict(ok, hard=False),
        message="predicted cells vary like the controls do"
        if ok
        else "some targets' cells are too alike or too scattered",
        value={n: c["median_variance_ratio"] for n, c in contexts.items()},
        threshold=[sanity.variance_ratio_low, sanity.variance_ratio_high],
        detail={"contexts": contexts,
                "note": "median per-gene variance across predicted cells, in log1p(CP10K), "
                        "against the same statistic on that context's control cells"},
    )


# --------------------------------------------------------------------------- #
# checks 6 and 7 — the rehearsal, where the truth is known
# --------------------------------------------------------------------------- #
#: The three §6.1 check 6 names, with the direction each must beat.
BETTER_THAN_NOTHING = {
    "pds_cosine": ("greater", 0.5),
    "expr_mse_unbiased_capped_norm": ("less", 1.0),
    "de_wilcoxon_lfc_nmae": ("less", 1.0),
}


def scored_method_arm(report: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The end-to-end method arm the two rehearsal checks read.

    Variant 2 is the only one scored end to end (§4.6); a panel-assisted arm
    would answer a different question, so it is never substituted.
    """
    from ..rehearsal import common

    best, label = None, "none"
    for entry in report.get("variants", []):
        if entry["variant"] != "cross_context":
            continue
        arm = entry["arms"].get(common.METHOD, {}).get(common.END_TO_END, {})
        if "scored" in arm and arm["scored"]:
            if best is None or entry.get("n_targets", 0) > best[1]:
                best = (arm, entry.get("n_targets", 0))
                label = f"{entry['variant']}/{entry['context']}"
    return (best[0] if best else None), label


def check_beats_no_change(cfg, report: dict[str, Any]) -> CheckResult:
    arm, label = scored_method_arm(report)
    if arm is None:
        return CheckResult(
            number=6, name="better_than_doing_nothing", status=WARN,
            message="the rehearsal produced no end-to-end scored arm, so there is "
            "nothing to compare against the no-change values",
            value=None, threshold=BETTER_THAN_NOTHING, detail={"source": label},
        )

    scored = arm["scored"]
    outcomes = {}
    ok = True
    for metric, (direction, point) in BETTER_THAN_NOTHING.items():
        value = scored.get(metric)
        if value is None:
            outcomes[metric] = {"value": None, "beats": None, "no_change": point}
            continue
        beats = value > point if direction == "greater" else value < point
        ok &= beats
        outcomes[metric] = {"value": value, "beats": bool(beats), "no_change": point,
                            "direction": direction}
    measured = [m for m, o in outcomes.items() if o["beats"] is not None]
    return CheckResult(
        number=6,
        name="better_than_doing_nothing",
        status=verdict(ok and bool(measured), hard=False),
        message=f"on {label}: "
        + ", ".join(
            f"{m} {o['value']:.3f} vs {o['no_change']}" for m, o in outcomes.items()
            if o["value"] is not None
        ),
        value={m: o["value"] for m, o in outcomes.items()},
        threshold={m: f"{d} than {p}" for m, (d, p) in BETTER_THAN_NOTHING.items()},
        detail={"source": label, "metrics": outcomes,
                "objective_vs_no_change": arm.get("normalized", {}).get("objective")},
    )


def check_de_call_count(cfg, report: dict[str, Any]) -> CheckResult:
    """The median over targets of predicted / real significant genes."""
    sanity = cfg.sanity
    arm, label = scored_method_arm(report)
    counts = (arm or {}).get("nsig_counts", {})
    ratios = {
        target: value["pred"] / value["real"]
        for target, value in counts.items()
        if value.get("real")
    }
    if not ratios:
        return CheckResult(
            number=7, name="reasonable_de_calls", status=WARN,
            message="the rehearsal reported no per-target nsig counts, so the ratio "
            "could not be taken",
            value=None,
            threshold=[sanity.nsig_ratio_low, sanity.nsig_ratio_high],
            detail={"source": label, "n_targets": len(counts)},
        )

    median = float(np.median(list(ratios.values())))
    ok = sanity.nsig_ratio_low <= median <= sanity.nsig_ratio_high
    return CheckResult(
        number=7,
        name="reasonable_de_calls",
        status=verdict(ok, hard=False),
        message=f"on {label}: the median target calls {median:.2f}x as many "
        "significant genes as the reference does",
        value=round(median, 4),
        threshold=[sanity.nsig_ratio_low, sanity.nsig_ratio_high],
        detail={
            "source": label,
            "n_targets": len(ratios),
            "mean_real_nsig": float(np.mean([v["real"] for v in counts.values()])),
            "mean_pred_nsig": float(np.mean([v["pred"] for v in counts.values()])),
            "note": "near zero means the prediction is too timid for the DE metrics to "
                    "score it; far above means it floods them",
        },
    )


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def summarize(results: list[CheckResult], on_mini: bool) -> str:
    lines = ["Sanity checks (CLAUDE.md §6.1)", ""]
    for result in results:
        lines.append(f"  [{result.status.upper():4}] {result.number}. {result.name}")
        lines.append(f"         {result.message}")
    failed = [r for r in results if r.status == FAIL]
    warned = [r for r in results if r.status == WARN]
    lines.append("")
    lines.append(
        f"{len(results) - len(failed) - len(warned)} passed, {len(warned)} warned, "
        f"{len(failed)} failed"
    )
    if on_mini:
        lines.append(
            "These statistics are noisy on mini_data, where warnings are expected; "
            "the numbers are not indicative of real performance (CLAUDE.md §5)."
        )
    return "\n".join(lines)


def report_dict(cfg, results: list[CheckResult], on_mini: bool) -> dict[str, Any]:
    return {
        "checks": [result.as_dict() for result in results],
        "n_pass": sum(1 for r in results if r.status == PASS),
        "n_warn": sum(1 for r in results if r.status == WARN),
        "n_fail": sum(1 for r in results if r.status == FAIL),
        "ok": not any(r.status == FAIL for r in results),
        "clean": all(r.status == PASS for r in results),
        "thresholds": {
            field: getattr(cfg.sanity, field)
            for field in ("library_ratio_low", "library_ratio_high", "control_mean_sigmas",
                          "max_abs_log2fc", "knockdown_min_control_cpm",
                          "knockdown_max_ratio", "knockdown_min_fraction",
                          "max_median_target_correlation", "variance_ratio_low",
                          "variance_ratio_high", "nsig_ratio_low", "nsig_ratio_high")
        },
        "on_mini_data": on_mini,
    }
