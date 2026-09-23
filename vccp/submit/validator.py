"""The submission's own validator (CLAUDE.md §8).

Everything §8 requires, checked before `vcc prep` ever sees the file — and
checked on the *prediction blocks* too, because §6.1's first sanity check is
"every rule of section 8" applied before the submission is written. Both
callers run the same rules over the same accumulator, so the two can never
drift apart.

The rules, in §8's own order:

1. `var` index equals `gene_names.csv`, in that exact order;
2. `obs` carries exactly `target_gene` and `context`;
3. the set of (context, target) pairs equals every context × every
   perturbation of `pert_counts.csv` — no control rows, no label outside the
   list, no construct id such as `ADNP-1`, and context labels spelled as the
   control files spell them;
4. each pair has exactly `cells_per_pert` rows;
5. values are finite, non-negative whole numbers;
6. no cell totals more than `submission.max_counts_per_cell`;
7. stored entries stay under `submission.max_stored_entries`, with no
   explicitly stored zeros;
8. the matrix is stored sparse.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .spec import SubmissionSpec

#: `ADNP-1` — a construct id where §8 requires the gene symbol.
CONSTRUCT_ID = re.compile(r"^(?P<symbol>.+)-(?P<construct>\d+)$")


@dataclass
class Rule:
    """One rule of §8, and whether the thing under test obeys it."""

    rule: str
    ok: bool
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "ok": self.ok, "message": self.message, **self.detail}


@dataclass
class ValidationResult:
    rules: list[Rule]
    source: str = ""
    spec: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(rule.ok for rule in self.rules)

    @property
    def failures(self) -> list[Rule]:
        return [rule for rule in self.rules if not rule.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source": self.source,
            "requires": self.spec,
            "totals": self.totals,
            "n_failed": len(self.failures),
            "rules": [rule.as_dict() for rule in self.rules],
        }

    def summary(self) -> str:
        lines = [f"submission validation: {'PASS' if self.ok else 'FAIL'} ({self.source})"]
        for rule in self.rules:
            lines.append(f"  [{'ok  ' if rule.ok else 'FAIL'}] {rule.rule}: {rule.message}")
        return "\n".join(lines)


@dataclass
class MatrixAudit:
    """Everything the value rules need, accumulated block by block.

    Kept as running totals so that a file of 360,000 × 18,533 is audited in
    chunks and never materialized (§8).
    """

    n_rows: int = 0
    stored_entries: int = 0
    explicit_zeros: int = 0
    n_negative: int = 0
    n_non_integer: int = 0
    n_non_finite: int = 0
    n_dense_blocks: int = 0
    n_over_cap: int = 0
    max_cell_total: float = 0.0
    largest_cells: list[float] = field(default_factory=list)

    def add(self, block) -> None:
        import scipy.sparse as sp

        if not sp.issparse(block):
            self.n_dense_blocks += 1
            block = sp.csr_matrix(np.asarray(block))

        data = block.data
        self.n_rows += block.shape[0]
        self.stored_entries += int(data.size)
        self.explicit_zeros += int((data == 0).sum())
        self.n_non_finite += int((~np.isfinite(data)).sum())
        finite = data[np.isfinite(data)]
        self.n_negative += int((finite < 0).sum())
        self.n_non_integer += int((finite != np.floor(finite)).sum())

        totals = np.asarray(block.sum(axis=1)).ravel()
        if totals.size:
            self.max_cell_total = max(self.max_cell_total, float(totals.max()))
            self.largest_cells = sorted(
                self.largest_cells + totals.tolist(), reverse=True
            )[:5]

    def count_over_cap(self, block, cap: int) -> None:
        totals = np.asarray(block.sum(axis=1)).ravel()
        self.n_over_cap += int((totals > cap).sum())


def looks_like_construct_id(label: str, targets: set[str]) -> str | None:
    """`ADNP-1` where `ADNP` is a listed target — §8's own example."""
    match = CONSTRUCT_ID.match(label)
    if match and match.group("symbol") in targets:
        return match.group("symbol")
    return None


def check_gene_axis(axis: Iterable[str], spec: SubmissionSpec) -> Rule:
    axis = list(axis)
    if axis == list(spec.axis):
        return Rule("gene_axis", True, f"{len(axis)} genes in gene_names.csv order")

    if len(axis) != spec.n_genes:
        return Rule(
            "gene_axis", False,
            f"{len(axis)} genes, but gene_names.csv lists {spec.n_genes}",
            {"n_genes": len(axis), "expected": spec.n_genes},
        )
    if set(axis) != set(spec.axis):
        missing = sorted(set(spec.axis) - set(axis))[:5]
        extra = sorted(set(axis) - set(spec.axis))[:5]
        return Rule(
            "gene_axis", False, "the gene set differs from gene_names.csv",
            {"missing_example": missing, "unexpected_example": extra},
        )
    first = next(i for i, (a, b) in enumerate(zip(axis, spec.axis)) if a != b)
    return Rule(
        "gene_axis", False,
        f"the genes are gene_names.csv's but not in its order (first differs at {first})",
        {"position": first, "found": axis[first], "expected": spec.axis[first]},
    )


def check_labels(pairs: Counter, spec: SubmissionSpec, max_reported: int = 10) -> list[Rule]:
    """Rules 3 and 4: which (context, target) pairs there are, and how many
    cells each has."""
    rules: list[Rule] = []
    targets = set(spec.targets)
    contexts = set(spec.contexts)

    seen_targets = {target for _, target in pairs}
    seen_contexts = {context for context, _ in pairs}

    controls = sorted(t for t in seen_targets if t == spec.control_label)
    rules.append(
        Rule(
            "no_control_rows", not controls,
            "no control rows" if not controls
            else f"{spec.control_label!r} rows are in the prediction; §8 forbids them",
            {"control_label": spec.control_label},
        )
    )

    constructs = {
        target: symbol
        for target in sorted(seen_targets - targets)
        if (symbol := looks_like_construct_id(target, targets))
    }
    rules.append(
        Rule(
            "gene_symbols_not_construct_ids", not constructs,
            "every label is a gene symbol" if not constructs
            else f"construct id(s) where §8 requires the gene symbol: "
                 f"{sorted(constructs)[:max_reported]}",
            {"examples": dict(list(constructs.items())[:max_reported])},
        )
    )

    unknown = sorted(seen_targets - targets - set(constructs) - {spec.control_label})
    rules.append(
        Rule(
            "labels_within_pert_counts", not unknown,
            "every perturbation is one of pert_counts.csv's" if not unknown
            else f"{len(unknown)} label(s) are not in pert_counts.csv: {unknown[:max_reported]}",
            {"examples": unknown[:max_reported]},
        )
    )

    bad_contexts = sorted(seen_contexts - contexts)
    rules.append(
        Rule(
            "context_labels", not bad_contexts,
            "context labels are the control files' own" if not bad_contexts
            else f"context label(s) not in the release: {bad_contexts[:max_reported]}",
            {"expected": list(spec.contexts), "examples": bad_contexts[:max_reported]},
        )
    )

    missing = sorted(spec.required_pairs - set(pairs))
    rules.append(
        Rule(
            "every_context_times_every_perturbation", not missing,
            f"all {len(spec.required_pairs)} (context, perturbation) pairs are present"
            if not missing
            else f"{len(missing)} pair(s) missing, e.g. {missing[:max_reported]}",
            {"n_missing": len(missing), "examples": missing[:max_reported]},
        )
    )

    wrong = {
        f"{context}/{target}": {"found": count, "expected": spec.cells_for(target)}
        for (context, target), count in sorted(pairs.items())
        if target in targets and count != spec.cells_for(target)
    }
    rules.append(
        Rule(
            "cells_per_perturbation", not wrong,
            "every perturbation has exactly its cell count" if not wrong
            else f"{len(wrong)} pair(s) have the wrong number of cells, e.g. "
                 f"{dict(list(wrong.items())[:3])}",
            {"n_wrong": len(wrong), "examples": dict(list(wrong.items())[:max_reported])},
        )
    )
    return rules


def check_values(audit: MatrixAudit, cfg, spec: SubmissionSpec) -> list[Rule]:
    """Rules 5 to 8: the matrix itself."""
    limits = cfg.submission
    return [
        Rule(
            "finite_non_negative_integers",
            audit.n_non_finite == 0 and audit.n_negative == 0 and audit.n_non_integer == 0,
            "every value is a finite non-negative whole number"
            if audit.n_non_finite == audit.n_negative == audit.n_non_integer == 0
            else f"{audit.n_non_finite} non-finite, {audit.n_negative} negative and "
                 f"{audit.n_non_integer} non-integer value(s)",
            {
                "n_non_finite": audit.n_non_finite,
                "n_negative": audit.n_negative,
                "n_non_integer": audit.n_non_integer,
            },
        ),
        Rule(
            "counts_per_cell",
            audit.n_over_cap == 0,
            f"the largest cell totals {audit.max_cell_total:,.0f} counts "
            f"(cap {limits.max_counts_per_cell:,})"
            if audit.n_over_cap == 0
            else f"{audit.n_over_cap} cell(s) exceed {limits.max_counts_per_cell:,} counts",
            {
                "max_cell_total": audit.max_cell_total,
                "cap": limits.max_counts_per_cell,
                "n_over_cap": audit.n_over_cap,
            },
        ),
        Rule(
            "stored_entries",
            audit.stored_entries <= limits.max_stored_entries,
            f"{audit.stored_entries:,} stored entries (cap {limits.max_stored_entries:,})",
            {"stored_entries": audit.stored_entries, "cap": limits.max_stored_entries},
        ),
        Rule(
            "no_explicit_zeros",
            audit.explicit_zeros == 0,
            "no explicitly stored zeros" if audit.explicit_zeros == 0
            else f"{audit.explicit_zeros:,} explicit zero(s) — they count toward the cap",
            {"explicit_zeros": audit.explicit_zeros},
        ),
        Rule(
            "sparse_storage",
            audit.n_dense_blocks == 0,
            "stored sparse" if audit.n_dense_blocks == 0
            else f"{audit.n_dense_blocks} block(s) are dense arrays",
            {"n_dense_blocks": audit.n_dense_blocks},
        ),
        Rule(
            "row_count",
            audit.n_rows == spec.total_cells,
            f"{audit.n_rows:,} rows" if audit.n_rows == spec.total_cells
            else f"{audit.n_rows:,} rows, but the round requires {spec.total_cells:,}",
            {"n_rows": audit.n_rows, "expected": spec.total_cells},
        ),
    ]


def audit_blocks(
    blocks: Iterable[tuple[str, str, Any]], cfg, spec: SubmissionSpec
) -> tuple[MatrixAudit, Counter]:
    """Run every block past the accumulator, keeping none of them."""
    audit = MatrixAudit()
    pairs: Counter = Counter()
    for context, target, block in blocks:
        audit.add(block)
        audit.count_over_cap(block, cfg.submission.max_counts_per_cell)
        pairs[(str(context), str(target))] += int(block.shape[0])
    return audit, pairs


def result_from_audit(
    cfg, spec: SubmissionSpec, audit: "MatrixAudit", pairs: Counter, *, source: str,
    axis: Iterable[str] | None = None,
) -> ValidationResult:
    """The rules, from an audit a caller accumulated itself.

    The sanity stage reads every block once for its own statistics; this lets
    it run §8's rules from that same pass instead of reading them again.
    """
    rules = [check_gene_axis(spec.axis if axis is None else axis, spec)]
    rules += check_labels(pairs, spec, cfg.sanity.max_reported)
    rules += check_values(audit, cfg, spec)
    return ValidationResult(
        rules=rules, source=source, spec=spec.describe(), totals=_totals(audit, pairs)
    )


def validate_blocks(
    cfg, spec: SubmissionSpec, blocks: Iterable[tuple[str, str, Any]], *, source: str,
    axis: Iterable[str] | None = None,
) -> ValidationResult:
    """§8's rules applied to the per-target prediction blocks.

    This is sanity check 1 (§6.1): it runs *before* the submission is
    written, so a violation stops the pipeline rather than producing a file
    that cannot be submitted.
    """
    audit, pairs = audit_blocks(blocks, cfg, spec)
    return result_from_audit(cfg, spec, audit, pairs, source=source, axis=axis)


def _totals(audit: MatrixAudit, pairs: Counter) -> dict[str, Any]:
    return {
        "n_rows": audit.n_rows,
        "n_pairs": len(pairs),
        "stored_entries": audit.stored_entries,
        "max_cell_total": audit.max_cell_total,
        "largest_cell_totals": audit.largest_cells,
    }


def iter_h5ad_blocks(path: Path, spec: SubmissionSpec, chunk_cells: int) -> Iterator[tuple]:
    """Stream a written submission back, one chunk of rows at a time."""
    import anndata as ad

    backed = ad.read_h5ad(path, backed="r")
    try:
        obs = backed.obs
        contexts = obs[spec.context_col].astype(str).to_numpy()
        targets = obs[spec.pert_col].astype(str).to_numpy()
        for start in range(0, backed.n_obs, chunk_cells):
            stop = min(start + chunk_cells, backed.n_obs)
            block = backed.X[start:stop]
            # One chunk can straddle several (context, target) groups, so it
            # is split on the labels rather than assumed homogeneous.
            labels = list(zip(contexts[start:stop], targets[start:stop]))
            for (context, target) in sorted(set(labels)):
                rows = np.flatnonzero(
                    (contexts[start:stop] == context) & (targets[start:stop] == target)
                )
                yield context, target, block[rows]
    finally:
        if backed.file is not None:
            backed.file.close()


def validate_file(cfg, spec: SubmissionSpec, path: Path) -> ValidationResult:
    """§8's rules applied to the submission file itself."""
    import anndata as ad
    import scipy.sparse as sp

    backed = ad.read_h5ad(path, backed="r")
    try:
        axis = list(backed.var_names)
        obs_columns = list(backed.obs.columns)
        is_sparse = sp.issparse(backed.X[:1]) if backed.n_obs else True
    finally:
        if backed.file is not None:
            backed.file.close()

    rules = [check_gene_axis(axis, spec)]
    extra = [c for c in obs_columns if c not in (spec.pert_col, spec.context_col)]
    missing = [c for c in (spec.pert_col, spec.context_col) if c not in obs_columns]
    rules.append(
        Rule(
            "obs_columns", not missing and not extra,
            f"obs carries exactly {spec.pert_col!r} and {spec.context_col!r}"
            if not missing and not extra
            else f"obs columns are {obs_columns}; §8 asks for exactly "
                 f"[{spec.pert_col!r}, {spec.context_col!r}]",
            {"found": obs_columns, "missing": missing, "unexpected": extra},
        )
    )

    audit, pairs = audit_blocks(
        iter_h5ad_blocks(path, spec, cfg.submission.write_chunk_cells), cfg, spec
    )
    if not is_sparse:
        audit.n_dense_blocks += 1
    rules += check_labels(pairs, spec, cfg.sanity.max_reported)
    rules += check_values(audit, cfg, spec)
    return ValidationResult(
        rules=rules,
        source=str(path),
        spec=spec.describe(),
        totals={**_totals(audit, pairs), "file_size_bytes": path.stat().st_size},
    )
