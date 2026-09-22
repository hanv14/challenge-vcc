"""`check-data` — verify the inputs before anything trains.

The full data cannot be tested from the cloud, so this stage is what tells
the user, on the server and in seconds, whether the pipeline will run: every
file of CLAUDE.md §3 present, every expected column there, every gene order
agreeing, and the sizes it found printed for the user to confirm.

Every check reports `ok`, `warn` or `fail` and names the file it read. Checks
are all attempted — one missing file does not hide the next problem — and the
stage fails at the end if anything failed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from ..paths import DataPaths, RunPaths
from . import challenge, coverage, gene_split, lincs, manifest as manifest_mod, reference, replogle

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "detail": self.detail,
        }


class DataCheckError(RuntimeError):
    """`check-data` found something that would break a run."""


class Checker:
    """Collects results so every problem is reported, not just the first."""

    def __init__(self) -> None:
        self.results: list[CheckResult] = []
        self.sizes: dict[str, Any] = {}
        #: Challenge genes any training source actually measures. The
        #: remainder are seen only in the challenge controls, and their
        #: perturbation responses are never supervised anywhere (§3.1).
        self.measured_symbols: set[str] = set()

    def record(self, name: str, status: str, message: str, **detail: Any) -> CheckResult:
        result = CheckResult(name, status, message, detail)
        self.results.append(result)
        return result

    def ok(self, name: str, message: str, **detail: Any) -> CheckResult:
        return self.record(name, OK, message, **detail)

    def warn(self, name: str, message: str, **detail: Any) -> CheckResult:
        return self.record(name, WARN, message, **detail)

    def fail(self, name: str, message: str, **detail: Any) -> CheckResult:
        return self.record(name, FAIL, message, **detail)

    def run(self, name: str, path: Path, fn):
        """Run a check that reads `path`, turning a missing file or a bad
        column into a `fail` that names the file instead of a traceback."""
        if not path.exists():
            self.fail(name, f"missing file: {path}", path=str(path))
            return None
        try:
            return fn()
        except (ValueError, KeyError, OSError) as exc:
            self.fail(name, f"{path}: {exc}", path=str(path))
            return None

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def warned(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == WARN]


def _compare_orders(name: str, left: list[str], right: list[str], left_name: str, right_name: str,
                    checker: Checker) -> bool:
    """Two gene lists must be equal, element for element."""
    if len(left) != len(right):
        checker.fail(
            name,
            f"{left_name} has {len(left):,} genes but {right_name} has {len(right):,}",
            n_left=len(left),
            n_right=len(right),
        )
        return False
    mismatched = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    if mismatched:
        first = mismatched[0]
        checker.fail(
            name,
            f"{left_name} and {right_name} disagree at {len(mismatched):,} position(s); "
            f"first at index {first}: {left[first]!r} vs {right[first]!r}",
            n_mismatched=len(mismatched),
            first_index=first,
        )
        return False
    checker.ok(name, f"{left_name} and {right_name} agree on {len(left):,} genes in order")
    return True


def check_data(cfg: Config) -> dict[str, Any]:
    """Run every check and return the report. Raises `DataCheckError` on failure."""
    log = get_logger()
    paths = DataPaths(cfg)
    checker = Checker()

    log.info("Checking inputs under %s", cfg.data_root)
    log.info("  challenge release: %s", cfg.vcc_root)

    # ---- reference and release gene axes ---------------------------------
    gene_ref = checker.run(
        "challenge_genes", paths.challenge_genes,
        lambda: reference.load_gene_reference(paths.challenge_genes),
    )
    gene_names = checker.run(
        "gene_names", paths.gene_names,
        lambda: reference.load_gene_names(paths.gene_names),
    )
    if gene_ref is not None and gene_names is not None:
        _compare_orders(
            "gene_axis", list(gene_ref.symbols), gene_names,
            "ref/challenge_genes.tsv", "gene_names.csv", checker,
        )
    axis = gene_names if gene_names is not None else (list(gene_ref.symbols) if gene_ref else None)
    n_genes = len(axis) if axis is not None else 0
    checker.sizes["n_challenge_genes"] = n_genes

    checker.run("hgnc", cfg.ref_dir, lambda: _check_hgnc(paths, checker))

    # ---- manifest --------------------------------------------------------
    man = checker.run(
        "manifest", paths.manifest, lambda: manifest_mod.load_manifest(paths.manifest)
    )
    if man is not None:
        checker.sizes["cells_per_pert"] = man.cells_per_pert
        checker.sizes["contexts"] = list(man.contexts)
        if axis is not None and man.n_genes != n_genes:
            checker.fail(
                "manifest_n_genes",
                f"manifest says n_genes={man.n_genes:,} but gene_names.csv has {n_genes:,}",
                manifest_n_genes=man.n_genes, gene_names_n_genes=n_genes,
            )
        else:
            checker.ok("manifest_n_genes", f"manifest n_genes = {man.n_genes:,}")

    # ---- perturbations ---------------------------------------------------
    pert_col = man.pert_col if man is not None else "target_gene"
    perts = checker.run(
        "challenge_perts", paths.challenge_perts,
        lambda: reference.load_challenge_perts(paths.challenge_perts),
    )
    pert_counts = checker.run(
        "pert_counts", paths.pert_counts,
        lambda: reference.load_pert_counts(paths.pert_counts, pert_col),
    )
    if perts is not None and pert_counts is not None:
        _check_pert_lists(perts, pert_counts, checker)
    if pert_counts is not None and man is not None:
        _check_cells_per_pert(pert_counts, man, paths, checker)
    if pert_counts is not None:
        checker.sizes["n_targets"] = len(pert_counts)

    # ---- gene split ------------------------------------------------------
    split = checker.run(
        "gene_split", paths.gene_split, lambda: gene_split.load_gene_split(paths.gene_split)
    )
    if split is not None:
        checker.sizes["n_panel"] = split.n_panel
        checker.sizes["n_rest"] = split.n_rest
        if axis is not None:
            _compare_orders(
                "gene_split_order", split.symbols, axis,
                "gene_split.csv", "gene_names.csv", checker,
            )

    # ---- phase 1 ---------------------------------------------------------
    checker.run("phase1", paths.phase1_lincs, lambda: _check_phase1(paths, axis, checker))

    # ---- phase 2 ---------------------------------------------------------
    _check_phase2(paths, axis, checker)

    # ---- replogle source files -------------------------------------------
    _check_replogle_source(paths, checker)

    # ---- phase 3 and the release -----------------------------------------
    _check_phase3(paths, axis, man, pert_counts, checker)

    # ---- coverage --------------------------------------------------------
    _check_coverage(paths, checker)
    _summarize_trainable_genes(axis, checker)

    return _finish(cfg, checker, log)


def _summarize_trainable_genes(axis: list[str] | None, checker: Checker) -> None:
    """How many challenge genes the training phases actually measure.

    `gene_coverage.csv` counts a gene as covered when any source file holds
    it, including LINCS genes outside the panel. What matters for the method
    is narrower: the genes Phase 1 or Phase 2 can supervise. The rest are
    reachable only through the challenge controls at Phase 3.
    """
    if axis is None:
        return
    measured = checker.measured_symbols & set(axis)
    controls_only = len(axis) - len(measured)
    checker.sizes["trainable_genes"] = {
        "n_measured_by_phase1_or_phase2": len(measured),
        "n_challenge_controls_only": controls_only,
    }
    checker.ok(
        "trainable_genes",
        f"{len(measured):,} of {len(axis):,} challenge genes are measured by Phase 1 or "
        f"Phase 2; {controls_only:,} are seen only in the challenge controls",
        **checker.sizes["trainable_genes"],
    )


def _check_hgnc(paths: DataPaths, checker: Checker) -> None:
    path = paths.hgnc  # raises FileNotFoundError, naming the glob, when absent
    frame = reference.load_hgnc(path)
    groups = frame["gene_group"].notna().sum()
    checker.ok(
        "hgnc",
        f"{path.name}: {len(frame):,} rows, {groups:,} with a gene_group",
        path=str(path), n_rows=int(len(frame)), n_with_group=int(groups),
    )


def _check_pert_lists(perts: list[str], pert_counts: reference.PertCounts, checker: Checker) -> None:
    """`pert_counts.csv` is the authority; `challenge_perts.txt` must match it."""
    listed, authority = set(perts), set(pert_counts.targets)
    only_txt = sorted(listed - authority)
    only_csv = sorted(authority - listed)
    if only_txt or only_csv:
        checker.fail(
            "pert_lists",
            "ref/challenge_perts.txt does not match pert_counts.csv: "
            f"{len(only_txt)} only in the txt (e.g. {only_txt[:3]}), "
            f"{len(only_csv)} only in the csv (e.g. {only_csv[:3]})",
            only_in_txt=only_txt[:20], only_in_csv=only_csv[:20],
        )
    else:
        checker.ok("pert_lists", f"challenge_perts.txt matches pert_counts.csv ({len(perts):,} targets)")


def _check_cells_per_pert(pert_counts: reference.PertCounts, man: manifest_mod.Manifest,
                          paths: DataPaths, checker: Checker) -> None:
    """A per-target count column must agree with the manifest, exactly.

    Disagreement is a failure reporting both numbers, never a silent choice
    of one over the other.
    """
    if pert_counts.cells_per_pert is None:
        checker.ok(
            "cells_per_pert",
            f"pert_counts.csv has no per-target count column; "
            f"manifest cells_per_pert = {man.cells_per_pert:,} applies",
            manifest_cells_per_pert=man.cells_per_pert,
        )
        return

    disagreeing = {
        target: count
        for target, count in pert_counts.cells_per_pert.items()
        if count != man.cells_per_pert
    }
    if disagreeing:
        sample = sorted(disagreeing.items())[:5]
        checker.fail(
            "cells_per_pert",
            f"pert_counts.csv column {pert_counts.count_column!r} disagrees with "
            f"manifest.json cells_per_pert for {len(disagreeing):,} target(s): "
            f"manifest says {man.cells_per_pert:,}, the file says "
            + ", ".join(f"{t}={c:,}" for t, c in sample),
            manifest_cells_per_pert=man.cells_per_pert,
            count_column=pert_counts.count_column,
            n_disagreeing=len(disagreeing),
            examples={t: int(c) for t, c in sample},
        )
        return
    checker.ok(
        "cells_per_pert",
        f"pert_counts.csv column {pert_counts.count_column!r} agrees with "
        f"manifest cells_per_pert = {man.cells_per_pert:,}",
        manifest_cells_per_pert=man.cells_per_pert,
    )


def _check_phase1(paths: DataPaths, axis: list[str] | None, checker: Checker) -> None:
    obs, var = lincs.read_phase1_metadata(paths.phase1_lincs)

    missing_obs = [c for c in lincs.REQUIRED_OBS if c not in obs.columns]
    missing_var = [c for c in lincs.REQUIRED_VAR if c not in var.columns]
    if missing_obs or missing_var:
        checker.fail(
            "phase1_columns",
            f"{paths.phase1_lincs.name}: missing obs {missing_obs} / var {missing_var}",
            missing_obs=missing_obs, missing_var=missing_var,
        )
        return

    import anndata as ad

    backed = ad.read_h5ad(paths.phase1_lincs, backed="r")
    try:
        present = list(backed.layers.keys())
    finally:
        if backed.file is not None:
            backed.file.close()
    missing_layers = [name for name in lincs.LAYERS if name not in present]
    if missing_layers:
        checker.fail(
            "phase1_layers",
            f"{paths.phase1_lincs.name}: missing layer(s) {missing_layers} (has {present})",
            missing=missing_layers,
        )
    else:
        checker.ok("phase1_layers", f"phase1_lincs.h5ad has layers {list(lincs.LAYERS)}")

    _check_subset_axis("phase1_genes", var, axis, paths.phase1_lincs.name, checker)

    checker.measured_symbols.update(var.index.astype(str))

    n_lines = obs["cell_line"].nunique()
    n_targets = obs["target_gene"].nunique()
    checker.sizes["phase1"] = {
        "n_rows": int(len(obs)),
        "n_cell_lines": int(n_lines),
        "n_targets": int(n_targets),
        "n_genes": int(len(var)),
        "n_has_sig": int(obs["has_sig"].sum()),
        "n_has_gmt": int(obs["has_gmt"].sum()),
        "n_challenge_perts": int(obs["is_challenge_pert"].sum()),
    }
    checker.ok(
        "phase1",
        f"phase1_lincs.h5ad: {len(obs):,} rows, {n_lines:,} cell lines, "
        f"{n_targets:,} targets, {len(var):,} panel genes",
        **checker.sizes["phase1"],
    )


def _check_subset_axis(name: str, var: pd.DataFrame, axis: list[str] | None,
                       file_label: str, checker: Checker) -> None:
    """A file measuring a subset of the genes must still agree with the axis:
    every `challenge_idx` in range, and pointing at the right symbol."""
    if axis is None:
        return
    idx = var["challenge_idx"].to_numpy()
    out_of_range = idx[(idx < 0) | (idx >= len(axis))]
    if out_of_range.size:
        checker.fail(
            name,
            f"{file_label}: {out_of_range.size:,} challenge_idx values outside "
            f"0..{len(axis) - 1} (e.g. {out_of_range[:5].tolist()})",
            n_out_of_range=int(out_of_range.size),
        )
        return

    symbols = var.index.astype(str).tolist()
    expected = [axis[i] for i in idx]
    mismatched = [i for i, (a, b) in enumerate(zip(symbols, expected)) if a != b]
    if mismatched:
        first = mismatched[0]
        checker.fail(
            name,
            f"{file_label}: {len(mismatched):,} gene(s) whose challenge_idx points at a "
            f"different symbol; first is {symbols[first]!r} -> idx {int(idx[first])} "
            f"= {expected[first]!r}",
            n_mismatched=len(mismatched),
        )
        return
    checker.ok(name, f"{file_label}: {len(idx):,} genes agree with the challenge gene axis")


def _check_phase2(paths: DataPaths, axis: list[str] | None, checker: Checker) -> None:
    contexts = paths.discover_phase2_contexts()
    if not contexts:
        checker.fail(
            "phase2", f"no Replogle context directories under {paths.phase2_dir}",
            path=str(paths.phase2_dir),
        )
        return
    checker.ok("phase2_contexts", f"Replogle contexts found: {', '.join(contexts)}",
               contexts=list(contexts))

    sizes: dict[str, Any] = {}
    for name, directory in contexts.items():
        ctx = checker.run(
            f"phase2:{name}", directory, lambda d=directory: replogle.load_replogle_context(d)
        )
        if ctx is None:
            continue

        if axis is not None:
            idx = ctx.challenge_idx
            bad = idx[(idx < 0) | (idx >= len(axis))]
            if bad.size:
                checker.fail(
                    f"phase2:{name}:genes",
                    f"{directory / 'genes.csv'}: {bad.size:,} challenge_idx outside "
                    f"0..{len(axis) - 1}",
                    n_out_of_range=int(bad.size),
                )
            else:
                expected = [axis[i] for i in idx]
                actual = ctx.genes["gene_symbol"].astype(str).tolist()
                mismatched = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
                if mismatched:
                    first = mismatched[0]
                    checker.fail(
                        f"phase2:{name}:genes",
                        f"{directory / 'genes.csv'}: {len(mismatched):,} gene(s) whose "
                        f"challenge_idx points at a different symbol; first is "
                        f"{actual[first]!r} -> {expected[first]!r}",
                        n_mismatched=len(mismatched),
                    )
                else:
                    checker.ok(
                        f"phase2:{name}:genes",
                        f"{name}: {len(idx):,} measured genes agree with the gene axis",
                    )

        pseudobulk = directory / "pseudobulk.h5ad"
        if not pseudobulk.is_file():
            checker.fail(f"phase2:{name}:pseudobulk", f"missing file: {pseudobulk}")
        else:
            checker.ok(f"phase2:{name}:pseudobulk", f"{name}: pseudobulk.h5ad present")

        source = Path(ctx.meta["source"])
        if not source.exists() and ctx.meta.get("source_rel"):
            source = (directory / ctx.meta["source_rel"]).resolve()
        if not source.exists():
            checker.fail(
                f"phase2:{name}:source",
                f"{directory / 'meta.json'} points at single-cell file {source}, "
                "which does not exist (neither 'source' nor 'source_rel' resolves)",
                source=str(source),
            )
        else:
            checker.ok(f"phase2:{name}:source", f"{name}: single-cell source {source.name}")

        checker.measured_symbols.update(ctx.genes["gene_symbol"].astype(str))

        n_cells = len(ctx.cells)
        n_controls = int(ctx.cells["is_control"].sum())
        sizes[name] = {
            "n_cells": n_cells,
            "n_controls": n_controls,
            "n_targets": len(ctx.targets),
            "n_genes": int(len(ctx.genes)),
            "n_panel": ctx.n_panel,
            "n_rest": ctx.n_rest,
        }
        checker.ok(
            f"phase2:{name}",
            f"{name}: {n_cells:,} cells ({n_controls:,} control), {len(ctx.targets):,} targets, "
            f"{len(ctx.genes):,} genes ({ctx.n_panel:,} panel / {ctx.n_rest:,} rest)",
            **sizes[name],
        )

    checker.sizes["phase2"] = sizes


def _check_replogle_source(paths: DataPaths, checker: Checker) -> None:
    singlecell = paths.discover_replogle_singlecell()
    bulk = paths.discover_replogle_bulk()
    if not singlecell:
        checker.fail(
            "replogle_singlecell",
            f"no *_raw_singlecell.h5ad under {paths.replogle_dir}",
            path=str(paths.replogle_dir),
        )
    else:
        checker.ok(
            "replogle_singlecell",
            f"single-cell files: {', '.join(sorted(singlecell))}",
            files=sorted(singlecell),
        )

    raw_bulk = {name: path for name, path in bulk.items() if name.endswith("_raw_bulk")}
    if not raw_bulk:
        checker.fail(
            "replogle_bulk",
            f"no *_raw_bulk.h5ad under {paths.replogle_dir} — the knockdown prior "
            "(fold_expr) has no source",
            path=str(paths.replogle_dir),
        )
        return

    found: dict[str, int] = {}
    for name, path in sorted(raw_bulk.items()):
        series = checker.run(
            f"fold_expr:{name}", path, lambda p=path: replogle.load_fold_expr(p)
        )
        if series is None:
            continue
        found[name] = int(len(series))
        checker.ok(
            f"fold_expr:{name}",
            f"{name}: fold_expr for {len(series):,} targets, median {series.median():.3f}",
            n_targets=int(len(series)), median_fold_expr=float(series.median()),
        )
    checker.sizes["replogle_bulk"] = found


def _check_phase3(paths: DataPaths, axis: list[str] | None, man, pert_counts, checker: Checker) -> None:
    genes = checker.run(
        "phase3_genes", paths.phase3_genes, lambda: challenge.load_phase3_genes(paths.phase3_genes)
    )
    if genes is not None and axis is not None:
        _compare_orders(
            "phase3_genes_order", genes["gene_symbol"].astype(str).tolist(), axis,
            "phase3/genes.csv", "gene_names.csv", checker,
        )

    control_files = paths.discover_phase3_controls()
    release_files = paths.discover_context_files()
    expected_contexts = list(man.contexts) if man is not None else sorted(control_files)

    missing_controls = [c for c in expected_contexts if c not in control_files]
    if missing_controls:
        checker.fail(
            "phase3_controls",
            f"manifest lists context(s) {missing_controls} with no "
            f"controls_<context>.h5ad under {paths.phase3_dir}",
            missing=missing_controls,
        )
    missing_release = [c for c in expected_contexts if c not in release_files]
    if missing_release:
        checker.fail(
            "release_contexts",
            f"manifest lists context(s) {missing_release} with no "
            f"context_<context>.h5ad under {paths.cfg.vcc_root}",
            missing=missing_release,
        )

    context_sizes: dict[str, Any] = {}
    context_col = man.context_col if man is not None else "context"
    control_label = man.control_label if man is not None else "non-targeting"

    for context in expected_contexts:
        path = control_files.get(context)
        if path is None:
            continue
        controls = checker.run(
            f"controls:{context}", path,
            lambda p=path, c=context: challenge.load_challenge_controls(p, c),
        )
        if controls is None:
            continue

        if axis is not None:
            _compare_orders(
                f"controls:{context}:genes", controls.var.index.astype(str).tolist(), axis,
                f"controls_{context}.h5ad var", "gene_names.csv", checker,
            )
        labels = controls.obs.get(context_col)
        if labels is None:
            checker.fail(
                f"controls:{context}:label",
                f"controls_{context}.h5ad has no obs[{context_col!r}] column",
            )
        else:
            unique = sorted(set(labels.astype(str)))
            if unique != [context]:
                checker.fail(
                    f"controls:{context}:label",
                    f"controls_{context}.h5ad obs[{context_col!r}] holds {unique} "
                    f"but the file is context {context!r} — the submission must reuse "
                    "this label exactly",
                    found=unique, expected=context,
                )
            else:
                checker.ok(f"controls:{context}:label", f"{context}: context label matches")

        context_sizes[context] = {
            "n_control_cells": controls.n_cells,
            "n_genes": controls.n_genes,
            "n_ntc_ids": int(controls.obs["ntc_id"].nunique()),
            "median_library_size": float(np.median(controls.library_size)),
        }
        checker.ok(
            f"controls:{context}",
            f"{context}: {controls.n_cells:,} control cells, "
            f"{int(controls.obs['ntc_id'].nunique()):,} ntc ids, "
            f"median library {np.median(controls.library_size):,.0f}",
            **context_sizes[context],
        )

        release = release_files.get(context)
        if release is not None:
            result = checker.run(
                f"release:{context}", release,
                lambda p=release: challenge.read_release_context(p),
            )
            if result is not None:
                obs, var_names = result
                if axis is not None:
                    _compare_orders(
                        f"release:{context}:genes", [str(v) for v in var_names], axis,
                        f"context_{context}.h5ad var", "gene_names.csv", checker,
                    )
                pert_col = man.pert_col if man is not None else "target_gene"
                if pert_col in obs.columns:
                    labels = sorted(set(obs[pert_col].astype(str)))
                    if labels != [control_label]:
                        checker.fail(
                            f"release:{context}:controls_only",
                            f"context_{context}.h5ad should hold control cells only but "
                            f"obs[{pert_col!r}] holds {labels[:5]}",
                            found=labels[:20], expected=control_label,
                        )
                    else:
                        checker.ok(
                            f"release:{context}:controls_only",
                            f"{context}: release file is {control_label!r} cells only",
                        )
                else:
                    checker.fail(
                        f"release:{context}:controls_only",
                        f"context_{context}.h5ad has no obs[{pert_col!r}] column",
                    )

    checker.sizes["contexts_detail"] = context_sizes

    targets = checker.run(
        "phase3_targets", paths.phase3_targets, lambda: challenge.load_targets(paths.phase3_targets)
    )
    if targets is not None and pert_counts is not None:
        _check_targets_table(targets, expected_contexts, pert_counts, checker)


def _check_targets_table(targets: pd.DataFrame, contexts: list[str],
                         pert_counts: reference.PertCounts, checker: Checker) -> None:
    """`targets.csv` must hold every context x every perturbation, once."""
    have = set(zip(targets["context"].astype(str), targets["target_gene"].astype(str)))
    want = {(c, t) for c in contexts for t in pert_counts.targets}
    missing = sorted(want - have)
    extra = sorted(have - want)
    duplicated = int(len(targets) - len(have))

    if missing or extra or duplicated:
        checker.fail(
            "phase3_targets",
            f"phase3/targets.csv: {len(missing):,} missing (e.g. {missing[:3]}), "
            f"{len(extra):,} unexpected (e.g. {extra[:3]}), {duplicated:,} duplicate row(s); "
            f"expected {len(contexts):,} contexts x {len(pert_counts):,} targets "
            f"= {len(want):,} rows",
            n_missing=len(missing), n_extra=len(extra), n_duplicated=duplicated,
        )
        return
    checker.ok(
        "phase3_targets",
        f"phase3/targets.csv: {len(want):,} rows = {len(contexts):,} contexts "
        f"x {len(pert_counts):,} targets",
        n_rows=len(want),
    )


def _check_coverage(paths: DataPaths, checker: Checker) -> None:
    pert = checker.run(
        "pert_coverage", paths.pert_coverage, lambda: coverage.load_pert_coverage(paths.pert_coverage)
    )
    if pert is not None:
        counts = pert.counts_per_source()
        checker.sizes["pert_coverage"] = counts
        covered = int(pert.covered_by_any().sum())
        checker.ok(
            "pert_coverage",
            f"pert_coverage.csv: {len(pert.frame):,} targets, {covered:,} covered by "
            "at least one source",
            n_targets=int(len(pert.frame)), n_covered=covered, per_source=counts,
        )

    gene = checker.run(
        "gene_coverage", paths.gene_coverage, lambda: coverage.load_gene_coverage(paths.gene_coverage)
    )
    if gene is not None:
        covered = int(gene.covered_by_any().sum())
        n_total = int(len(gene.frame))
        checker.sizes["gene_coverage"] = {
            "n_genes": n_total,
            "n_covered": covered,
            "n_challenge_controls_only": n_total - covered,
            "per_source": gene.counts_per_source(),
        }
        checker.ok(
            "gene_coverage",
            f"gene_coverage.csv: {n_total:,} genes, {covered:,} measured in a training "
            f"source, {n_total - covered:,} only in the challenge controls",
            **checker.sizes["gene_coverage"],
        )


def _finish(cfg: Config, checker: Checker, log) -> dict[str, Any]:
    report = {
        "data_root": str(cfg.data_root),
        "vcc_root": str(cfg.vcc_root),
        "sizes": checker.sizes,
        "results": [r.as_dict() for r in checker.results],
        "n_ok": len(checker.results) - len(checker.failed) - len(checker.warned),
        "n_warn": len(checker.warned),
        "n_fail": len(checker.failed),
    }

    run_paths = RunPaths(cfg)
    run_paths.ensure()
    run_paths.check_data_report.write_text(json.dumps(report, indent=2, default=str))

    log.info("")
    log.info("Sizes found — please confirm these are the data you meant to run on:")
    for line in _size_lines(checker.sizes):
        log.info("  %s", line)

    log.info("")
    for result in checker.warned:
        log.warning("WARN  %s: %s", result.name, result.message)
    for result in checker.failed:
        log.error("FAIL  %s: %s", result.name, result.message)

    log.info(
        "check-data: %d ok, %d warn, %d fail  ->  %s",
        report["n_ok"], report["n_warn"], report["n_fail"], run_paths.check_data_report,
    )

    if checker.failed:
        raise DataCheckError(
            f"{len(checker.failed)} data check(s) failed; see {run_paths.check_data_report}"
        )
    return report


def _size_lines(sizes: dict[str, Any]) -> list[str]:
    lines = []
    if "n_challenge_genes" in sizes:
        panel = sizes.get("n_panel")
        rest = sizes.get("n_rest")
        suffix = f" ({panel:,} panel / {rest:,} rest)" if panel is not None else ""
        lines.append(f"challenge genes : {sizes['n_challenge_genes']:,}{suffix}")
    if "contexts" in sizes:
        lines.append(f"contexts        : {', '.join(sizes['contexts'])}")
    if "n_targets" in sizes:
        lines.append(
            f"targets         : {sizes['n_targets']:,} x "
            f"{sizes.get('cells_per_pert', '?')} cells each"
        )
    for context, detail in sizes.get("contexts_detail", {}).items():
        lines.append(
            f"  context {context:<6}: {detail['n_control_cells']:,} control cells, "
            f"median library {detail['median_library_size']:,.0f}"
        )
    if "phase1" in sizes:
        p1 = sizes["phase1"]
        lines.append(
            f"phase 1 (LINCS) : {p1['n_rows']:,} rows, {p1['n_cell_lines']:,} cell lines, "
            f"{p1['n_targets']:,} targets, {p1['n_genes']:,} panel genes"
        )
    for name, detail in sizes.get("phase2", {}).items():
        lines.append(
            f"phase 2 {name:<8}: {detail['n_cells']:,} cells, {detail['n_targets']:,} targets, "
            f"{detail['n_genes']:,} genes"
        )
    if "trainable_genes" in sizes:
        tg = sizes["trainable_genes"]
        lines.append(
            f"trainable genes : {tg['n_measured_by_phase1_or_phase2']:,} measured by phase 1 or 2; "
            f"{tg['n_challenge_controls_only']:,} reachable only through challenge controls"
        )
    if "gene_coverage" in sizes:
        gc = sizes["gene_coverage"]
        lines.append(
            f"gene coverage   : {gc['n_covered']:,} of {gc['n_genes']:,} genes measured in a "
            f"training source; {gc['n_challenge_controls_only']:,} only in challenge controls"
        )
    return lines
