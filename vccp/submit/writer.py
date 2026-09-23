"""Assembling the one submission file (CLAUDE.md §8).

**One** `.h5ad` holds every context of the round; `obs["context"]` separates
them. The validation round is 360,000 × 18,533 with on the order of 2×10⁹
stored entries, so the file is built **block by block**: each target's cells
are read back from `predictions/model/`, appended to an on-disk CSR dataset
and dropped. Nothing concatenates the submission in RAM, and `indptr` is
64-bit from the first block because at that size a 32-bit one overflows.

The control cells are inputs only. They are resampled to build predicted
cells and never written here (§8).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_utils import get_logger
from .spec import SubmissionSpec

#: Counts fit in int32 by construction — no cell may exceed
#: `submission.max_counts_per_cell`, and a single gene cannot exceed a cell.
COUNT_DTYPE = np.int32
#: §8: with 2×10⁹ stored entries a 32-bit `indptr` overflows.
INDPTR_DTYPE = np.int64


@dataclass
class WriteReport:
    path: str
    n_rows: int = 0
    n_blocks: int = 0
    stored_entries: int = 0
    bytes_on_disk: int = 0
    contexts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "n_rows": self.n_rows,
            "n_blocks": self.n_blocks,
            "stored_entries": self.stored_entries,
            "bytes_on_disk": self.bytes_on_disk,
            "rows_per_context": self.contexts,
            "note": "one file for every context of the round; obs['context'] "
            "separates them, and no control rows are present (§8)",
        }


def load_block(path: Path, target: str, expected_rows: int, n_genes: int):
    """One target's predicted cells, in the shape §8 requires."""
    import scipy.sparse as sp

    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: run the `predict` stage before assembling the submission"
        )
    block = sp.load_npz(path).tocsr()
    if block.shape != (expected_rows, n_genes):
        raise ValueError(
            f"{path} holds {block.shape[0]} x {block.shape[1]} cells x genes, but "
            f"{target} needs {expected_rows} x {n_genes}"
        )
    block.eliminate_zeros()
    block = block.astype(COUNT_DTYPE)
    block.indptr = block.indptr.astype(INDPTR_DTYPE)
    return block


def write_submission(cfg, spec: SubmissionSpec, run_paths) -> WriteReport:
    """Assemble `submission/prediction.h5ad` from the per-target blocks."""
    import h5py
    import pandas as pd
    from anndata.io import sparse_dataset, write_elem

    log = get_logger()
    path = run_paths.submission
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and renamed at the end, so an interrupted run
    # never leaves a half-written file that looks submittable.
    staging = path.with_suffix(".h5ad.tmp")

    report = WriteReport(path=str(path))
    contexts: list[str] = []
    targets: list[str] = []

    with h5py.File(staging, "w") as handle:
        handle.attrs["encoding-type"] = "anndata"
        handle.attrs["encoding-version"] = "0.1.0"
        dataset = None

        for context in spec.contexts:
            rows_here = 0
            for target in spec.targets:
                block = load_block(
                    run_paths.prediction_block(context, target),
                    target,
                    spec.cells_for(target),
                    spec.n_genes,
                )
                if dataset is None:
                    write_elem(handle, "X", block)
                    dataset = sparse_dataset(handle["X"])
                else:
                    dataset.append(block)

                rows_here += block.shape[0]
                report.n_blocks += 1
                report.stored_entries += int(block.nnz)
                contexts.extend([context] * block.shape[0])
                targets.extend([target] * block.shape[0])
                del block

            report.contexts[context] = rows_here
            log.info("  %s: %d rows appended", context, rows_here)

        report.n_rows = len(contexts)
        obs = pd.DataFrame(
            {
                spec.pert_col: pd.Categorical(targets, categories=list(spec.targets)),
                spec.context_col: pd.Categorical(contexts, categories=list(spec.contexts)),
            },
            index=pd.Index([f"cell_{i}" for i in range(report.n_rows)], name=None),
        )
        var = pd.DataFrame(index=pd.Index(list(spec.axis), name=None))
        write_elem(handle, "obs", obs)
        write_elem(handle, "var", var)

    os.replace(staging, path)
    report.bytes_on_disk = path.stat().st_size
    log.info(
        "submission written: %d rows x %d genes, %d stored entries, %.1f MB -> %s",
        report.n_rows, spec.n_genes, report.stored_entries,
        report.bytes_on_disk / 1024**2, path,
    )
    return report
