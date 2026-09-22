"""`check-data` — it passes on good data, and names the file on bad data.

This is the stage that has to give the user a clear pass/fail on the server
*before* any training starts (CLAUDE.md §1.1), so each test breaks one thing
in a symlink mirror of `mini_data` and asserts the failure points at it.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from vccp.data.checks import FAIL, DataCheckError, check_data
from vccp.paths import DataPaths, RunPaths

from tests.conftest import replace_in_mirror, unlink_in_mirror


def failures(report_path) -> dict[str, str]:
    report = json.loads(report_path.read_text())
    return {r["name"]: r["message"] for r in report["results"] if r["status"] == FAIL}


def run_expecting_failure(cfg) -> dict[str, str]:
    with pytest.raises(DataCheckError):
        check_data(cfg)
    return failures(RunPaths(cfg).check_data_report)


def test_check_data_passes_on_mini(mini_cfg):
    report = check_data(mini_cfg)
    assert report["n_fail"] == 0
    assert report["n_ok"] > 0


def test_check_data_reports_the_sizes_it_found(mini_cfg):
    """The user confirms the data by these numbers, so they must be present
    and must come from the data rather than from a default."""
    report = check_data(mini_cfg)
    sizes = report["sizes"]

    assert sizes["n_challenge_genes"] == sizes["n_panel"] + sizes["n_rest"]
    assert sizes["n_targets"] > 0
    assert sizes["cells_per_pert"] > 0
    assert len(sizes["contexts"]) == len(sizes["contexts_detail"])
    assert sizes["phase1"]["n_cell_lines"] > 0
    assert set(sizes["phase2"]) == set(DataPaths(mini_cfg).discover_phase2_contexts())
    assert (
        sizes["trainable_genes"]["n_measured_by_phase1_or_phase2"]
        + sizes["trainable_genes"]["n_challenge_controls_only"]
        == sizes["n_challenge_genes"]
    )


def test_check_data_writes_its_report(mini_cfg):
    check_data(mini_cfg)
    report_path = RunPaths(mini_cfg).check_data_report
    assert report_path.is_file()
    assert json.loads(report_path.read_text())["results"]


def test_missing_file_fails_and_names_it(mirrored_cfg):
    paths = DataPaths(mirrored_cfg)
    unlink_in_mirror(paths.phase1_lincs)

    found = run_expecting_failure(mirrored_cfg)
    assert "phase1" in found
    assert "phase1_lincs.h5ad" in found["phase1"]


def test_missing_replogle_source_file_fails(mirrored_cfg):
    """`meta.json` points at the single-cell file the cells are streamed from;
    if neither its absolute nor its relative path resolves, training would
    fail hours later instead of here."""
    paths = DataPaths(mirrored_cfg)
    name, directory = next(iter(paths.discover_phase2_contexts().items()))
    meta = json.loads((directory / "meta.json").read_text())
    # Not `.resolve()` — that would follow the mirror's link into mini_data.
    unlink_in_mirror(directory / meta["source_rel"])

    found = run_expecting_failure(mirrored_cfg)
    assert f"phase2:{name}:source" in found


def test_disagreeing_gene_order_fails(mirrored_cfg):
    """A file whose gene order differs silently corrupts every downstream
    index, so the orders are compared element by element."""
    paths = DataPaths(mirrored_cfg)
    names = pd.read_csv(paths.gene_names)
    swapped = names.copy()
    swapped.iloc[0], swapped.iloc[1] = names.iloc[1].copy(), names.iloc[0].copy()
    replace_in_mirror(paths.gene_names, lambda p: swapped.to_csv(p, index=False))

    found = run_expecting_failure(mirrored_cfg)
    assert "gene_axis" in found
    assert "disagree" in found["gene_axis"]


def test_shorter_gene_axis_fails_with_both_lengths(mirrored_cfg):
    paths = DataPaths(mirrored_cfg)
    names = pd.read_csv(paths.gene_names).iloc[:-1]
    replace_in_mirror(paths.gene_names, lambda p: names.to_csv(p, index=False))

    found = run_expecting_failure(mirrored_cfg)
    assert "gene_axis" in found
    assert "genes but" in found["gene_axis"]


def test_pert_list_mismatch_fails(mirrored_cfg):
    paths = DataPaths(mirrored_cfg)
    perts = [line for line in paths.challenge_perts.read_text().splitlines() if line.strip()]
    replace_in_mirror(
        paths.challenge_perts, lambda p: p.write_text("\n".join(perts[:-1]) + "\n")
    )

    found = run_expecting_failure(mirrored_cfg)
    assert "pert_lists" in found


def test_per_target_cell_count_disagreeing_with_manifest_fails_with_both_numbers(mirrored_cfg):
    """Decided at the M0 review: never silently prefer one over the other
    (PLAN.md §8.11)."""
    paths = DataPaths(mirrored_cfg)
    manifest = json.loads(paths.manifest.read_text())
    wrong = manifest["cells_per_pert"] + 1

    frame = pd.read_csv(paths.pert_counts)
    frame["n_cells"] = wrong
    replace_in_mirror(paths.pert_counts, lambda p: frame.to_csv(p, index=False))

    found = run_expecting_failure(mirrored_cfg)
    assert "cells_per_pert" in found
    message = found["cells_per_pert"]
    assert str(manifest["cells_per_pert"]) in message, "the manifest's number must be reported"
    assert str(wrong) in message, "the file's number must be reported"


def test_per_target_cell_count_agreeing_with_manifest_passes(mirrored_cfg):
    paths = DataPaths(mirrored_cfg)
    manifest = json.loads(paths.manifest.read_text())

    frame = pd.read_csv(paths.pert_counts)
    frame["n_cells"] = manifest["cells_per_pert"]
    replace_in_mirror(paths.pert_counts, lambda p: frame.to_csv(p, index=False))

    assert check_data(mirrored_cfg)["n_fail"] == 0


def test_targets_table_missing_a_row_fails(mirrored_cfg):
    paths = DataPaths(mirrored_cfg)
    targets = pd.read_csv(paths.phase3_targets).iloc[1:]
    replace_in_mirror(paths.phase3_targets, lambda p: targets.to_csv(p, index=False))

    found = run_expecting_failure(mirrored_cfg)
    assert "phase3_targets" in found
    assert "missing" in found["phase3_targets"]


def test_manifest_gene_count_disagreeing_fails(mirrored_cfg):
    paths = DataPaths(mirrored_cfg)
    manifest = json.loads(paths.manifest.read_text())
    manifest["n_genes"] = manifest["n_genes"] + 1
    replace_in_mirror(paths.manifest, lambda p: p.write_text(json.dumps(manifest)))

    found = run_expecting_failure(mirrored_cfg)
    assert "manifest_n_genes" in found


def test_release_context_with_a_perturbed_row_fails(mirrored_cfg):
    """The release files are control cells only; a perturbed row there would
    mean the round's inputs are not what the pipeline assumes (§3.3)."""
    import anndata as ad

    paths = DataPaths(mirrored_cfg)
    manifest = json.loads(paths.manifest.read_text())
    context = manifest["contexts"][0]
    path = paths.context_h5ad(context)

    adata = ad.read_h5ad(path)
    labels = adata.obs[manifest["pert_col"]].astype(str)
    labels.iloc[0] = "SOME_TARGET"
    adata.obs[manifest["pert_col"]] = labels.values
    replace_in_mirror(path, lambda p: adata.write_h5ad(p))

    found = run_expecting_failure(mirrored_cfg)
    assert f"release:{context}:controls_only" in found


def test_control_file_with_a_wrong_context_label_fails(mirrored_cfg):
    """The submission must reuse the control file's context label exactly
    (CLAUDE.md §8), so a label that disagrees with the file name is fatal."""
    import anndata as ad

    paths = DataPaths(mirrored_cfg)
    manifest = json.loads(paths.manifest.read_text())
    context = manifest["contexts"][0]
    path = paths.phase3_controls(context)

    adata = ad.read_h5ad(path)
    adata.obs[manifest["context_col"]] = "NOT_" + context
    replace_in_mirror(path, lambda p: adata.write_h5ad(p))

    found = run_expecting_failure(mirrored_cfg)
    assert f"controls:{context}:label" in found
