"""The submission file, the `vcc` tool and the checklist (§8, §4.8).

The validator's rule-by-rule tests are in `test_validator.py`; these are
about the file the pipeline actually writes — that it is one file for every
context, sparse, with 64-bit `indptr`, no control rows and the gene axis in
`gene_names.csv` order — and about what the run says it produced.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vccp import checklist as checklist_mod
from vccp.submit import package, validator
from vccp.submit.spec import load_spec


@pytest.fixture(scope="module")
def spec(pipeline_cfg):
    return load_spec(pipeline_cfg)


def test_one_file_holds_every_context(pipeline_cfg, pipeline_run, spec):
    import anndata as ad

    backed = ad.read_h5ad(pipeline_run.submission, backed="r")
    try:
        contexts = sorted(set(backed.obs[spec.context_col].astype(str)))
        targets = sorted(set(backed.obs[spec.pert_col].astype(str)))
        n_obs = backed.n_obs
        axis = list(backed.var_names)
    finally:
        backed.file.close()

    assert contexts == sorted(spec.contexts), "one file, obs['context'] separates them (§8)"
    assert targets == sorted(spec.targets)
    assert n_obs == spec.total_cells
    assert axis == list(spec.axis)


def test_no_control_rows_are_written(pipeline_run, spec):
    import anndata as ad

    backed = ad.read_h5ad(pipeline_run.submission, backed="r")
    try:
        labels = set(backed.obs[spec.pert_col].astype(str))
    finally:
        backed.file.close()
    assert spec.control_label not in labels


def test_the_matrix_is_sparse_integer_with_a_64_bit_indptr(pipeline_run):
    """§8: at the real shape a 32-bit `indptr` overflows."""
    import h5py

    with h5py.File(pipeline_run.submission, "r") as handle:
        assert handle["X"].attrs["encoding-type"] == "csr_matrix"
        assert handle["X/indptr"].dtype == np.int64
        assert handle["X/data"].dtype.kind in "iu"
        data = handle["X/data"][:]
    assert (data != 0).all(), "no explicitly stored zeros (§8)"
    assert (data > 0).all()


def test_the_written_file_passes_our_own_validator(pipeline_cfg, pipeline_run, spec):
    result = validator.validate_file(pipeline_cfg, spec, pipeline_run.submission)
    assert result.ok, result.summary()


def test_validate_json_records_the_tool_and_the_totals(pipeline_run):
    report = json.loads(pipeline_run.validate_report.read_text())
    assert report["ok"] is True
    assert report["totals"]["n_rows"] > 0
    assert "vcc_prep" in report
    if not report["vcc_prep"]["ran"]:
        assert report["vcc_prep"]["skipped_reason"]


def test_the_challenge_tool_is_optional_and_says_so(pipeline_run):
    """`vcc` is not on PATH in the cloud; its absence is reported, not fatal."""
    report = json.loads((pipeline_run.submission_dir / "package.json").read_text())
    if package.tool_path() is None:
        assert report["ran"] is False
        assert "not on PATH" in report["skipped_reason"]
        assert report["vcc"] is None
    else:
        assert report["ran"] is True


def test_validate_refuses_a_run_whose_sanity_checks_warned(pipeline_cfg, pipeline_run):
    """§6.1: warnings make `validate` refuse unless they are accepted.

    `mini.yaml` accepts them through `sanity.accept_warnings` (DECISIONS.md
    D41), which is what lets `all` finish there; this turns that off, as
    `server.yaml` leaves it, and checks that the gate is real.
    """
    import dataclasses

    from vccp.submit.stage import run_validate

    report = json.loads(pipeline_run.sanity_report.read_text())
    if report["clean"]:
        pytest.skip("this run produced no warnings to refuse")

    strict = dataclasses.replace(
        pipeline_cfg, sanity=dataclasses.replace(pipeline_cfg.sanity, accept_warnings=False)
    )
    with pytest.raises(RuntimeError, match="allow-warnings"):
        run_validate(strict, None)


def test_the_server_config_does_not_accept_warnings_silently(repo_root):
    from vccp.config import load_config

    assert load_config(repo_root / "configs" / "server.yaml").sanity.accept_warnings is False


def test_validate_refuses_to_run_before_sanity(pipeline_cfg, tmp_path):
    import dataclasses

    from vccp.submit.stage import run_validate

    empty = dataclasses.replace(pipeline_cfg, output_root=tmp_path / "empty")
    with pytest.raises(RuntimeError, match="run the `sanity` stage"):
        run_validate(empty, None)


def test_a_block_of_the_wrong_shape_is_refused(pipeline_cfg, pipeline_run, spec, tmp_path):
    import scipy.sparse as sp

    from vccp.submit.writer import load_block

    path = tmp_path / "short.npz"
    sp.save_npz(path, sp.csr_matrix(np.ones((2, spec.n_genes), dtype=np.int32)))
    with pytest.raises(ValueError, match="cells x genes"):
        load_block(path, spec.targets[0], spec.cells_for(spec.targets[0]), spec.n_genes)


# --------------------------------------------------------------------------- #
# the checklist
# --------------------------------------------------------------------------- #
def test_every_element_is_done_or_an_explained_deviation(pipeline_cfg, pipeline_run):
    report = json.loads(pipeline_run.checklist.read_text())
    assert report["n_items"] == 15
    assert report["n_missing"] == 0, [
        item for item in report["items"] if item["status"] == "missing"
    ]
    assert report["complete"] is True
    for item in report["items"]:
        if item["status"] == checklist_mod.DEVIATION:
            assert item["reason"], item


def test_every_named_artifact_exists(pipeline_run):
    report = json.loads(pipeline_run.checklist.read_text())
    for item in report["items"]:
        for name, present in item["artifacts"].items():
            if item["status"] != checklist_mod.DEVIATION:
                assert present, f"item {item['item']} is missing {name}"


def test_a_missing_artifact_is_reported_as_missing_not_as_a_deviation(
    pipeline_cfg, tmp_path
):
    import dataclasses

    from vccp.paths import RunPaths

    empty = RunPaths(dataclasses.replace(pipeline_cfg, output_root=tmp_path / "nothing"))
    empty.ensure()
    report = checklist_mod.build(pipeline_cfg, empty)
    assert report["n_missing"] > 0
    assert report["complete"] is False


# --------------------------------------------------------------------------- #
# the results summary
# --------------------------------------------------------------------------- #
def test_the_summary_says_mini_numbers_are_not_indicative(pipeline_run):
    text = pipeline_run.summary_md.read_text()
    assert "not indicative" in text
    assert "Element checklist" in text
    assert "PANEL-ASSISTED" in text or "panel-assisted" in text


def test_the_summary_draws_figures(pipeline_run):
    figures = sorted(pipeline_run.figures.glob("*.png"))
    assert figures, "the defense needs figures, not only tables"
    for figure in figures:
        assert figure.stat().st_size > 1000
        assert f"figures/{figure.name}" in pipeline_run.summary_md.read_text()
