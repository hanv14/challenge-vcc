"""The lookup proxy: leaderboard-like scores inside one screen (D128).

The scores are checked on cases whose answer is known (a perfect prediction,
no prediction, every sign wrong), then the script is run end to end on
`mini_data`, whose numbers mean nothing but whose report must be whole.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np

from tests.conftest import MINI_CONFIG, REPO_ROOT
from vccp.eval import lookup_proxy as score


def load_script():
    path = REPO_ROOT / "scripts" / "measure" / "lookup_proxy.py"
    spec = importlib.util.spec_from_file_location("lookup_proxy_script", path)
    module = importlib.util.module_from_spec(spec)
    # Registered first: the script's dataclasses resolve their module by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def truth_panel(n=12, genes=40, seed=0):
    return np.random.default_rng(seed).normal(size=(n, genes))


def test_pds_ranks_a_perfect_prediction_first_and_no_prediction_in_the_middle():
    truth = truth_panel()
    own = np.arange(truth.shape[0])
    np.testing.assert_allclose(score.pds_like(truth, truth, own), 1.0)
    np.testing.assert_allclose(score.pds_like(np.zeros_like(truth), truth, own), 0.5)


def test_pds_drops_excluded_genes_and_each_queries_own_gene():
    truth = truth_panel()
    own = np.arange(truth.shape[0])
    pred = np.zeros_like(truth)
    pred[:, 0] = truth[:, 0] * 100      # all of the prediction sits on gene 0
    exclude = np.zeros(truth.shape[1], dtype=bool)
    exclude[0] = True
    np.testing.assert_allclose(score.pds_like(pred, truth, own, exclude=exclude), 0.5)
    np.testing.assert_allclose(
        score.pds_like(pred, truth, own, own_gene=np.zeros(truth.shape[0], dtype=int)), 0.5)


def test_nmae_reads_one_for_no_change_and_zero_for_the_truth():
    truth = truth_panel()
    significant = np.ones_like(truth, dtype=bool)
    np.testing.assert_allclose(score.nmae_like(np.zeros_like(truth), truth, significant), 1.0)
    np.testing.assert_allclose(score.nmae_like(truth, truth, significant), 0.0)
    few = np.zeros_like(significant)
    few[:, :3] = True
    assert np.all(np.isnan(score.nmae_like(truth, truth, few)))


def test_reach_is_one_when_every_sign_is_right_and_zero_when_none_is():
    truth = truth_panel()
    significant = np.ones_like(truth, dtype=bool)
    confidence = np.abs(truth)
    np.testing.assert_allclose(score.reach_like(truth, confidence, truth, significant), 1.0)
    np.testing.assert_allclose(score.reach_like(-truth, confidence, truth, significant), 0.0)
    # A prediction of exactly zero has no sign, so it is never right.
    np.testing.assert_allclose(
        score.reach_like(np.zeros_like(truth), confidence, truth, significant), 0.0)


def test_bh_calls_what_survives_and_nothing_outside_the_valid_genes():
    p = np.array([[0.001, 0.01, 0.02, 0.5, 0.0001]])
    valid = np.array([[True, True, True, True, False]])
    called = score.bh_significant(p, valid, alpha=0.05)
    # m = 4 valid: thresholds 0.0125, 0.025, 0.0375, 0.05 for ranks 1..4.
    np.testing.assert_array_equal(called, [[True, True, True, False, False]])


def test_the_script_runs_end_to_end_on_mini(tmp_path, capsys):
    script = load_script()
    out = tmp_path / "out"
    argv = ["--config", str(MINI_CONFIG), "--set", f"output_root={out}", "--targets", "all",
            "--pairings", "1", "--bootstrap", "20", "--min-gate", "1",
            "--snr-taus", "1", "--hard-taus", "4", "--ranks", "3", "--combined", "3:1"]
    assert script.main(argv) == 0
    reports = list((out / "measure" / "measure").glob("lookup_proxy_*_all.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert set(report["panels"]) == {"round", "all"}
    for panel in report["panels"].values():
        arms = panel["arms"]
        assert {"raw", "gene", "snr tau=1", "snr_hard tau=4", "lowrank k=3",
                "lowrank_snr k=3 tau=1"} <= set(arms)
        assert abs(arms["raw"]["energy"] - 1.0) < 1e-6 and arms["raw"]["pds_gain"] == 0.0
        assert panel["gate"]["verdict"] in ("pass", "fail")
    # The round's targets come from the all-target cache: no second pass.
    capsys.readouterr()
    assert script.main(["--config", str(MINI_CONFIG), "--set", f"output_root={out}",
                        "--targets", "round", "--pairings", "1", "--bootstrap", "20",
                        "--min-gate", "1"]) == 0
    log = next((out / "measure" / "measure").glob("lookup_proxy_*_round.log")).read_text()
    assert "taking the round's targets from" in log
