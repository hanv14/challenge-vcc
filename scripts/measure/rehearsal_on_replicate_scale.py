#!/usr/bin/env python
"""Measurement 1: re-read a finished rehearsal on the leaderboard's own column.

    python scripts/measure/rehearsal_on_replicate_scale.py runs/server_shared runs/server_cal

Before D120 the rehearsal reported cell-eval2's `from_baseline` average
(1 = each metric's constant perfection anchor). The leaderboard enrols
`from_replicate` (1 = the measured split-half replicate). This recomputes
every cross-context arm and calibration-grid point of a finished rehearsal
on `from_replicate`, from the raw metrics and the bundle on disk, with
cell-eval2's own `score_one` and catalog policies. Read-only.
"""
import json, os, sys
from dataclasses import replace
import polars as pl
from cell_eval2.catalog import CATALOG
from cell_eval2.scoring import score_one

SIX = ["pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_lfc_nmae",
       "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
       "de_wilcoxon_sig_jaccard"]
SHORT = ["pds", "expr", "nmae", "fid", "reach", "jac"]

def fmt(d):
    return "  ".join(f"{s} {d[m]:+.3f}" if d.get(m) is not None else f"{s}   —  " for m, s in zip(SIX, SHORT))

for run in sys.argv[1:]:
  print(f"\n======== {run}")
  report = json.load(open(os.path.join(run, "rehearsal", "report.json")))
  for entry in report["variants"]:
      if entry["variant"] != "cross_context":
          continue
      ctx = entry["context"]
      bdir = os.path.join(run, "rehearsal", "scale", f"cross_context_{ctx}")
      if not os.path.exists(os.path.join(bdir, "manifest.json")):
          print(f"\n### {ctx}: no bundle"); continue
      man = json.load(open(os.path.join(bdir, "manifest.json")))
      base = pl.read_csv(os.path.join(bdir, "baseline_agg.csv")).filter(pl.col("statistic") == "mean")
      anch = pl.read_parquet(os.path.join(bdir, "anchor_agg.parquet"))
      b = {m: float(base[m][0]) for m in SIX}
      r = {row["metric"]: float(row["replicate"]) for row in anch.iter_rows(named=True) if row["metric"] in SIX}
      pol = {m: replace(CATALOG[m].scoring, anchor=r[m], allow_negative_baseline=False) for m in SIX}

      def lb(raw):
          per = {m: score_one(raw.get(m), b[m], pol[m]) for m in SIX}
          return sum(per.values()) / len(SIX), per

      print(f"\n### {ctx}  ({entry.get('n_targets')} targets)  rule_digest={man.get('rule_digest')}")
      print("    rule_mismatches:", man.get("rule_mismatches"))
      print("    baseline (0):  " + fmt(b))
      print("    replicate (1): " + fmt(r))
      print("    arms:  arm | from_baseline avg (what the rehearsal reported) | from_replicate avg (leaderboard's column)")
      for arm, d in entry["arms"].items():
          sc = d.get("official") or {}
          raw = sc.get("scored") or {}
          if not raw:
              print(f"      {arm:12s} no score: {sc.get('error')}"); continue
          fb = (sc.get("leaderboard") or {}).get("avg_score")
          avg, per = lb(raw)
          print(f"      {arm:12s} {fb if fb is None else round(fb, 4)!s:>8}  {avg:+.4f}   " + fmt(per))
      grid = entry.get("calibration_grid") or []
      if grid:
          print("    grid: thr scale shared | from_baseline | from_replicate | per metric (from_replicate)")
          rows = []
          for p in grid:
              if p.get("error") or not p.get("metrics"):
                  continue
              avg, per = lb(p["metrics"])
              rows.append((avg, p, per))
          for avg, p, per in sorted(rows, key=lambda x: -x[0]):
              fb = p.get("leaderboard")
              print(f"      {p['confidence_threshold']:.2f} {p['effect_scale']:4.2f} {p.get('shared_scale', -1.0):5.2f} | "
                    f"{fb if fb is None else round(fb, 4)!s:>8} | {avg:+.4f} | " + fmt(per))
