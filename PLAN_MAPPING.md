# Next cycle: a trustworthy instrument, then the panel→rest mapping

Written 2026-09-28, after the first full server runs. Two parts, in order,
because the second cannot be read without the first.

**Part A** establishes how much a number moves between runs, now that the
arithmetic is deterministic (DECISIONS.md D99). Every comparison from here
needs that figure, and four of the last six conclusions were drawn without it.

**Part B** attacks the panel→rest mapping, which is the step between the
perturbation module (which works) and the submission (which does not). It
starts by asking what is achievable rather than by trying to improve it.

Nothing here changes the shipped configuration until a measurement says so.
`configs/server.yaml` remains the submission path, `perturbation_mode: pooled`,
`warm_start: none`.

---

## 1. The five facts this plan starts from

All from the seed-2 `all` run and the diagnostics beside it.

1. **The perturbation module learns.** Held-out targets, 908 of them:
   `pert_mse_ratio_to_no_change` 0.847 and `pert_pearson` 0.360 aggregate;
   per screen 0.903/0.323 (K562_essential), 0.985/0.167 (K562_gwps),
   0.652/0.590 (rpe1). It is weakest on the screen holding 272 of the 300
   validation targets.
2. **The mapping does not.** `map_control_mse` is 0.965, 1.144 and 1.109 per
   screen, aggregate 1.072, in units where **predicting each gene's control
   mean scores exactly 1.0**. `map_control_pearson` has never exceeded 0.08 in
   any run. On two of three screens the trained mapping is *worse on control
   cells than the zero-initialized model it started from*, and control cells
   are data it trains on.
3. **Controls-only adaptation makes it worse still.** Rehearsal variant 1,
   rest genes: 2.030, 2.167, 2.505 against a floor of 1.0. Phase 3 no longer
   adapts (D94), which is why the submission is at the floor rather than below
   it.
4. **The per-cell OT premise is refuted.** Pairing controls against controls
   reproduces 95–107% of the per-cell target variation that pairing them
   against perturbed cells does, and removes the same share of the residual
   (D100). `pooled` is and stays the default.
5. **End to end the submission sits at the no-change floor.** Leaderboard
   scale: method −0.038/−0.030/−0.253 against a floor of
   −0.041/−0.059/−0.263, with the upper bound at +0.053/+0.049/+0.079. About
   10% of the available headroom is captured, and sanity check 7 reports the
   median target calling 0.00× as many significant genes as the reference.

Fact 2 is the one worth a cycle. Facts 1 and 5 together say the panel is
predicted usefully and the other 17,578 genes are not, and those genes are
most of what the metrics see.

---

## 2. Part A — how much does a number move?

### A0. Determinism actually achieved (16 minutes, first)

D99 turned the fused attention backends off. That has not run on the GPU. Two
short runs of one configuration must now agree **exactly**.

```bash
source scripts/server_env.sh
cd /data/han/projects/VCC

for rep in a b; do
  name=noise_det_$rep
  mkdir -p runs/$name && cp -r runs/server/priors runs/server/checkpoints runs/$name/
  nice -n 10 ionice -c3 python -m vccp phase2 \
    --config configs/server_isolate.yaml --run-name $name --seed 2 \
    --set phase2.steps=1000 --set train.evals_per_run=1 \
    2>&1 | tee runs/$name/phase2.log
done

python - <<'PY'
import json
a, b = (json.load(open(f"runs/noise_det_{r}/phase2/metrics.json")) for r in "ab")
va, vb = (m["arms"][m["main_arm"]]["validation_final"] for m in (a, b))
same = {k: va[k] == vb[k] for k in sorted(va)}
print(f"{sum(same.values())}/{len(same)} metrics identical to the last bit")
for k, ok in same.items():
    if not ok:
        print(f"  DIFFERS {k}: {va[k]!r} vs {vb[k]!r}")
PY
```

Also check the header line of either log:

```
determinism   : on (tf32 off, deterministic kernels, fused attention off)
```

**Exit condition.** Every metric identical. If any differ, the header names
which backend is still enabled and Part A stops there — the protocol in A2
would be measuring two things at once, and the remedy is another switch in
`_deterministic_attention`, not another run.

**Run: passed** (D103). 26 of 26 metrics identical to the last bit, header
"fused attention off".

### A1. Seed-to-seed spread (2.2 hours)

With the arithmetic fixed, a configuration has one answer per seed. What
varies is the seed: which targets are held out, which cells are drawn, how
the weights initialize. That spread is the error bar on every one-run-per-
configuration comparison, and it has never been measured on a deterministic
run.

```bash
for seed in 0 1 2; do
  name=noise_seed$seed
  mkdir -p runs/$name && cp -r runs/server/priors runs/server/checkpoints runs/$name/
  nice -n 10 ionice -c3 python -m vccp phase2 \
    --config configs/server_isolate.yaml --run-name $name --seed $seed \
    2>&1 | tee runs/$name/phase2.log
done

python - <<'PY'
import json, statistics as st
key = "pert_mse_ratio_to_no_change"
best, last = [], []
for seed in (0, 1, 2):
    m = json.load(open(f"runs/noise_seed{seed}/phase2/metrics.json"))
    arm = m["arms"][m["main_arm"]]
    best.append(arm["validation"][key])
    last.append(arm["validation_final"][key])
    print(seed, "best", round(best[-1], 4), "at", arm["best_step"],
          "| last", round(last[-1], 4), "| retained", arm["signal_retained"])
for name, values in (("best", best), ("last", last)):
    print(f"{name}: mean {st.mean(values):.4f}  sd {st.stdev(values):.4f}  "
          f"range {max(values) - min(values):.4f}")
PY
```

**Run** (D103): best 0.7802 / 0.8092 / 0.7915 for seeds 0 / 1 / 2, mean
0.7936, **sd 0.0146**, range 0.0290; all three end at or beside their best
(retained 0.99–1.00).

### A2. What the answer licenses

The sd across three seeds, call it `s`, sets the protocol for the rest of the
project:

| `s` | what a comparison needs |
|---|---|
| under 0.02 | one seed per configuration is enough for a 0.05 effect |
| 0.02–0.06 | three shared seeds, compared **paired** (same seeds both sides) |
| over 0.06 | three seeds is not enough; only effects above ~0.15 are readable, and the metric itself needs strengthening (more `eval_targets`, more `delta_eval_cells`) before anything else |

Whatever it is, it goes in `DECISIONS.md` as the standing error bar, and every
later claim states how many seeds are behind it. The three earlier readings
that turned out to be noise — D84's seeding, D96's paired measurement, D99's
arithmetic — all shared one cause: a difference was reported without the
spread of the thing it was a difference of.

**Deliverable.** `DECISIONS.md` entry giving `s` for the selection metric, and
a note in `README.md` under "What to read before you submit".

---

## 3. Part B — the panel→rest mapping

### B0. Read the curve we already have (minutes, no GPU)

`runs/server/phase2/curves.csv` records `map_control` every 25 steps — the
mapping's **training** loss on control cells, beside `map_perturbed`,
`map_delta`, `pert_bulk` and `pert_cells`. Fact 2 says the *validation*
number ends worse than the zero-initialized model. The training curve says
whether the optimiser is also failing on the data in front of it, and the two
answers point at different problems.

```bash
python - <<'PY'
import csv
rows = list(csv.DictReader(open("runs/server/phase2/curves.csv")))
cols = ["map_control", "map_perturbed", "map_delta", "pert_bulk", "pert_cells"]
print("step  " + "  ".join(f"{c:>12}" for c in cols))
for row in rows[:: max(1, len(rows) // 12)]:
    print(f"{row['step']:>5} " + "  ".join(
        f"{float(row[c]):12.4f}" if row.get(c) not in (None, "") else f"{'—':>12}"
        for c in cols))
PY
```

Three readings, each with a different consequence:

* **`map_control` falls while validation rises** → the mapping is fitting
  something that does not generalize, at the level of individual cells. Go to
  B1 to find out whether anything generalizes there.
* **`map_control` rises too** → the shared objective is pulling against it.
  The mapping, the perturbation module and the delta term share one value
  head and one set of latents, weighted `loss_mapping: 1.0`,
  `loss_perturbation: 1.0`, `loss_delta: 0.5`. A sweep of those three weights
  is then the cheapest experiment in this document, and B1 tells us what the
  mapping could reach if it won that argument.
* **`map_control` is flat near its starting value** → the mapping is receiving
  no useful gradient at all, and that is a bug to find, not a weight to tune.
  Candidates: the 2048-gene subsample per step against ~7,000 measured rest
  genes; the mask from `genes.csv`; the `mapping_no_change` normalization.

**Read** (D104): flat. `map_control` bounces between 0.92 and 1.11 with no
trend over 6,000 steps. But the logged value is a different 16 cells each
time, and its movement is mostly common to every predictor, so a ridge-sized
gain (0.014) would be invisible in it either way. The paired readings are
`val_map_control_mse` in the same file (fixed cells at every evaluation) and
D102's held-out 1.144 / 1.109 — those are what say the mapping moves
backwards. Which of the three bullets above holds is still open; §B3 arm 1
with `loss_delta=0` as well separates "fits noise" from "dragged by the shared
objectives".

**Then measured on the paired curve, and the reading above is superseded**
(D105–D107). `val_map_control_mse` is flat within every run, and its level is
set by the seed's 32-cell draw: 0.97 for seed 0, 1.07 for seed 2, and 1.07
for an arm whose mapping received no gradient at all. So the mapping was
never worse than no change. In the deterministic 12,000-step run its
pearson climbs 0.058 → 0.11 and its group-level change
(`map_delta_ratio_to_no_change`) falls to 0.80 against a floor of 0.46.
Phase 2 now reports `map_control_ratio_to_no_change` on the same cells.

### B1. What is achievable at all (CPU, under an hour)

Before improving the mapping, bound it. Two numbers, both from control cells
of one screen, both streamed through `ReplogleCells` so nothing large is held.

**B1a — the linear ceiling.** Ridge regression from panel to rest on control
cells, accumulating `XᵀX` (724×724) and `XᵀY` (724×7,216) in one pass, solved
for a few penalties, scored on held-out cells in the same control-SD units the
model uses. A linear map is a weak model; if it reaches pearson 0.3 where the
gene-token model reaches 0.06, the model is the problem and B0's branch tells
us which part. If ridge also lands at 0.08 and mse ≈ 1.0, the panel does not
predict single-cell rest genes at this depth, and no amount of architecture
will change that.

**B1b — the target's reliability.** Split each control cell's counts in two by
binomial thinning, put both halves in control-SD units, and correlate half A's
rest genes against half B's. Corrected for the halved depth by
Spearman–Brown, `r_full = 2r / (1 + r)`, that is an upper bound on the
correlation *any* predictor can reach against a single-cell target. At ~11,000
UMIs most of the 7,000 rest genes are at zero or one count, so this bound may
be near zero — in which case per-cell rest-gene prediction is not a solvable
problem and B2 is the answer rather than a fallback.

**Built.** `vccp/diagnostics/mapping.py` with `scripts/mapping_ceiling.py` as
its CLI, writing `reports/mapping_ceiling.json`: per screen, ridge pearson and
mse/no-change at each penalty, the library-size-only fit, the split-half
reliability raw and Spearman–Brown corrected, and the model's own
`map_control_pearson` read from `phase2/metrics.json` beside them. It picks
the branch itself — `model-underperforms`, `not-predictable-per-cell`,
`linear-is-no-better`, or a refusal — and exits non-zero on anything but the
first, which is the only one with a cheap fix.

```bash
python scripts/mapping_ceiling.py --config configs/server.yaml --run-name server
```

Run it from the checkout, with `--config` pointing at the config you use; it
writes into that run's `reports/`. Defaults are 20,000 control cells to fit on
and 2,048 held out, which is minutes per screen on CPU. Two things to check in
the output before reading the verdict: `best_at_grid_edge` must be false (a
ceiling chosen at the largest penalty offered is a statement about the grid,
and the verdict says so), and `n_eval_cells` should be the 2,048 asked for
rather than a fifth of a small screen.

**Run, and the answer is §B3** (DECISIONS.md D102). Pearson on held-out
control cells, with the ceiling `sqrt(reliability)`:

| screen | model | ridge | depth only | ceiling | model's share |
|---|---|---|---|---|---|
| K562_essential | 0.033 | 0.140 | 0.054 | 0.294 | **11%** |
| K562_gwps | 0.016 | 0.146 | 0.077 | 0.294 | **5%** |
| rpe1 | 0.061 | 0.188 | 0.065 | 0.317 | **19%** |

The per-cell quantity exists — 8.6–10% of the variance is reproducible signal
— so §B2's reframe is not forced. A ridge regression reaches about half the
ceiling; the gene-token model reaches 5–19%, less than library size alone on
two screens, and on two screens its error is *above* the no-change baseline on
data it trains on. Go to §B3, and start with the third bullet of §B0 rather
than with capacity: a model that has moved backwards from its zero-initialized
state is not short of parameters.

The run also corrected the instrument: the ridge came back above what the
diagnostic called the ceiling, because the ceiling on a predictor's
correlation is the square root of the reliability, not the reliability (D101).

### B2. The reframe, if B1 says the task is not solvable per cell

**None of the six metrics scores a single cell's rest genes.** Expression
accuracy and perturbation discrimination are computed on a group-sum
pseudobulk; the DE metrics run a Wilcoxon test of the target's cells against
the control cells, which is a statement about the group's distribution.
CLAUDE.md §6 says so explicitly. What the scoring does reward is realistic
cell-to-cell variation — sanity check 5, and the expression metric's
finite-cell noise correction — and that comes from resampling **real** control
cells, which §4.7's generator already does.

So the method may be solving a harder problem than it is scored on. The
reframe: predict one rest-gene fold-change vector per (context, target)
instead of one per cell, and let all cell-to-cell variation come from the
control resample.

It is cheap because the machinery exists. `generate_counts` already accepts a
`(n_genes,)` fold change as well as `(n_cells, n_genes)` — the pooled shape is
a supported input, not a new code path. The change is in what Phase 3's
prediction asks the model for: the pooled control profile in, one rest change
out, rather than a per-cell profile in and a per-cell rest profile out.

This **is** a deviation from §4.7 step 2 as written, and would be recorded as
one, with the B1 numbers as its justification. It is also the only branch in
this document with a direct line to the leaderboard: the group-level rest
change is exactly what `expr_mse_unbiased_capped_norm` and the DE metrics
consume.

### B3. Only if B1 says the model is underperforming a linear baseline

Then the question is capacity or optimisation, and the candidates in order of
cost:

1. **Loss weights** (`loss_mapping`, `loss_perturbation`, `loss_delta`) — one
   arm each, 45 minutes, and B0 may already have named the culprit.
2. **Latent width** — 64 latents carry every cell's panel through to 7,000
   gene queries. A Perceiver bottleneck that narrow is a deliberate choice for
   cost (§4.2 asks for a modest first build), and it is the obvious suspect
   for a mapping that cannot beat a per-gene mean. `model.n_latents: 128` is
   one config key.
3. **Output genes per step** — 2,048 of ~7,000 measured rest genes per step is
   an unbiased estimate of the loss but a noisy gradient for a per-gene
   decoder.

Each is one `--set`, each gets the seed count Part A prescribes, and none is
worth starting before B1.

---

## 4. Cost and order

| step | where | cost | blocks |
|---|---|---|---|
| A0 determinism | GPU | 16 min | everything |
| B0 read the curve | CPU | minutes | B1's interpretation |
| A1 seed spread | GPU | 2.2 h | every later comparison |
| B1 ceilings | CPU | <1 h + build | B2 vs B3 |
| B2 reframe *or* B3 arms | both | 1 day | the next submission |

A0 and B0 first and together — one is 16 GPU-minutes, the other needs no GPU
and reads a file that already exists. A1 can run while B1's script is being
written. That leaves roughly a week before 2026-10-15 for whichever of B2 or
B3 the ceilings choose, plus a full `all` and a submission.

If A0 fails, stop and fix determinism: without it A1 measures two things and
B3 cannot be read at all.

---

## 5. What this does not touch

* The submission path. `configs/server.yaml` still produces a validated
  `prediction.h5ad` that `vcc prep` accepts; the seed-2 file is the current
  best and should be uploaded independently of any of this.
* `perturbation_mode: pooled`, `warm_start: none`, `train_projections: false`,
  `checkpoint_selection: best` — all measured, all unchanged here.
* Phase 1 and the priors. Phase 1's weights are settled (D90, D91) and the
  LINCS prior blocks are used by every arm regardless.
* The rehearsal, the sanity checks, the validator and the generator. They are
  the instruments that found all of the above and none of them is in question.

## 6. How each step is recorded

One `DECISIONS.md` entry per measurement, whichever way it comes out,
including the ones that refute something already written — D86, D89, D95 and
D98 were all superseded in place rather than deleted, and that history is the
most useful part of the file. `PLAN_MAPPING.md` gets the result inline, as
PLAN_PERCELL.md's §7 and §11 did. A deviation from CLAUDE.md §4 goes in
`checklist.json` as well, so it appears in the run rather than only in prose.

---

## 7. Results after B0, and what B3 became (D105–D107)

| arm (seed 2) | steps | held-out pert | pert, training targets | map_delta ÷ no change |
|---|---|---|---|---|
| all objectives on (`noise_seed2`) | 6,000 | 0.7915 | 0.7447 | 0.830 |
| `loss_mapping=0 loss_delta=0` | 6,000 | **0.7397** | 0.7033 | 1.001 |
| all objectives on (`det_12k`) | 12,000 | 0.7435 (best, 10,500) | 0.6900 | 0.798 |

* The mapping objectives cost the perturbation module 0.052 (3.5 sd). The
  delta term is the only one teaching the group-level rest change.
* 12,000 steps are worth 0.048, flat from 7,500, with no collapse.
  `configs/server.yaml` is now 12,000.
* **Next arm:** `--set phase2.loss_mapping=0` alone (delta kept), 6,000 steps,
  seed 2. It asks whether the per-cell level loss is the whole cost. If
  `pert` lands near 0.74 and `map_delta` near 0.83, that loss pays for
  nothing the metrics score, and dropping it is a §4.5 deviation to record.

**Measured** (D108): `loss_mapping=0` alone gives 0.7721 held out, 0.019 better
than both on (1.3 sd, not a difference), with `map_delta` 0.846 against 0.830.
The level loss stays, because it is the only thing that anchors the
mapping's absolute level, and Phase 3's prediction reads that level directly.
The open item is to predict the rest change as
`map(predicted panel) − map(control panel)`, matching what the delta term
trains.
