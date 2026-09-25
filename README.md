# Virtual Cell Challenge 2026 — prototype

One shared gene-token model, updated by three phases, predicting the effect of
a CRISPRi knockdown on cells of an unseen context. `CLAUDE.md` is the
specification; `PLAN.md` maps each of its elements to the module that
implements it; `DECISIONS.md` records every non-obvious choice.

The deliverable of a run is **`submission/prediction.h5ad`** — one file for
every context of the round — together with a rehearsal report, a sanity
report and a results summary with figures.

```
vccp/
  data/      loaders for every file of CLAUDE.md §3, and `check-data`
  priors/    the gene vocabulary: prior blocks, both gene roles, the checks
  models/    the shared gene-token core, LoRA adapters, checkpoints
  train/     training loop, freeze bookkeeping, L2-SP
  phases/    Phase 1 (LINCS), Phase 2 (Replogle), Phase 3 (challenge), forgetting
  rehearsal/ Phase 3's procedure where the answers are known, and the calibration
  predict/   the knockdown prior and the count-space cell generator
  eval/      the official scorer, the no-change scale, the leaderboard scale
  sanity/    the seven checks of §6.1
  submit/    the one-file writer, the §8 validator, `vcc prep`
  report/    `summary.md` and its figures
```

---

## On the server, from a clean checkout

### 1. Environment

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate vcc
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install lightning torchmetrics
```

`environment.yml` already pins `cell-eval2==0.16.0`, the official scorer.
Install its CPU base only — **not** the `[gpu]` or `[gpudge]` extras, which
pull their own `torch` and can break the model's environment (CLAUDE.md §1).
The DE backend the config pins, `pdex`, comes with it.

The challenge's own packaging tool is separate and you install it yourself:

```bash
pip install vcc        # whatever the challenge's instructions say
vcc --version
```

Everything works without it; `validate` then skips `vcc prep --dry-run` and
no `.vcc` is written.

### 2. Every session: the thread and GPU limits

```bash
source scripts/server_env.sh
```

This is the block from CLAUDE.md §1.2 plus `POLARS_MAX_THREADS` and
`RAYON_NUM_THREADS`, which `cell-eval2` needs or polars starts one thread per
core. The pipeline **never overrides** these: whatever the environment says,
it respects, and it logs the effective values at the start of every run. Keep
`POLARS_MAX_THREADS` equal to `resources.cpu_threads` in your config.

**One knob worth knowing about.** That block sets `NUMBA_NUM_THREADS=1`, and
`cell-eval2` runs its Wilcoxon test through numba — so the scorer, and with it
the whole rehearsal, is single-threaded however high `resources.cpu_threads`
is. The pipeline caps itself at the smallest pool rather than raising one
(§1.2 says never override the environment), so this is correct but slow. If
the rehearsal is taking too long, raising it to match `cpu_threads` is your
call and still a good neighbour on a 192-core machine:

```bash
export NUMBA_NUM_THREADS=4      # same as resources.cpu_threads
```

`python scripts/check_env.py` prints the pools and says when one of them is
capping the scorer.

`CUDA_VISIBLE_DEVICES=1` in that script chooses the GPU. The code only ever
addresses `cuda:0`, which is whatever that variable makes visible, and caps
itself at `resources.gpu_memory_fraction` (0.5) of it.

### 3. Check the machine before the first run

```bash
python scripts/check_env.py --config configs/server.yaml
```

Seconds, writes nothing, and tells you whether the packages are importable,
whether `cell-eval2` and its `vcc2026` preset resolve, whether the challenge's
`vcc` is on `PATH`, which device the config resolves to, whether the thread
limits are in force, whether the data roots exist and whether there is disk
space under `output_root`. It exits non-zero only on something that would
stop a run; missing `vcc` is a warning.

### 4. Check the data

```bash
python -m vccp check-data --config configs/server.yaml
```

Every file of §3 present, every expected column there, every gene order
agreeing, and the sizes it found — cells, genes and targets per source —
printed for you to confirm before hours of training. Read those numbers: they
are the first place a wrong `data_root` or a half-copied directory shows up.
It writes `reports/check_data.json`.

If a stage ever dies with an HDF5 error — `Can't synchronously read data
(filter returned failure during read)`, or anything about a filter — one of
the input files is truncated, damaged in transit, or written with a
compression this environment cannot decode. `check-data` reads blocks of every
matrix and will normally catch it first; to find the file directly:

```bash
python scripts/probe_data.py --config configs/server.yaml
```

It reads blocks of each `X` and each layer of every `.h5ad` the pipeline
opens and names the file, the element and the row range that failed. Add
`--full` to read every file end to end (slow on the single-cell files), or
`--only <substring>` to probe one. `checks.probe_matrices: false` turns the
check-data probe off if you would rather not pay for it every run.

### 5. Run everything

```bash
nice -n 10 ionice -c3 python -m vccp all --config configs/server.yaml
```

`nice`/`ionice` keep other people's interactive work ahead of this one. `all`
runs every stage in order and is **resumable**: a stage that finished under
the same config is skipped, so an interrupted run continues where it stopped.
`--force` reruns anyway.

### 5b. Choosing `perturbation_mode`, once

The perturbation module can be supervised two ways, and which is better is a
measurement rather than a judgement (DECISIONS.md D59):

* **`pooled`** — one control pseudobulk in, the target's pseudobulk out. One
  mean effect per (context, target).
* **`percell`** — a *cell* in, that cell perturbed out, with the truth built
  by optimal transport between a drawn control batch and a drawn perturbed
  batch. This is what CLAUDE.md §4.2 asks for.

`pooled` is the default because it is the arm with a leaderboard number
behind it. To decide, run the A/B **once**, on a copy of a finished run so
the priors and the Phase 1 core are already there:

```bash
# Is the coupling informative on this data at all? Seconds, and it gates
# everything below: a degenerate coupling makes per-cell training identical
# to pooled training, and the two look the same from their curves.
python scripts/coupling_check.py --config configs/server.yaml

# The A/B itself. One Phase 2 arm per configuration, so budget for
# len(mode_sweep_epsilons) x rehearsal.phase2_steps on top of a normal
# rehearsal.
python -m vccp rehearsal --config configs/server_sweep.yaml --force
```

`configs/server_sweep.yaml` is `server.yaml` with a `rehearsal.mode_sweep`
block and nothing else changed, so the only thing that differs between rows
is the configuration being measured. It writes to `runs/server_sweep/`, so
it leaves your main run alone.

Then read the **Mode sweep** table at the top of
`runs/<run>/rehearsal/summary.txt`:

| what to look for | what it means |
|---|---|
| the `floor` row | predicting no change. An arm below it has not earned a submission, however it ranks against the others |
| `overall` | the mean of the six official metrics on the leaderboard's 0–1 scale — the reading the leaderboard agreed with, where the calibration objective did not |
| every row failing identically | a property of the dataset, not of the configurations; the table says so and names the scorer's reason |

If a `percell` row wins, set `phase2.perturbation_mode: percell` and its
`ot_epsilon` in `server.yaml` and rerun `all`. If `pooled` wins, change
nothing. Either way the losing mode stays in the codebase as the control.

**`percell` costs about 2.3x the wall clock** (measured on `mini_data`: 706 s
against 1653 s), since it pushes `cells_per_pert` rows through the encoder
per target instead of one. Lower `predict.cells_per_forward` first if the GPU
runs out.

### 6. Check the submission with the challenge's own tool, then package it

Our own `validate` stage has already checked every rule of §8 and, if `vcc` is
on `PATH`, has run `vcc prep --dry-run` for you. To run it yourself:

```bash
vcc prep runs/server/submission/prediction.h5ad \
    -g /data/share/han/VCC/gene_names.csv \
    --perts /data/share/han/VCC/pert_counts.csv \
    --dry-run

vcc prep runs/server/submission/prediction.h5ad \
    -g /data/share/han/VCC/gene_names.csv \
    --perts /data/share/han/VCC/pert_counts.csv \
    -o runs/server/submission/prediction.vcc
```

(`python -m vccp package --config configs/server.yaml` does exactly the second
command when the tool is installed.) Upload the `.vcc`.

---

## What to read before you submit

In `runs/<run_name>/`:

**`sanity/summary.txt`** — the seven checks of §6.1, each `pass`, `warn` or
`fail` with the value measured and the threshold it was measured against. A
**fail** of check 1 stops the pipeline before a submission is written; the
others do not stop the run but `validate` refuses the submission unless you
pass `--allow-warnings`. What each warning means:

| check | a warning here means |
|---|---|
| 2 plausible magnitudes | library sizes or per-gene means are off — usually a units bug |
| 3 the knockdown is visible | target genes are not going down; check `phase3/knockdown_<ctx>.csv` |
| 4 targets differ | the model is predicting nearly the same profile for everything |
| 5 cells vary | the generated cells are too alike, or too scattered, to be real cells |
| 6 better than doing nothing | **the prediction is not beating a no-change submission** |
| 7 reasonable DE calls | the prediction is too timid for the DE metrics to score, or floods them |

Checks 6 and 7 are the ones to take seriously: they are measured on the
rehearsal, where the truth is known, and a warning there says the submission
is unlikely to score.

**`rehearsal/summary.txt`** — Phase 3's own procedure, run on Replogle data
whose answers were hidden. Three variants, each against an **upper bound**
(the same model that *did* see the held-out perturbed cells) and a
**no-change floor** (predict that nothing happened):

* `cross_context` is the one that matters. It produces whole predicted cells
  and is scored with the six official metrics exactly as the challenge scores
  them. Its "objective vs no change" line should be **positive** and above the
  floor's. Where `cell-eval2` could build the two ends, a leaderboard-scale
  line follows (0 = the organizers' mean-response baseline, 1 = a split-half
  replicate of the real data).
* `controls_only` and `unseen_genes` are labelled **PANEL-ASSISTED**: they are
  fed the true panel values, so their official metrics are not end-to-end
  performance and must not be read as such.

**`phase2/metrics.json`** — two readings that say whether the model is
learning at all, before any of the above is worth looking at:

* `map_delta_ratio_to_no_change`, the error on the **change** the mapping
  predicts, against predicting no change. The per-state MSEs beside it are
  dominated by the baseline expression level; this is the only quantity the
  six official metrics score. Read it against
  `map_delta_noise_floor_ratio`, which is what a *perfect* predictor could
  reach at these cell counts — at the floor means everything the cells can
  show has been learned, at 1.0 means nothing about the change has been.
* `phase1_contribution`, a per-metric `scratch − warm` difference between the
  `core_scratch` arm (no Phase 1 at all) and the warm-started one. Positive
  means Phase 1 helped. The two arms do not spend equal effort on Phase 2's
  own objective — `n_phase2_steps` is reported with it, and the difference
  favours `core_scratch`, so a small positive number is a stronger result
  than it looks. Check `arms_share_initialization` first: `false` means the
  arms started from different weights and the table cannot be read.
  **One run of this is one draw.** Three server runs put `core_scratch`
  0.12 apart on a quantity whose whole effect is 0.18, so repeat it before
  it changes anything — three runs differing only in the seed:

  ```bash
  for seed in 0 1 2; do
    nice -n 10 ionice -c3 python -m vccp phase2 \
      --config configs/server.yaml --run-name server_seed$seed --seed $seed
  done
  ```

  Any single config value can be overridden the same way, so a sweep needs no
  second config file: `--set train.lr=0.0001 --set phase2.steps=3000`,
  repeatable, written into the run's `config.yaml` like any other setting.

  Each run needs `priors/` and `checkpoints/` in its run directory (copy
  them from a finished run, as for any stage run on its own). Average the
  per-metric `improvement` across the three; the spread between them is the
  error bar. Three such runs put the gap at −0.091 ± 0.045 read at each arm's
  best and +0.023 ± 0.022 read at the last step, i.e. **not measurable** —
  see DECISIONS.md D86 before spending a cycle on it again.
* `signal_retained`, per arm: the share of what the arm learned that its last
  step still held. 1.0 ended at its best, 0.0 ended knowing no more than the
  no-change baseline, **negative ended worse than that**. Below 0.5 the arm is
  flagged and the summary tables it out. This is the reading
  `divergence_final_over_best` cannot give: on a ratio against no change an
  arm that gave back everything it learned still reads as 1.22x.
* `weights_kept`, per arm: `best` or `final`. `train.checkpoint_selection`
  defaults to `best`, so a phase saves the weights it measured as best rather
  than whatever the last step produced — the checkpoint is what the next phase
  inherits (D85). Set it to `final` to go back to the old behaviour.

`phase2.eval_targets` (default 256, `0` = all) sets how many held-out targets
`pert_mse_ratio_to_no_change` is scored over. It was `batch_size` — 32 of the
1,907 K562_gwps offers — which made the metric noisier than the effects being
read off it (D88). Raise it for any run whose purpose is to choose between two
configurations; it costs forward passes at evaluation time and nothing else.

`train.evals_per_run` (default 4) sets how often a phase evaluates, and so the
resolution at which all three of those readings exist: `checkpoint_selection`
can only pick an evaluated step, and `signal_retained` can only be measured
across them. Four over 6,000 steps showed an arm at 0.82 and then at 1.00 with
nothing in between. Each evaluation costs a full validation pass, so raise it
when diagnosing (`configs/server_isolate.yaml` uses 8) and leave it at 4 for
the submission run.

**`reports/coupling_check.json`**, when `scripts/coupling_check.py` has been
run — whether the OT pairing carries information on this data at all. Read
`target_spread` (how much the per-cell targets vary; zero means the per-cell
loss is the pooled loss under another name) against `residual_reduction`
(whether pairing brings each target nearer its control cell; above 1 means
the pairing adds more sampling noise than it removes). The two pull opposite
ways, which is why `phase2.ot_epsilon` is swept.

**`reports/summary.md`** — everything above as tables and figures, for the
proposal defense. `reports/figures/*.png` are drawn to be legible on a
projector.

**`checklist.json`** — every element of CLAUDE.md §4.8 as `done`, a
`deviation` with its reason, or `missing`.

---

## Keeping track of what you submitted

Every run stamps the code that produced it — the git commit, whether the tree
was dirty and which files differed, and the versions of torch, numpy, scipy,
anndata, cell-eval2 and pdex. It goes into `config.yaml`, `checklist.json`
and the top of `reports/summary.md`, and the commit is logged on the first
line of every run. Any `.vcc` can therefore be traced to the code that made
it.

Two habits make that worth something:

* **one run name per submission** — `--run-name server_v1`, `server_v2`, … so
  a new run never overwrites a previous version's artifacts;
* **record three columns per upload**: run name, commit, leaderboard score.

The settings that decide a prediction live in `phase3_policy.json` (the
generator's threshold and scale) and `predictions/model/index.json` (what the
blocks were actually built with). Those two, plus the commit, are the whole
provenance of a submission.

Runs are also **reproducible**: `train.deterministic` (on by default) turns
off TF32, asks torch for deterministic kernels and fixes the cuBLAS
workspace, so the same checkpoint and seeds give the same predictions. And a
stage whose inputs have been rewritten since it ran is no longer considered
finished — rerunning `predict` un-finishes `sanity` and `validate`, so a
`.vcc` can never quietly describe predictions that no longer exist.

## Where the outputs go

Everything is under `output_root/<run_name>/` and nothing is written anywhere
else:

```
config.yaml                    the exact config this run used, plus its hash
log.txt                        every line the run printed
.stages/                       which stages have finished (this is the resume)
reports/check_data.json        the input check
priors/                        prior_features.npz  coverage.csv  checks.json
checkpoints/                   core_phase1.pt  core_phase2.pt  core_phase3.pt
phase1/  phase2/               metrics.json  curves.csv
rehearsal/                     report.json  summary.txt  scale/
phase3_policy.json             what Phase 3 may adapt, and the generator settings
phase3/                        adapt_<context>.json  knockdown_<context>.csv
predictions/model/             index.json, then <context>/<target>.npz per target
reports/forgetting.json        Phase 1's validation under each phase's core
reports/frozen_check.json      the frozen parameters, verified unchanged
sanity/                        report.json  summary.txt
submission/prediction.h5ad     the deliverable (+ prediction.vcc where `vcc` exists)
reports/validate.json          every rule of §8, checked against that file
reports/summary.md             the results summary
reports/figures/*.png          its figures
checklist.json                 CLAUDE.md §4.8, item by item
```

---

## Diagnostics you can run on their own

Neither trains anything that the pipeline keeps, and neither writes a
checkpoint. Both take the same `--config` as everything else.

```bash
# Can the perturbation module fit a handful of targets it may memorize?
# Exits non-zero unless the verdict is can-fit.
python scripts/capacity_check.py --config configs/server.yaml --targets 8 --steps 2000

# Does the OT coupling carry information on real cells, or do distances
# concentrate until every pairing is uniform? Exits non-zero on a verdict of
# degenerate or pairs-on-noise.
python scripts/coupling_check.py --config configs/server.yaml
```

There is also a fourth **rehearsal variant**, off by default, that measures
the one boundary the other three never test:

```bash
python -m vccp rehearsal --config configs/server_xassay.yaml --force
```

Variants 1–3 all stay inside Replogle. `cross_assay` learns what a
perturbation does from **LINCS only** — bulk, CRISPR knockout — and meets a
Replogle screen through its control cells alone, which is structurally what
every submission does: the challenge is a different lab, protocol and
sequencing depth from Replogle. It retrains nothing, so it costs an
adaptation, a prior rebuild and three scored arms per screen.

Read its two method arms against each other. They are the **same weights**
asked through different perturbation-type rows — `method` as CRISPRi, which
is what Phase 3 does, and `method_source_modality` as CRISPR knockout, the
modality the weights learned. The gap is what Phase 1's knockout-specific
offset is worth on a knockdown screen, which is the question CLAUDE.md §3.3
raises and nothing else in the pipeline answers. If they score the same, the
type vocabulary is doing nothing there.

Run the coupling check before trusting `phase2.perturbation_mode: percell`
on a new dataset: a degenerate coupling makes per-cell training identical to
pooled training, and the two are indistinguishable from their curves.

---

## Running stages separately

```bash
python -m vccp <stage> --config configs/server.yaml
```

in this order — each reads what the ones before it wrote:

| stage | what it does | writes |
|---|---|---|
| `check-data` | verify every input file, column, gene order — and that the matrices decode | `reports/check_data.json` |
| `priors` | the gene vocabulary and its checks | `priors/` |
| `phase1` | LINCS: control + perturbation → signature → perturbed | `phase1/`, `checkpoints/core_phase1.pt` |
| `phase2` | Replogle: Siamese panel→rest, and the perturbation module | `phase2/`, `checkpoints/core_phase2.pt` |
| `rehearsal` | the three variants, and the generator calibration | `rehearsal/`, `phase3_policy.json` |
| `phase3` | adapt on each challenge context's controls | `phase3/`, `checkpoints/core_phase3.pt` |
| `forgetting` | re-score Phase 1's validation under every core | `reports/forgetting.json` |
| `predict` | every (context, target): panel, rest, knockdown, cells | `predictions/model/` |
| `sanity` | the seven checks, on the predictions | `sanity/` |
| `validate` | assemble the submission, then check it against §8 | `submission/`, `reports/validate.json` |
| `package` | `vcc prep`, where the tool exists | `submission/prediction.vcc` |
| `report` | the results summary, the figures, the checklist | `reports/`, `checklist.json` |

Useful flags: `--run-name <name>` writes to a different run directory (so two
configurations can sit side by side), `--force` reruns a finished stage,
`--allow-warnings` accepts a submission the sanity checks warned about, `-v`
turns on debug logging.

There is no `submit` stage: §6.1 requires a sanity failure to stop the
pipeline *before* the submission is written, so `validate` assembles the file
and then validates what it wrote.

---

## Expected runtimes

**Measured, on `mini_data`, CPU only** (this repository, 4 threads):

| | |
|---|---|
| `pytest` | ~4 min |
| `python -m vccp all --config configs/mini.yaml` | ~9 min |

**Estimated, on the server, one GPU** — these are estimates from the mini
timings and the configured budgets, not measurements, because the full data
was never available to this build. Read the timestamps in `log.txt` for what
your machine actually does.

| stage | what drives it | rough |
|---|---|---|
| `check-data` | reading headers only | minutes |
| `priors` | streaming control cells for the co-expression blocks | 20–60 min |
| `phase1` | `phase1.steps` (400) | under an hour |
| `phase2` | `phase2.steps` (600), twice if the core-freeze ablation is on | 1–3 h |
| `rehearsal` | retrains Phase 2 per variant, then ~**43 scoring calls** | 2–5 h |
| `phase3` | `phase3.adapt_steps` (300) per context | under an hour |
| `predict` | 300 targets × 3 contexts × 400 cells | 10–30 min |
| `sanity` | reads every block once | 10–30 min |
| `validate` | writes ~2×10⁹ stored entries block by block | 20–60 min |
| `report` | figures | minutes |

The design target is **about 12 hours on one GPU** (CLAUDE.md §0). If it looks
like overrunning, the knobs in order of effect are:

1. `rehearsal.max_targets` (100) — every scoring call runs a Wilcoxon DE over
   this many targets;
2. `rehearsal.calibration_thresholds` × `calibration_scales` — 16 points is 16
   scoring calls;
3. `eval.build_scale: false` — saves six more scoring calls per dataset;
4. `train.core_freeze_ablation: false` — saves one whole Phase 2 arm;
5. `phase1.steps` / `phase2.steps`.

If the GPU runs out of memory, lower `train.output_genes_per_step`, then
`model.gene_chunk`, then the batch sizes. The failure message names the key.

---

## Switching to the final test round (from 2026-10-22)

The round is data, not code. New contexts (D, E, F) and a different
perturbation list need **only a config change**:

```yaml
# configs/server_final.yaml
data_root: /data/han/projects/VCC/data      # the processed phases, rebuilt for the round
vcc_root: /data/share/han/VCC_final         # the new release: manifest.json,
                                            # gene_names.csv, pert_counts.csv,
                                            # context_D.h5ad, context_E.h5ad, ...
output_root: /data/han/projects/VCC/runs
run_name: final
```

Then:

```bash
python scripts/check_env.py --config configs/server_final.yaml
python -m vccp check-data --config configs/server_final.yaml
nice -n 10 ionice -c3 python -m vccp all --config configs/server_final.yaml
```

Nothing in the code names a context, a target or a count: contexts come from
`manifest.json` and the control files, perturbations from `pert_counts.csv`,
cells per perturbation from the manifest (or from `pert_counts.csv` where it
carries a column), and the gene axis from `gene_names.csv`. A test fails if
any module hardcodes one of them.

Two things to make sure of before that run:

* `processed/phases/` must be rebuilt for the new round by the `data_prep/`
  scripts — `gene_split.csv`, `phase3/genes.csv`, `phase3/targets.csv` and
  `phase3/controls_<context>.h5ad` all describe the round;
* `ref/challenge_perts.txt` must match the new `pert_counts.csv`;
  `check-data` fails loudly if it does not.

---

## Development

```bash
pytest                                        # the whole suite, on mini_data
pytest tests/test_validator.py -q             # one file
python -m vccp all --config configs/mini.yaml # the whole pipeline, on CPU
```

`mini_data/` is **read-only input**: a session-scoped fixture fails the test
run if anything writes to it. Tests that need a broken input file work on a
tmp directory of symlinks.

The mini data deliberately has a different gene count, perturbation list and
cell count from the real round, so anything sized by hand fails there.
