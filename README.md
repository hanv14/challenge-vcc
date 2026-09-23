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

**`reports/summary.md`** — everything above as tables and figures, for the
proposal defense. `reports/figures/*.png` are drawn to be legible on a
projector.

**`checklist.json`** — every element of CLAUDE.md §4.8 as `done`, a
`deviation` with its reason, or `missing`.

---

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
