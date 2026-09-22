# PLAN.md — build plan for the Virtual Cell Challenge 2026 prototype

Milestone **M0**. This file maps every item of the `CLAUDE.md` section 4.8 element
checklist to the module that implements it, the artifact it produces and the test that
covers it, and lists everything in the specification that is ambiguous, contradictory or
infeasible together with the resolution I propose.

Nothing but this file has been written yet. Sections 1–4 are the build plan; section 8
is the list of open questions I would like reviewed before M1.

---

## 1. What I verified in the repository first

Read in full: `CLAUDE.md`, `docs/metrics.md` (2,636 lines), `docs/vcc2026-metrics-brief.pdf`
(10 pages, text extracted locally — no `pdftoppm` in this environment, so it was
decompressed with `zlib` instead), `data_prep/phase_data.py`, and the headers of the other
`data_prep/` modules. `mini_data/` was inspected with a throwaway virtualenv
(`anndata`, `h5py`, `pandas`, `pyarrow`) outside the repository; **no repository file was
created, modified or committed except this one.**

What `mini_data/` actually contains — every number below is read from the data, and no
number below will appear as a literal in `vcc/`:

| | mini_data | full data (per CLAUDE.md §3.4) |
|---|---|---|
| challenge genes | 1,983 | 18,533 |
| gene split | 955 panel / 1,028 rest | ≤978 panel / rest |
| contexts | A, B, C (460 control cells, 46 `ntc_id`, median library 19,985) | A, B, C → D, E, F |
| `cells_per_pert` (manifest) | 50 | 400 |
| challenge targets | 30 | 300 |
| Phase 1 LINCS | 688 rows, 20 cell lines, 574 targets, 955 panel genes, `has_sig` 688/688, `has_gmt` 624/688 | larger |
| Phase 2 K562_gwps | 6,314 cells, 178 targets, 1,130 genes (716 panel / 414 rest) | ~2.0 M cells, ~9,900 targets |
| Phase 2 rpe1 | 4,592 cells, 151 targets, 1,244 genes (812 panel / 432 rest) | ~248 k cells, ~2,400 targets |
| targets shared K562_gwps ∩ rpe1 | **81** | many |
| challenge targets in K562_gwps / rpe1 / LINCS | 27 / **0** / 8 | 272 / **0** / — |
| challenge genes measured only in challenge controls | 541 (27 %) | ~10,000 (54 %) |
| median Replogle cells per target | 25 | hundreds |

Three consequences for the plan, all confirmed rather than assumed:

* **Rehearsal variant 2 (cross-context K562_gwps ↔ rpe1) is feasible on `mini_data`** —
  81 shared targets, enough to hold out a cohort and score it.
* **RPE1 contains none of the challenge targets on mini either**, exactly as on the server.
  Nothing in the pipeline may assume a challenge target is present in a given Replogle context.
* **Phase 1 is trainable on mini** (20 cell lines, `sig` everywhere, `gmt` on 91 % of rows),
  so the per-cell-line reporting §4.4 asks for is meaningful here.

Environment: this container starts with no scientific Python at all. `cell-eval2` **0.16.0**
is reachable on PyPI (the docs describe 0.15.0 — see §8.2), as is CPU `torch`; 30 GB of
writable disk is free. The `vcc` packaging tool is not on `PATH` here, as expected.

---

## 2. Repository layout to be created

```
vcc/                         the package — python -m vcc <stage> --config configs/<name>.yaml
  __main__.py                sets thread env vars via os.environ.setdefault BEFORE any heavy import
  cli.py                     stage registry, `all`, resume (skip finished stages) / --force
  config.py                  typed config load + defaults + resources block
  runtime.py                 device resolution, torch thread/memory caps, empty_cache between stages, seeds
  paths.py                   every path derived from config; no literal path anywhere else
  checklist.py               writes checklist.json from per-stage status records

  data/    reference.py manifest.py gene_split.py lincs.py replogle.py challenge.py
           coverage.py checks.py            <- check-data
  priors/  scope.py blocks.py coexpr.py lincs_blocks.py hgnc_block.py phenotype.py
           plm.py build.py checks.py
  models/  embedding.py tokens.py context.py perceiver.py decoder.py adapters.py
           heads.py losses.py core.py
  train/   loop.py replay.py l2sp.py freeze.py
  phases/  phase1.py phase2.py phase3.py forgetting.py
  rehearsal/ variants.py upper_floor.py calibrate.py report.py
  predict/ perturbed_panel.py panel_to_rest.py knockdown.py generator.py run.py
  eval/    official.py scale.py nochange.py
  sanity/  checks.py report.py
  submit/  writer.py validator.py package.py limits.py
  report/  summary.py figures.py

configs/ mini.yaml  server.yaml
scripts/ server_env.sh
tests/   (see §5)
README.md  DECISIONS.md  PLAN.md
```

`data_prep/` and `mini_data/` are never modified. `ReplogleCells` and `log1p_cp10k` are
imported from `data_prep.phase_data`, not reimplemented (`vcc/data/replogle.py` is a thin
adapter that adds masks, pseudobulk and `fold_expr`, and nothing else).

### Stage graph

`check-data → priors → phase1 → phase2 → rehearsal → phase3 → predict → sanity → validate → package → report`

`all` runs them in order. Each stage writes `<run>/.stages/<stage>.done` containing its
config hash; a stage is skipped when that file exists and the hash matches, unless
`--force`. Outputs all live under `output_root/<run_name>/`.

### Run directory

```
runs/<run_name>/
  config.yaml  seeds.json  log.txt  .stages/
  priors/      prior_features.npz  coverage.csv  checks.json
  checkpoints/ core_phase1.pt core_phase2.pt core_phase3.pt
  phase1/      metrics.json  curves.csv
  phase2/      metrics.json  curves.csv
  rehearsal/   report.json  summary.txt
  phase3_policy.json
  phase3/      adapt_<context>.json
  predictions/model/<context>/<target>.npz        (per-target blocks, chunked)
  submission/  prediction.h5ad  [prediction.vcc]
  sanity/      report.json  summary.txt
  reports/     frozen_check.json forgetting.json validate.json summary.md figures/*.png
  checklist.json
```

---

## 3. Element checklist → module → artifact → test

The table is the required mapping; §4 gives the design detail behind the rows that need it.

| # | Element | Module(s) | Artifact | Test |
|---|---|---|---|---|
| 1 | Gene vocabulary (4.1) | `priors/build.py`, `models/embedding.py` | `priors/prior_features.npz` | `test_priors_build.py::test_vocabulary_covers_every_challenge_gene_in_order`, `test_embedding.py::test_embedding_is_W_prior_plus_delta_and_delta_starts_zero` |
| 2 | Prior blocks (4.1) | `priors/{coexpr,lincs_blocks,hgnc_block,phenotype,plm,blocks,build}.py` | `priors/coverage.csv` | `test_priors_blocks.py::test_blocks_1_to_6_present_with_pca_size_and_missing_flags`, `::test_block7_absent_unless_configured_and_never_downloads` |
| 3 | Two gene roles (4.1) | `priors/build.py`, `models/embedding.py` | both role matrices in `priors/prior_features.npz` | `test_priors_build.py::test_feature_role_and_target_role_matrices_differ_and_are_both_stored` |
| 4 | Prior checks (4.1) | `priors/checks.py` | `priors/checks.json` | `test_priors_checks.py::test_complex_neighbour_and_response_correlation_checks_are_reported` |
| 5 | Shared gene-token model (4.2) | `models/*` | `checkpoints/core_phase{1,2,3}.pt` | `test_model_shapes.py::test_same_core_accepts_any_input_and_output_gene_subset`, `::test_three_checkpoints_share_core_parameter_names` |
| 6 | Frozen core + adapters (4.3) | `models/adapters.py`, `train/freeze.py` | `reports/frozen_check.json` | `test_freeze.py::test_frozen_parameters_are_bitwise_unchanged_after_a_training_step` |
| 7 | Forgetting guard (4.3) | `train/{replay,l2sp}.py`, `phases/forgetting.py` | `reports/forgetting.json` | `test_forgetting.py::test_phase1_validation_rescored_after_each_phase`, `test_l2sp.py::test_l2sp_and_replay_are_on_by_default_in_phases_2_and_3` |
| 8 | Phase 1 (4.4) | `phases/phase1.py`, `models/heads.py` | `phase1/metrics.json` | `test_phase1.py::test_one_training_step_and_heldout_target_split`, `::test_metrics_reported_per_cell_line` |
| 9 | Phase 2 (4.5) | `phases/phase2.py` | `phase2/metrics.json` | `test_phase2.py::test_siamese_shared_weights_on_control_and_perturbed_cells`, `::test_metrics_report_control_and_perturbed_separately`, `::test_loss_masked_to_measured_genes` |
| 10 | Rehearsal (4.6) | `rehearsal/*` | `rehearsal/report.json`, `phase3_policy.json` | `test_rehearsal.py::test_three_variants_with_upper_bound_and_floor`, `test_calibrate.py::test_policy_records_threshold_and_scale` |
| 11 | Phase 3 adaptation (4.7) | `phases/phase3.py` | `phase3/adapt_<context>.json` | `test_phase3.py::test_adapt_reports_loss_before_and_after_and_freezes_perturbation_module` |
| 12 | Phase 3 prediction (4.7) | `predict/*` | `predictions/model/` | `test_predict.py::test_cells_per_pert_distinct_integer_cells_per_target`, `test_knockdown.py::test_target_gene_reduced_by_fold_expr`, `test_generator.py::test_thinning_and_stochastic_rounding_preserve_integrality` |
| 13 | Submission + validation (8) | `submit/*` | `submission/prediction.h5ad`, `reports/validate.json`, `submission/prediction.vcc` (server) | `test_submission_writer.py::test_one_file_all_contexts_sparse_no_explicit_zeros`, `test_validator.py` (9 violation cases), `test_package.py::test_vcc_prep_skipped_cleanly_when_absent` |
| 14 | Sanity checks (6.1) | `sanity/*` | `sanity/report.json` | `test_sanity.py` — one test per check, each feeding a deliberately broken prediction |
| 15 | Results summary (5) | `report/*` | `reports/summary.md`, `reports/figures/` | `test_report.py::test_summary_collects_every_required_section_and_says_mini_is_not_indicative` |
| — | checklist itself | `checklist.py` | `checklist.json` | `test_all_pipeline.py::test_all_on_mini_produces_every_artifact_and_no_unreported_deviation` |

---

## 4. Design detail per element

### 4.1 Item 1–3 — vocabulary, prior blocks, two roles

`priors/blocks.py` defines one interface:

```python
class PriorBlock:
    name: str
    role: Literal["feature", "target"]
    def build(self, scope: PriorScope) -> tuple[np.ndarray, np.ndarray]:  # (n_genes, d), covered mask
```

`build.py` runs every enabled block, reduces each to `priors.block_dim` (default 32) by PCA
(truncated SVD for the sparse/multi-hot blocks), standardizes it, zero-fills uncovered
genes, appends one missing-indicator column per block, and concatenates — separately for the
feature role and the target role. `prior_features.npz` stores `feature_prior`,
`target_prior`, `gene_symbols`, per-block column offsets, and the per-block coverage masks;
`coverage.csv` is one row per block with its covered-gene count and fraction.

Blocks: (1) challenge-control co-expression, (2) Replogle-control co-expression — both via
**streaming randomized SVD over the cell × gene matrix**, never materializing a
gene × gene matrix (18,533² would be 1.4 GB and its dense PCA far more; `coexpr.py` reads
cells in blocks sized from `resources.max_ram_gb`); (3) LINCS `sig` co-variation,
(4) GMT co-membership — genes as columns of the Phase 1 layers; (5) HGNC `gene_group`
multi-hot → truncated SVD, accepting either `hgnc_complete_set.txt` or `hgnc_subset.txt`;
(6) perturbation phenotype, **target role only** — Replogle pseudobulk delta in control-SD
units plus LINCS `sig`; (7) protein-LM embeddings, built only when
`priors.plm_path` points at an existing file, never downloaded.

`models/embedding.py` implements `embedding(g) = W · prior(g) + δ(g)` with one shared `W`
per role, `δ` initialized to zero and L2-regularized (`priors.delta_l2`).

**Leakage** (`priors/scope.py`): a `PriorScope(exclude_targets, exclude_contexts,
controls_only_genes)` parameterizes every block; blocks 3, 4 and 6 are rebuilt under the
rehearsal's scope, blocks 1, 2, 5 are response-free and reused except that block 2 drops the
held-out context. Each scope hashes to a cache key so a rehearsal fold rebuilds only what
it must. `priors/checks.py` writes nearest-neighbour results for the complex families named
in §4.1 (ribosomal, proteasome, mitochondrial respiratory chain — membership read from HGNC
`gene_group`, not hardcoded gene lists) and the correlation between target-role embedding
similarity and Replogle knockdown-response similarity.

### 4.2 Items 5–7 — the shared model, adapters, forgetting guard

`models/core.py` assembles: value encoder MLP added to the gene embedding → Perceiver latent
bottleneck (`model.n_latents`, default 64) → gene-query decoder that scores any output gene
embedding against the latents, in chunks of `model.gene_chunk`. A perturbation token
(target-role embedding + learned knockdown-type embedding) and a context vector (pooled
control profile of that context) enter through FiLM. Every width, chunk and batch size is a
config key; a CUDA OOM is caught once and re-raised with the message
*"reduce `<key>` in your config"* — never retried larger.

`models/adapters.py` registers named adapters (`phase1`, `phase2`, `phase3`, and one per
context) as LoRA or bottleneck modules; `train/freeze.py` snapshots the SHA-256 of every
frozen tensor before a phase and re-checks it after, writing `reports/frozen_check.json`.

`phases/forgetting.py` re-scores the held-out Phase 1 validation split after Phase 1,
Phase 2 and Phase 3 using the Phase 1 adapters over the then-current core and `δ`, writing
`reports/forgetting.json`. Note that with the specification's default (core frozen after
Phase 1) the core contributes no drift, but `δ` is shared and does keep training, so the
measurement is real rather than trivially flat — see §8.9.

### 4.3 Items 8–9 — Phase 1 and Phase 2

**Phase 1.** Input = `layers['ctrl']` (panel, per-gene standardized) + perturbation token +
cell-line context. Two heads trained jointly: control + perturbation → `sig` and `gmt`
(the `gmt` loss masked where `has_gmt` is false), then control + signature → `pert`.
Validation split **by target gene**; `phase1/metrics.json` reports overall and per cell line.

**Phase 2.** The panel→rest map runs on single cells through `ReplogleCells`, with the
**same weights** on control and perturbed cells (Siamese), and the loss masked to the genes
that context measures (`genes.csv`). The Phase 1 perturbation module is adapted to Replogle
(control profile + perturbation token → perturbed panel, in control-SD units) on pseudobulk
and on sampled cells. Targets held out by gene. `phase2/metrics.json` reports control cells
and perturbed cells separately. Replay of Phase 1 batches and the L2-SP penalty are on by
default.

### 4.4 Item 10 — the rehearsal

`rehearsal/variants.py` runs all three variants on the Replogle contexts, with the held-out
target cohort capped by `rehearsal.max_targets` (default 100):

1. **Controls-only mapping** — fit the panel→rest map on the held-out context's controls
   only, feed the *true* perturbed panel, score the predicted rest genes; against an
   **upper bound** (map also trained on perturbed cells) and a **no-change floor**.
2. **Cross-context** — learn perturbation behaviour in one Replogle context, adapt to the
   other on its controls only, predict its perturbed cells for the shared targets
   (81 on mini, many more on the server). Full predicted cells, so scored with the six
   official metrics exactly as the challenge scores them: predictions pushed through the
   same Phase 3 cell generator into raw counts, truth = that context's real perturbed and
   real control cells in counts.
3. **Unseen genes** — a random gene set hidden from all training except the rehearsal
   context's controls, scored on its perturbed predictions.

Variants 1 and 3 predict only part of the genes: a full profile is assembled as *true panel
values + predicted rest* and pushed through the same generator so the official metrics can
run, and per-gene correlation and error on the genes involved are reported alongside
(see §8.6). The upper bound and the floor are reference arms only and are never submitted.

`rehearsal/calibrate.py` fits the two generator settings — the confidence threshold below
which a predicted change is zeroed, and the global effect-size scale — on variant 2 by grid
search (`rehearsal.calibration_grid`, default 4 × 4 = 16 scorings), and writes
`phase3_policy.json` with those two numbers plus the list of modules Phase 3 may adapt,
chosen from the variant results. `rehearsal/report.json` holds every variant, arm and
metric; `rehearsal/summary.txt` is the readable version.

### 4.5 Items 11–12 — Phase 3 and prediction

`phases/phase3.py` reads `phase3_policy.json` and trains only what it permits — context
adapters and `δ` for genes first seen here — on each context's challenge controls
(panel → rest, both halves observed), with the perturbation module frozen and verified
frozen. `phase3/adapt_<context>.json` records the loss before and after.

`predict/run.py`, per (context, target) from `targets.csv`:

1. control panel profile + perturbation token → predicted perturbed panel;
2. predicted panel → predicted rest through the adapted shared network;
3. `predict/knockdown.py` applies the target gene's own knockdown prior —
   per-target `fold_expr` from Replogle bulk when available, otherwise the median
   `fold_expr` (see §8.7) — whether the target is a panel or a rest gene;
4. `predict/generator.py` draws a **fresh independent** sample of that context's control
   cells (with replacement if needed — never one block reused across targets, which the
   expression metric's across-perturbation budget penalizes), applies the predicted per-gene
   fold changes **in count space** (binomial thinning down, scaling with stochastic rounding
   up), holds genes below the confidence threshold at a fold change of exactly 1, and emits
   `cells_per_pert` distinct integer-count cells. The generator is a swappable component
   (`predict.generator.name`) with the calibrated settings in the config.

Each target's block is written to `predictions/model/<context>/<target>.npz` as it is
produced and freed; nothing holds a cells × all-genes matrix for all targets at once.

### 4.6 Item 13 — submission, validation, packaging

`submit/writer.py` assembles **one** `.h5ad` for every context of the round by appending CSR
blocks to an on-disk sparse dataset (`indptr` int64, integer dtype for the data,
`eliminate_zeros()` on every block), with `obs` = exactly `target_gene` and `context`, no
control rows, and `var` index = `gene_names.csv` in order.

`submit/validator.py` checks every rule of §8 before `vcc prep` runs and writes
`reports/validate.json`. `submit/limits.py` holds the format constants
(`max_counts_per_cell`, `max_stored_entries`) as config defaults, not as module literals.

`submit/package.py` detects `vcc` with `shutil.which` (see §8.3 for the name-collision
guard): present → `validate` runs `vcc prep --dry-run` and `package` writes the `.vcc`;
absent → both skip with a clear message. On `mini_data` a `vcc prep` rejection is recorded
as a warning, never as a pass condition.

### 4.7 Item 14 — sanity checks

`sanity/checks.py` implements all seven checks of §6.1, each returning pass/warn/fail with
the measured value and its config-supplied threshold. Check 1 failing stops the pipeline
before the submission is written; other failures are printed prominently and make `validate`
refuse the submission unless `--allow-warnings`. Checks 6 and 7 read the rehearsal's official
metrics and the scorer's `nsig_counts` diagnostics. `sanity/report.json` and
`sanity/summary.txt`.

### 4.8 Item 15 — results summary

`report/summary.py` collects, without re-running anything: data sizes per source; the method
as built with checklist status and deviations; Phase 1 and Phase 2 held-out metrics and
curves; all three rehearsal variants against upper bound and floor, plus the six official
metrics for variant 2 on the 0–1 leaderboard scale where it can be built; forgetting across
the three phases; Phase 3 adaptation losses and the knockdown check; the sanity report; the
submission summary. `report/figures.py` draws each figure with matplotlib at projector-legible
sizes, one message per figure. When `data_root` is `mini_data` the summary says at the top
that the numbers are not indicative.

### 4.9 Evaluation wrapper

`vcc/eval/official.py` is a thin wrapper around `cell_eval2.compute_metrics` with
`replace(EvalConfig.from_preset("vcc2026"), pert_col="target_gene")` and an explicit
`de.backend` from config. It returns the six scored metrics plus
`de_wilcoxon_nsig_counts_real` / `_pred`, and logs the installed `cell_eval2` version into
every report that uses it. Metrics are never reimplemented. For scoring only, the wrapper
adds the real control cells to its **in-memory** copy if the installed scorer requires a
control group in the prediction object — never to a file that is submitted. `eval/scale.py`
builds the 0-end (`cell-eval2 baseline`) and 1-end (`run --anchor`) reference points on the
rehearsal context's real data where the data allow, and reports the raw six alone with an
explicit note where they cannot be built. `eval/nochange.py` holds the documented no-change
values (0.5 / 1.0 / ≈1.0 / ≈0 / ≈0 / ≈0) used by the sanity checks and the calibration
objective.

### 4.10 Resources and the shared server

`vcc/__main__.py` sets the eight thread variables with `os.environ.setdefault` **before**
importing numpy, torch, polars or scanpy, never overriding what the environment says, and
logs the effective values at the start of every run. `scripts/server_env.sh` contains
exactly the block from §1.2 plus `POLARS_MAX_THREADS` and `RAYON_NUM_THREADS`, both set to
`resources.cpu_threads`. Every pool size comes from `resources.cpu_threads` (default 4) or
`resources.num_workers` (default 2); no `n_jobs=-1` and no `os.cpu_count()` anywhere — a test
greps for both. GPU use is `cuda:0` only, capped with
`torch.cuda.set_per_process_memory_fraction(resources.gpu_memory_fraction)` (default 0.5),
with `torch.cuda.empty_cache()` between stages.

---

## 5. Test inventory

All tests run on `mini_data`, on CPU, and are fast. Beyond the per-element tests in §3:

* `test_config.py` — both configs load; `resources` defaults; unknown keys rejected.
* `test_check_data.py` — `check-data` passes on mini and names the missing file clearly
  when one is hidden (via a tmp copy of the tree; `mini_data/` is never touched).
* `test_data_alignment.py` — every loader returns genes in challenge order; gene orders
  agree across `gene_names.csv`, `gene_split.csv`, `phase3/genes.csv` and the control files.
* `test_no_hardcoded_sizes.py` — an AST scan of `vcc/` rejecting the numeric literals
  18533, 1983, 300, 30, 400, 50, 360000 and the string literals of context and Replogle
  context names, outside `configs/` and an allow-listed `submit/limits.py`.
* `test_validator.py` — nine cases, one per §7 violation: a control row, a construct id
  (`ADNP-1`), a missing target, a wrong cell count, a non-integer value, a negative value,
  a cell above 1,000,000 counts, explicit stored zeros, dense storage, wrong gene order.
* `test_official_scorer.py` — the wrapper returns all six metrics plus the two `nsig`
  diagnostics on a tiny prediction; skipped **with a clear reason** if `cell_eval2` cannot
  be imported.
* `test_sanity.py` — seven tests, each feeding a deliberately broken prediction (NaNs,
  wrong cell counts, target gene not reduced, all targets identical, all cells identical,
  worse than no change, no DE calls at all) and asserting the matching check catches it.
* `test_all_pipeline.py` — runs `python -m vcc all --config configs/mini.yaml` end to end
  and asserts every artifact of §3 exists and every checklist item is `done` or an
  explicitly reported `deviation`.

---

## 6. Milestone order

| | content | exit condition |
|---|---|---|
| M0 | this file | committed, reviewed |
| M1 | configs, `scripts/server_env.sh`, every loader of §3, `check-data`, CLI skeleton | `pytest` green; `python -m vcc check-data --config configs/mini.yaml` passes |
| M2 | prior blocks 1–6 (7 gated), both roles, embedding module, prior checks | items 1–4 artifacts exist |
| M3 | gene-token core, adapters, freeze check, forgetting guard, Phase 1, Phase 2 | items 5–9 artifacts exist |
| M4 | three rehearsal variants, calibration, `phase3_policy.json`, Phase 3, prediction, generator | items 10–12 artifacts exist |
| M5 | submission writer, validator, packaging, sanity, summary, `checklist.json` | `python -m vcc all --config configs/mini.yaml` green |
| M6 | README, DECISIONS.md, final message | definition of done in §9 of CLAUDE.md |

Budget targets: `all` on `mini_data` under 15 minutes on CPU; `all` on the server under
~12 hours on one GPU, with every budget a config key and the mini budgets set separately in
`configs/mini.yaml` (different values, **not** a code branch).

---

## 7. Known risks

* **Runtime on mini.** The rehearsal runs the official scorer several times (three variants
  × arms, plus the calibration grid). Wilcoxon DE over ~2,000 genes × ~100 targets is the
  slow part. Mitigation: `rehearsal.max_targets` (default 100), a small default calibration
  grid, and the scorer's own result cache. If 15 minutes is threatened, the *mini config's*
  grid shrinks — never the code path.
* **Replogle depth on mini.** 25 cells per target means noisy DE; sanity checks 6 and 7 may
  warn on mini. §6.1 allows this explicitly; the checks still run and report.
* **Thin LINCS overlap with challenge targets** (8/30 on mini, and the target-role prior for
  the other 22 rests on Replogle and co-expression alone). This is the real data's shape too.
* **`cell-eval2` API drift** — see §8.2.

---

## 8. Ambiguities, contradictions and infeasibilities — with proposed resolutions

Each item states what I found, why it matters, and what I propose to do. Items marked
**[decision wanted]** are the ones where I would most like your view before M1; the rest I
will implement as proposed and record in `DECISIONS.md` unless you say otherwise.

### 8.1 The brief and `CLAUDE.md` §8 disagree about the submission **[decision wanted]**

`docs/vcc2026-metrics-brief.pdf` chapter 0 says a submission's *"perturbation labels match
the reference exactly, **including the non-targeting control label**"* and that *"the number
of predicted cells per perturbation is **unconstrained**"*. `CLAUDE.md` §8 says the opposite
on both points: **no control rows at all**, and **exactly `cells_per_pert` cells** per
(context, target).

`CLAUDE.md` §8 declares itself the highest authority and instructs me to record the
difference, so **I will follow §8**: no control rows, exactly `cells_per_pert` cells, and the
scorer gets the real controls added to an in-memory copy only. The risk if §8 is wrong is
that the real `vcc prep` rejects a submission with no control group. I will make the
validator's control-row rule and the cell-count rule two config switches
(`submit.include_controls: false`, `submit.enforce_cells_per_pert: true`) so that flipping
them is a config change and not a rewrite, and I will report whatever `vcc prep --dry-run`
says about it on the server. **Please confirm that the challenge's own submission guideline
(which §8 restates) is newer than this brief.**

### 8.2 `cell-eval2` version and API cannot be verified from the docs **[decision wanted]**

`docs/metrics.md` and the brief describe **0.15.0** with `rule_version 3`; PyPI publishes
only **0.16.0**. The docs themselves record that metric semantics have moved *within* a
version more than once (#172, #271, #343, #348, #351) and that `rule_version` was bumped
twice in two releases. So the six metrics' exact values, and possibly the
`EvalConfig`/`compute_metrics` signatures that `CLAUDE.md` §6 names, may differ from what
the docs describe.

Proposed: pin `cell-eval2==0.16.0` in `environment.yml` (the only published version), keep
`vcc/eval/official.py` a *thin* wrapper that introspects the installed `EvalConfig` rather
than assuming attribute names, log `cell_eval2.__version__` and the resolved config into
every report that uses it, and let the scorer tests skip with a clear reason if the import
or the preset is unavailable. I will not reimplement or patch any metric. If the installed
0.16.0 turns out to lack the `vcc2026` preset, that becomes a reported deviation rather than
a local reimplementation.

### 8.3 `python -m vcc` collides with the challenge's `vcc` tool **[decision wanted]**

The package must be importable as `vcc` (§7 CLI) and the challenge's packaging tool is also
called `vcc` (§8). On the server, if that tool installs a Python package named `vcc`, then
`python -m vcc` run from outside the repository resolves to *theirs*, and running from inside
the repository shadows *theirs* for any Python that imports it.

Proposed: keep the package name `vcc` as specified, and add two guards — at startup, assert
`vcc.__file__` resolves under the repository root and fail with an explicit message if not;
and never invoke the tool as a module, only as the executable found by `shutil.which("vcc")`,
after confirming it is not a shim for our own package. The README will tell the user to run
`all` from the repository root. If you would rather rename the package (e.g. `vccp`) to
remove the collision entirely, that is a one-line change now and a painful one later.

### 8.4 "Maximize the mean of the six official metrics" is not well defined as written

§4.7 calibrates the generator by *"maximizing the mean of the six official metrics"*, but two
of the six are lower-is-better and the six live on different scales, so their raw mean is not
a meaningful objective (an `lfc_nmae` of 0.9 and a `sig_jaccard` of 0.9 are not comparable
quantities).

Proposed: define the objective as the mean of the six **normalized against their documented
no-change points and directions** — `(0.5 - u)/0.5` style per member, i.e. each metric mapped
so that no-change is 0 and better is positive — and, where `eval/scale.py` can build the
baseline and replicate anchors on the rehearsal data, use the leaderboard's own
0 = baseline / 1 = replicate scaling instead. The objective actually used is written into
`phase3_policy.json` and `DECISIONS.md`.

### 8.5 Sanity checks 2 and 5 are under-specified

Check 2's parenthetical (*"allowing for the knocked-down gene and genes the model predicts to
move"*) makes it unfalsifiable as written, and check 5 does not say in which space the
variance is measured.

Proposed: check 2 is evaluated only on genes the model did **not** predict to move (predicted
|log2 FC| below the generator's confidence threshold) and excludes the target gene; it
requires each such gene's predicted mean to fall inside a bootstrap range of that context's
control per-gene means, and warns above a configurable violating fraction. Check 5 is
computed on `log1p(CP10K)` per gene, taking the median over genes, against the same statistic
on that context's control cells. Both thresholds live in the config with the §6.1 defaults.

### 8.6 Rehearsal variants 1 and 3 mix truth and prediction in one profile

§4.6 says to score variants 1 and 3 "with the same metrics where a full profile can be
assembled (true panel values plus predicted rest)". The official metrics need raw counts,
and a profile that is true on the panel and predicted on the rest will score well on the
panel for reasons that have nothing to do with the model.

Proposed: assemble the fold changes (true on the panel, predicted on the rest), push them
through the **same** Phase 3 generator against resampled real control cells to get raw
counts, and score those — so the number is comparable to variant 2 — but **always report the
panel-only and rest-only per-gene correlation and error beside it**, and say in the summary
that the panel half is truth. The upper bound and the floor arms go through the identical
path, which is what makes the comparison meaningful.

### 8.7 Which `fold_expr` applies to a challenge context

§4.7's default knockdown prior is `ctrl × median fold_expr (Replogle)`, but the medians
differ by screen (0.155 K562, 0.088 RPE1) and the challenge contexts A/B/C are not identified
with any Replogle cell line.

Proposed: config key `phase3.knockdown.source` with default `pooled` — the median over every
Replogle bulk file present — overridden by the per-target `fold_expr` whenever the target
appears in Replogle bulk (a gene may have several promoter rows; take the median over them).
The resolved value per target is logged and appears in the Phase 3 knockdown figure. §4.7
notes this prior does not move the score, so a wrong choice is cheap; it is still recorded.

### 8.8 `device` appears twice with different values

§1's `configs/mini.yaml` snippet has top-level `device: auto`, while §1.2 says to add a
`resources:` block "with these keys and defaults (`device: cpu` for `mini.yaml`)" — and
`device` is not among the `resources` keys §1.2 lists.

Proposed: one key only. `device` stays at the top level exactly as the §1 snippet shows, and
`mini.yaml` keeps `auto`, which resolves to CPU in this container (no CUDA) and so satisfies
both readings. `resources` carries `cpu_threads`, `num_workers`, `gpu_memory_fraction` and
`max_ram_gb` only. Tell me if you want an explicit `device: cpu` in `mini.yaml` instead.

### 8.9 The forgetting guard is inert under the specification's own default

§4.3 says later phases train adapters and `δ` by default and the core is unfrozen *only if
the config says so*. With the core frozen, most of what `reports/forgetting.json` measures
cannot move, and the replay and L2-SP machinery §4.8 item 7 requires "on by default in Phases
2–3" has almost nothing to act on.

It is not entirely inert — `δ` is shared across phases and does keep training, so Phase 1
validation can still drift, and that is what the report will show. Proposed: keep the
specified default (core frozen), apply replay and L2-SP to the parameters that *do* train,
state plainly in `reports/forgetting.json` and the summary which parameters were trainable in
each phase, and expose `train.unfreeze_core` so the guard can be exercised. If you would
rather the default unfreeze the core in Phase 2 — which is where the capacity gain would be —
say so and I will flip the default rather than the code.

### 8.10 `prediction.vcc` cannot exist in the cloud, but item 13 lists it

Checklist item 13's artifact list includes `submission/prediction.vcc`, annotated "(server)".
The §9 definition of done requires every item to be `done` or an explained `deviation`.

Proposed: `checklist.json` records item 13 as `done` with
`"vcc_prep": "skipped — tool not on PATH"` when the tool is genuinely absent and every other
part of the item passed, and the `all`-on-mini test accepts that status for this one
sub-artifact only. This is a reporting convention, not a deviation from the method; it is
recorded in `DECISIONS.md` either way.

### 8.11 Smaller points I will resolve without asking

* **`pert_counts.csv` has only a `target_gene` column in `mini_data`.** The real file may
  carry a count column. The loader reads `target_gene` and uses any per-target count column
  if present, falling back to the manifest's `cells_per_pert`.
* **Gene-gene co-expression PCA does not scale.** Blocks 1 and 2 use streaming randomized
  SVD over the cell × gene matrix rather than a materialized gene × gene correlation matrix.
* **The `nowhere-measured` genes** (541 on mini, ~10,000 on the server) are supervised only
  by the challenge controls at Phase 3; their perturbation responses are never supervised
  anywhere. That is the method as specified, not a gap I will paper over, and the summary
  reports how many genes are in that position.
* **`ReplogleCells` reads `meta.json["source"]`, an absolute server path**, and falls back
  to `source_rel`. On mini the absolute path does not exist and the fallback is used. I will
  rely on the existing fallback and not modify `data_prep/`.
* **`resources.max_ram_gb` drives chunk sizes** for the co-expression blocks, the submission
  writer and the prediction loop; each derives its block size rather than hardcoding one.
* **`check-data` hard-stops `all`** before any training unless `--skip-check` is passed, so
  the server run fails in seconds rather than hours when a file is wrong.
