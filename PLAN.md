# PLAN.md — build plan for the Virtual Cell Challenge 2026 prototype

This file maps every item of the `CLAUDE.md` section 4.8 element checklist to the module
that implements it, the artifact it produces and the test that covers it. It is kept
current as the build proceeds — §6 marks which milestones are done.

Section 8 is the list of specification questions raised at M0; **all eleven were resolved by
the user on review**, and each now records the decision taken. Section 9 lists the four
points where those decisions depart from `CLAUDE.md` as written — these are also recorded in
`DECISIONS.md` and go into the final message.

**Status: M0, M1 and M2 complete.** Sections 1–7 are the build plan; module names in §2 that do
not exist yet are what later milestones will add.

---

## 1. What I verified in the repository first

Read in full: `CLAUDE.md`, `docs/metrics.md` (2,636 lines), `docs/vcc2026-metrics-brief.pdf`
(10 pages, text extracted locally — no `pdftoppm` in this environment, so it was
decompressed with `zlib` instead), `data_prep/phase_data.py`, and the headers of the other
`data_prep/` modules. `mini_data/` was inspected with a throwaway virtualenv
(`anndata`, `h5py`, `pandas`, `pyarrow`) outside the repository; **no repository file was
created or modified except this one.**

What `mini_data/` actually contains — every number below is read from the data, and no
number below will appear as a literal in `vccp/`:

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

The package is named **`vccp`**, not `vcc`, to avoid colliding with the challenge's own
`vcc` command-line tool (§8.3). Every command is therefore
`python -m vccp <stage> --config configs/<name>.yaml`.

```
vccp/                        the package
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
imported from `data_prep.phase_data`, not reimplemented (`vccp/data/replogle.py` is a thin
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
  checkpoints/ core_phase1.pt core_phase2.pt core_phase2_frozen_core.pt core_phase3.pt
  phase1/      metrics.json  curves.csv
  phase2/      metrics.json  curves.csv
  rehearsal/   report.json  summary.txt
  phase3_policy.json
  phase3/      adapt_<context>.json  knockdown_<context>.csv
  predictions/model/<context>/<target>.npz        (per-target blocks, chunked)
  submission/  prediction.h5ad                    (the deliverable; the user packages it)
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
| 6 | Frozen core + adapters (4.3) | `models/adapters.py`, `train/freeze.py` | `reports/frozen_check.json` | `test_freeze.py::test_declared_frozen_parameters_are_bitwise_unchanged_after_a_training_step` |
| 7 | Forgetting guard (4.3) | `train/{replay,l2sp}.py`, `phases/forgetting.py` | `reports/forgetting.json` | `test_forgetting.py::test_phase1_validation_rescored_after_each_phase`, `::test_core_freeze_ablation_reports_both_arms`, `test_l2sp.py::test_l2sp_and_replay_are_on_by_default_in_phases_2_and_3` |
| 8 | Phase 1 (4.4) | `phases/phase1.py`, `models/heads.py` | `phase1/metrics.json` | `test_phase1.py::test_one_training_step_and_heldout_target_split`, `::test_metrics_reported_per_cell_line` |
| 9 | Phase 2 (4.5) | `phases/phase2.py` | `phase2/metrics.json` | `test_phase2.py::test_siamese_shared_weights_on_control_and_perturbed_cells`, `::test_metrics_report_control_and_perturbed_separately`, `::test_loss_masked_to_measured_genes` |
| 10 | Rehearsal (4.6) | `rehearsal/*` | `rehearsal/report.json`, `phase3_policy.json` | `test_rehearsal.py::test_three_variants_with_upper_bound_and_floor`, `::test_arms_share_control_resample_and_generator_seed`, `::test_panel_assisted_metrics_are_keyed_separately`, `test_calibrate.py::test_policy_records_threshold_and_scale` |
| 11 | Phase 3 adaptation (4.7) | `phases/phase3.py` | `phase3/adapt_<context>.json` | `test_phase3.py::test_adapt_reports_loss_before_and_after_and_freezes_perturbation_module` |
| 12 | Phase 3 prediction (4.7) | `predict/*` | `predictions/model/` | `test_predict.py::test_cells_per_pert_distinct_integer_cells_per_target`, `test_knockdown.py::test_target_gene_reduced_by_fold_expr`, `::test_prior_skipped_below_min_control_cpm`, `::test_resolved_source_recorded_per_target`, `test_generator.py::test_thinning_and_stochastic_rounding_preserve_integrality` |
| 13 | Submission + validation (8) | `submit/*` | `submission/prediction.h5ad`, `reports/validate.json` (`.vcc` out of scope — §8.10) | `test_submission_writer.py::test_one_file_all_contexts_sparse_no_explicit_zeros`, `test_validator.py` (10 violation cases), `test_package.py::test_vcc_prep_skipped_cleanly_when_absent` |
| 14 | Sanity checks (6.1) | `sanity/*` | `sanity/report.json` | `test_sanity.py` — one test per check, each feeding a deliberately broken prediction; plus `::test_check2_bounds_moved_genes_by_max_abs_log2fc` |
| 15 | Results summary (5) | `report/*` | `reports/summary.md`, `reports/figures/` | `test_report.py::test_summary_collects_every_required_section_and_says_mini_is_not_indicative` |
| — | checklist itself | `checklist.py` | `checklist.json` | `test_all_pipeline.py::test_all_on_mini_produces_every_artifact_and_no_unreported_deviation` |

---

## 4. Design detail per element

### 4.1 Items 1–3 — vocabulary, prior blocks, two roles

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
units plus LINCS `sig`; (7) protein-LM embeddings, built only when `priors.plm_path` points
at an existing file, never downloaded.

The knockdown-phenotype blocks carry **direction and size separately**: the response is
z-scored per target for the direction decomposition, and the magnitude comes back as its own
feature alongside the source's quality signals (Replogle `fold_expr`,
`anderson_darling_counts`, cell count; LINCS `cc_q75_median`, number of signatures). Without
that, a near-null response — which most knockdowns are — normalizes into a confident-looking
random direction. The direction basis is learned from the targets that did respond
(`priors.phenotype_weight_by_magnitude`) and every target is then projected into it;
`priors/checks.json` reports the fraction of targets below
`priors.phenotype_magnitude_floor` and, next to it, how far magnitude merely tracks
sequencing depth.

`models/embedding.py` implements `embedding(g) = W · prior(g) + δ(g)` with one shared `W`
per role, `δ` initialized to zero and L2-regularized (`model.delta_l2`).

**The layout travels with the artifact.** The block set is not fixed — the server's third
Replogle screen adds two blocks — so `coverage.csv` and `prior_features.npz` both record
every block's name, width, column labels, column offsets per role, coverage and the seed its
cell sampling used, plus a `layout_hash` that a checkpoint is matched against.

**Leakage** (`priors/scope.py`): a `PriorScope(exclude_targets, exclude_contexts,
controls_only_genes)` parameterizes every block; blocks 3, 4 and 6 are rebuilt under the
rehearsal's scope, blocks 1, 2, 5 are response-free and reused except that block 2 drops the
held-out context. Each scope hashes to a cache key so a rehearsal fold rebuilds only what it
must. `priors/leakage.py` states, per rehearsal variant and per Replogle context, which blocks are
**rebuilt**, **dropped** or **reused** under that variant's scope, and writes it to
`priors/checks.json`; `tests/test_leakage.py` rebuilds under each scope and asserts the plan
matches what really happens. Variant 2's asymmetry is the case that matters: the held-out
screen's *responses* are dropped while its *control* co-expression is reused, because
"adapt using only its controls" is the variant.

`priors/checks.py` writes nearest-neighbour results for the complex families named in
§4.1 (ribosomal, proteasome, mitochondrial respiratory chain — membership read from HGNC
`gene_group`, not hardcoded gene lists), the correlation between target-role embedding
similarity and Replogle knockdown-response similarity, and the magnitude report above.

### 4.2 Items 5–7 — the shared model, adapters, forgetting guard

`models/core.py` assembles: value encoder MLP added to the gene embedding → Perceiver latent
bottleneck (`model.n_latents`, default 64) → gene-query decoder that scores any output gene
embedding against the latents, in chunks of `model.gene_chunk`. A perturbation token
(target-role embedding + learned knockdown-type embedding) and a context vector (pooled
control profile of that context) enter through FiLM. Every width, chunk and batch size is a
config key; a CUDA OOM is caught once and re-raised with the message
*"reduce `<key>` in your config"* — never retried larger.

`models/adapters.py` registers named adapters (`phase1`, `phase2`, `phase3`, and one per
context) as LoRA or bottleneck modules. `train/freeze.py` snapshots the SHA-256 of every
tensor **declared frozen for that phase** before the phase and re-checks it after, writing
`reports/frozen_check.json`; the declaration itself comes from the config, so the check
stays meaningful under either core-freezing policy.

**Core-freezing policy** (decided at review, §8.9): `train.unfreeze_core` is
**`true` for Phase 2** and **`false` for Phase 3**, both configurable. Phase 2 has the data
to support training the core; Phase 3 sees only control cells, where updating the core would
risk exactly the perturbation knowledge Phases 1–2 built. Replay and L2-SP apply to whatever
trains in a phase, and are on by default in Phases 2 and 3.

`phases/forgetting.py` re-scores the held-out Phase 1 validation split after Phase 1,
Phase 2 and Phase 3, using the Phase 1 adapters over the then-current core and `δ`, and
states in `reports/forgetting.json` which parameters were trainable in each phase.

**Core-freeze ablation.** Because the Phase 2 default is a judgement call, Phase 2 trains a
second arm with the core frozen (`train.core_freeze_ablation`, default on;
`train.ablation_steps_fraction` lets the server shrink it). Phase 1 validation is scored
under both arms, and the comparison is reported in `reports/forgetting.json` and surfaced in
`rehearsal/report.json` under `core_freeze_ablation` and in the summary — so the choice of
default rests on a measurement rather than on an assumption. Both checkpoints are kept
(`core_phase2.pt`, `core_phase2_frozen_core.pt`); only the configured arm continues into the
rehearsal and Phase 3.

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
and perturbed cells separately.

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

**Arm comparability** (decided at review, §8.6/1): within a variant, the method, the upper
bound and the floor use the **same resampled control cells and the same generator settings**,
with the seed fixed per `(variant, target)` in `seeds.json`, so the arms differ only in the
predicted fold changes. A test asserts the three arms draw identical control-cell indices.

**Panel-assisted scoring** (decided at review, §8.6/2): variants 1 and 3 predict only part of
the genes. A full profile is assembled as *true panel values + predicted rest*, pushed
through the same generator, and scored with the official metrics — but those values are
written under the key **`official_panel_assisted`**, never under variant 2's
`official` key, and the summary labels them as panel-assisted so they cannot later be read as
end-to-end performance. Per-gene correlation and error are reported separately for the panel
half and the rest half. The upper bound and floor arms go through the identical path.

`rehearsal/calibrate.py` fits the two generator settings — the confidence threshold below
which a predicted change is zeroed, and the global effect-size scale — on variant 2 by grid
search (`rehearsal.calibration_grid`, default 4 × 4 = 16 scorings), maximizing the objective
defined in §8.4, and writes `phase3_policy.json` with those two numbers, the objective used,
and the list of modules Phase 3 may adapt. `rehearsal/report.json` holds every variant, arm
and metric; `rehearsal/summary.txt` is the readable version.

### 4.5 Items 11–12 — Phase 3 and prediction

`phases/phase3.py` reads `phase3_policy.json` and trains only what it permits — context
adapters and `δ` for genes first seen here — on each context's challenge controls
(panel → rest, both halves observed), with the core and the perturbation module frozen and
verified frozen. `phase3/adapt_<context>.json` records the loss before and after.

`predict/run.py`, per (context, target) from `targets.csv`:

1. control panel profile + perturbation token → predicted perturbed panel;
2. predicted panel → predicted rest through the adapted shared network;
3. `predict/knockdown.py` applies the target gene's own knockdown prior — per-target
   `fold_expr` from Replogle bulk when available (median over that gene's promoter rows),
   otherwise the pooled median over every Replogle bulk file present
   (`phase3.knockdown.source`, default `pooled`). Two rules decided at review (§8.7):
   * the **resolved source is recorded per target** (`pooled` or `per_target:<screen>`) in
     `phase3/knockdown_<context>.csv` and shown in the knockdown figure, because a
     `fold_expr` measured in K562 or RPE1 need not transfer to an unidentified context;
   * the prior is applied **only where the target gene is detectably expressed** in that
     context's controls (mean CP10K above `phase3.knockdown.min_control_cpm`, defaulting to
     the same floor as sanity check 3); below that floor the gene is left unchanged.
4. `predict/generator.py` draws a **fresh independent** sample of that context's control
   cells (with replacement if needed — never one block reused across targets, which the
   expression metric's across-perturbation budget penalizes), applies the predicted per-gene
   fold changes **in count space** (binomial thinning down, scaling with stochastic rounding
   up), holds genes below the confidence threshold at a fold change of exactly 1, and emits
   `cells_per_pert` distinct integer-count cells. The generator is a swappable component
   (`predict.generator.name`) with the calibrated settings in the config.

Each target's block is written to `predictions/model/<context>/<target>.npz` as it is
produced and freed; nothing holds a cells × all-genes matrix for all targets at once.

### 4.6 Item 13 — submission and validation

`submit/writer.py` assembles **one** `.h5ad` for every context of the round by appending CSR
blocks to an on-disk sparse dataset (`indptr` int64, integer dtype for the data,
`eliminate_zeros()` on every block), with `obs` = exactly `target_gene` and `context`, no
control rows, and `var` index = `gene_names.csv` in order — the format of `CLAUDE.md` §8.

`submit/validator.py` checks every rule of §8 before anything else and writes
`reports/validate.json`. `submit/limits.py` holds the format constants
(`max_counts_per_cell`, `max_stored_entries`) as config defaults, not as module literals.

**The user packages the submission themselves** (§8.1, §8.10). `submit/package.py` is still
built — it detects `vcc` with `shutil.which` and runs `vcc prep --dry-run` / writes the
`.vcc` when the tool is present — but `submission/prediction.h5ad` is the deliverable, and
the `.vcc` is not required for the run to be complete. The README gives the user the exact
`vcc prep` command to run on the server.

### 4.7 Item 14 — sanity checks

`sanity/checks.py` implements all seven checks of §6.1, each returning pass/warn/fail with
the measured value and its config-supplied threshold. Check 1 failing stops the pipeline
before the submission is written; other failures are printed prominently and make `validate`
refuse the submission unless `--allow-warnings`. Checks 6 and 7 read the rehearsal's official
metrics and the scorer's `nsig_counts` diagnostics.

**Check 2 has two halves** (decided at review, §8.5), because restricting it to the
unchanged genes would leave the moved ones unbounded:

* genes the model did **not** predict to move (predicted |log2 FC| below the generator's
  confidence threshold), excluding the target gene, must have a predicted mean inside a
  bootstrap range of that context's control per-gene means;
* genes the model **did** predict to move, excluding the target gene, must satisfy
  |log2 FC vs control| ≤ `sanity.max_abs_log2fc` (default 10).

Both halves report the violating fraction and the worst offenders by name.
**Check 5** is computed on `log1p(CP10K)` per gene, taking the median over genes, against the
same statistic on that context's control cells.

### 4.8 Item 15 — results summary

`report/summary.py` collects, without re-running anything: data sizes per source; the method
as built with checklist status and deviations; Phase 1 and Phase 2 held-out metrics and
curves; the core-freeze ablation; all three rehearsal variants against upper bound and floor
(with variants 1 and 3 labelled panel-assisted); the six official metrics for variant 2 on
the 0–1 leaderboard scale where it can be built; forgetting across the three phases; Phase 3
adaptation losses and the knockdown check, annotated with each target's `fold_expr` source;
the sanity report; the submission summary. `report/figures.py` draws each figure with
matplotlib at projector-legible sizes, one message per figure. When `data_root` is
`mini_data` the summary says at the top that the numbers are not indicative.

### 4.9 Evaluation wrapper

`vccp/eval/official.py` is a thin wrapper around `cell_eval2.compute_metrics` with
`replace(EvalConfig.from_preset("vcc2026"), pert_col="target_gene")` and an explicit
`de.backend` from config. It **introspects the installed `EvalConfig`** rather than assuming
attribute names (§8.2), returns the six scored metrics plus
`de_wilcoxon_nsig_counts_real` / `_pred`, and logs `cell_eval2.__version__` and the resolved
config into every report that uses it. Metrics are never reimplemented. For scoring only,
the wrapper adds the real control cells to its **in-memory** copy if the installed scorer
requires a control group in the prediction object — never to a file that is submitted.
`eval/scale.py` builds the 0-end (`cell-eval2 baseline`) and 1-end (`run --anchor`) reference
points on the rehearsal context's real data where the data allow, and reports the raw six
alone with an explicit note where they cannot be built. `eval/nochange.py` holds the
documented no-change values (0.5 / 1.0 / ≈1.0 / ≈0 / ≈0 / ≈0) used by the sanity checks and
the calibration objective.

### 4.10 Resources and the shared server

`vccp/__main__.py` sets the eight thread variables with `os.environ.setdefault` **before**
importing numpy, torch, polars or scanpy, never overriding what the environment says, and
logs the effective values at the start of every run. `scripts/server_env.sh` contains exactly
the block from §1.2 plus `POLARS_MAX_THREADS` and `RAYON_NUM_THREADS`, both set to
`resources.cpu_threads`. Every pool size comes from `resources.cpu_threads` (default 4) or
`resources.num_workers` (default 2); no `n_jobs=-1` and no `os.cpu_count()` anywhere — a test
greps for both. GPU use is `cuda:0` only, capped with
`torch.cuda.set_per_process_memory_fraction(resources.gpu_memory_fraction)` (default 0.5),
with `torch.cuda.empty_cache()` between stages.

`device` is a single top-level key: **`cpu` in `mini.yaml`** (explicit, not left to `auto`)
and **`auto` in `server.yaml`**. `resources` carries `cpu_threads`, `num_workers`,
`gpu_memory_fraction` and `max_ram_gb`.

---

## 5. Test inventory

All tests run on `mini_data`, on CPU, and are fast. Beyond the per-element tests in §3:

* `test_config.py` — both configs load; `resources` defaults; `device` is `cpu` on mini and
  `auto` on server; unknown keys rejected.
* `test_check_data.py` — `check-data` passes on mini; names the missing file clearly when one
  is hidden (via a tmp copy of the tree; `mini_data/` is never touched); and **fails with
  both numbers reported** when a per-target count column in `pert_counts.csv` disagrees with
  the manifest's `cells_per_pert` (§8.11).
* `test_skip_check.py` — `--skip-check` warns prominently and records
  `checks_skipped: true` in the run config; no documented command or script passes it.
* `test_data_alignment.py` — every loader returns genes in challenge order; gene orders agree
  across `gene_names.csv`, `gene_split.csv`, `phase3/genes.csv` and the control files.
* `test_no_hardcoded_sizes.py` — an AST scan of `vccp/` rejecting the numeric literals
  18533, 1983, 300, 30, 400, 50, 360000 and the string literals of context and Replogle
  context names, outside `configs/` and an allow-listed `submit/limits.py`.
* `test_validator.py` — ten cases, one per §7/§8 violation: a control row, a construct id
  (`ADNP-1`), a missing target, a wrong cell count, a non-integer value, a negative value, a
  non-finite value, a cell above 1,000,000 counts, explicit stored zeros, dense storage,
  wrong gene order.
* `test_official_scorer.py` — the wrapper returns all six metrics plus the two `nsig`
  diagnostics on a tiny prediction; skipped **with a clear reason** if `cell_eval2` cannot be
  imported.
* `test_sanity.py` — one test per check, each feeding a deliberately broken prediction (NaNs,
  wrong cell counts, target gene not reduced, all targets identical, all cells identical,
  worse than no change, no DE calls at all), plus a prediction with an absurd fold change to
  exercise the `max_abs_log2fc` half of check 2.
* `test_all_pipeline.py` — runs `python -m vccp all --config configs/mini.yaml` end to end
  and asserts every artifact of §3 exists and every checklist item is `done` or an explicitly
  reported `deviation`.

---

## 6. Milestone order

| | content | exit condition |
|---|---|---|
| M0 | this file | committed, reviewed ✅ |
| M1 | configs, `scripts/server_env.sh`, every loader of §3, `check-data`, CLI skeleton | `pytest` green; `python -m vccp check-data --config configs/mini.yaml` passes ✅ |
| M2 | prior blocks 1–6 (7 gated), both roles, embedding module, prior checks | items 1–4 artifacts exist ✅ |
| M3 | gene-token core, adapters, freeze check, forgetting guard + core-freeze ablation, Phase 1, Phase 2 | items 5–9 artifacts exist |
| M4 | three rehearsal variants, calibration, `phase3_policy.json`, Phase 3, prediction, generator | items 10–12 artifacts exist |
| M5 | submission writer, validator, sanity, summary, `checklist.json` | `python -m vccp all --config configs/mini.yaml` green |
| M6 | README, DECISIONS.md, final message | definition of done in §9 of CLAUDE.md |

Budget targets: `all` on `mini_data` under 15 minutes on CPU; `all` on the server under
~12 hours on one GPU, with every budget a config key and the mini budgets set separately in
`configs/mini.yaml` (different values, **not** a code branch).

---

## 7. Known risks

* **Runtime on mini.** The rehearsal runs the official scorer several times (three variants ×
  arms, plus the calibration grid), and Phase 2 now trains a second ablation arm. Wilcoxon DE
  over ~2,000 genes × ~100 targets is the slow part. Mitigation: `rehearsal.max_targets`
  (default 100), a small default calibration grid,
  `train.ablation_steps_fraction`, and the scorer's own result cache. If 15 minutes is
  threatened, the *mini config's* budgets shrink — never the code path.
* **Replogle depth on mini.** 25 cells per target means noisy DE; sanity checks 6 and 7 may
  warn on mini. §6.1 allows this explicitly; the checks still run and report.
* **Thin LINCS overlap with challenge targets** (8/30 on mini, and the target-role prior for
  the other 22 rests on Replogle and co-expression alone). This is the real data's shape too.
* **`cell-eval2` API drift** — see §8.2.

---

## 8. Specification questions raised at M0 — and the decisions taken

All eleven were answered by the user on review. Each entry states the question and the
decision now in force.

### 8.1 The brief and `CLAUDE.md` §8 disagree about the submission — **resolved**

`docs/vcc2026-metrics-brief.pdf` chapter 0 says a submission's perturbation labels match the
reference *"including the non-targeting control label"* and that the number of predicted
cells per perturbation is *"unconstrained"*. `CLAUDE.md` §8 says the opposite on both: no
control rows, exactly `cells_per_pert` cells.

**Decision.** Write `submission/prediction.h5ad` in the format of the submission guideline
that §8 restates — no control rows, exactly `cells_per_pert` cells, one file for all
contexts, sparse, `gene_names.csv` order. **The user packages the submission themselves.**
Both rules stay behind config switches (`submit.include_controls: false`,
`submit.enforce_cells_per_pert: true`) so a change of guideline is a config change.

### 8.2 `cell-eval2` version and API cannot be verified from the docs — **resolved**

The docs describe 0.15.0 / `rule_version 3`; PyPI publishes only 0.16.0, and the docs record
metric semantics moving *within* a version more than once.

**Decision (accepted as proposed).** Pin `cell-eval2==0.16.0` in `environment.yml`; keep the
wrapper thin and introspective rather than assuming attribute names; log the installed
version and resolved config into every report that uses it; let the scorer tests skip with a
clear reason if the import or the preset is unavailable. Never reimplement or patch a metric.
If 0.16.0 lacks the `vcc2026` preset, that is a reported deviation, not a local
reimplementation.

### 8.3 `python -m vcc` collides with the challenge's `vcc` tool — **resolved**

**Decision.** The package is named **`vccp`**. Every command is `python -m vccp <stage>`.
The challenge's `vcc` tool is invoked only as the executable found by `shutil.which("vcc")`.
This departs from `CLAUDE.md` §2 and §7 — see §9.

### 8.4 "Maximize the mean of the six official metrics" is not well defined — **resolved**

Two of the six are lower-is-better and the six live on different scales, so their raw mean is
not a meaningful objective.

**Decision (accepted as proposed).** The calibration objective is the mean of the six
**normalized against their documented no-change points and directions**, so that no-change is
0 and better is positive; where `eval/scale.py` can build the baseline and replicate anchors
on the rehearsal data, the leaderboard's own 0 = baseline / 1 = replicate scaling is used
instead. The objective actually used is written into `phase3_policy.json` and `DECISIONS.md`.

### 8.5 Sanity checks 2 and 5 are under-specified — **resolved, with an addition**

**Decision.** As proposed, **plus**: check 2 must also bound the genes the model *does*
predict to move — |log2 FC vs control| ≤ `sanity.max_abs_log2fc` (default 10), excluding the
target gene — reporting the violating fraction and the worst offenders. Rationale from the
user: the genes below the confidence threshold are copies of control cells by construction,
so restricting the check to them would leave the moved genes unbounded. Check 5 is computed
on `log1p(CP10K)` per gene, median over genes. Implementation in §4.7.

### 8.6 Rehearsal variants 1 and 3 mix truth and prediction — **resolved, with two requirements**

**Decision.** As proposed, **plus**: (1) within a variant, the method, the upper bound and
the floor must use the same resampled control cells and the same generator settings, with the
seed fixed per `(variant, target)`, so the arms differ only in the predicted fold changes;
(2) these official-metric values are keyed **`official_panel_assisted`** in
`rehearsal/report.json` and labelled as such in the summary, never reported under variant 2's
key. Implementation in §4.4.

### 8.7 Which `fold_expr` applies to a challenge context — **resolved, with two additions**

**Decision.** As proposed (`phase3.knockdown.source`, default `pooled`, overridden by a
per-target value where the target is in Replogle bulk), **plus**: (1) record the resolved
source per target (`pooled`, `per_target:<screen>`) in the Phase 3 output and show it in the
knockdown figure, since a `fold_expr` measured in K562 or RPE1 need not transfer to an
unidentified context; (2) apply the prior only where the target gene is detectably expressed
in that context's controls (mean CP10K above `phase3.knockdown.min_control_cpm`, default the
same floor as sanity check 3) — below it, leave the gene unchanged. Implementation in §4.5.

### 8.8 `device` appears twice with different values — **resolved**

**Decision.** One top-level `device` key; `resources` as listed in §1.2.
`mini.yaml` sets **`device: cpu` explicitly** rather than relying on `auto` resolving to CPU
in the container; `server.yaml` keeps `device: auto`. This departs from the §1 `mini.yaml`
snippet — see §9.

### 8.9 The forgetting guard is inert under the specification's own default — **resolved: default flipped**

**Decision.** `train.unfreeze_core` is **`true` for Phase 2** and **`false` for Phase 3**,
both configurable. The user's rationale: Phase 2 has enough data to support training the core
and is where the forgetting guard is worth having; Phase 3 sees only control cells, where
updating the core risks exactly the perturbation knowledge Phases 1–2 built. Replay and L2-SP
apply to whatever trains; `reports/forgetting.json` states which parameters were trainable in
each phase. **The rehearsal reports Phase 1 validation after Phase 2 with the core unfrozen
versus frozen**, so the choice is backed by a measurement rather than by judgement.
Implementation in §4.2. This departs from `CLAUDE.md` §4.3's stated default — see §9.

### 8.10 `prediction.vcc` cannot exist in the cloud — **resolved: out of scope**

**Decision.** The deliverable is `submission/prediction.h5ad` in the challenge's format; the
user runs `vcc prep` themselves. `submit/package.py` is still built and still runs when the
tool is on `PATH`, but the `.vcc` is not required for the run to be complete, and
`checklist.json` records item 13 as `done` with
`"vcc_prep": "not run — the user packages the submission"`. This departs from checklist item
13 as written — see §9.

### 8.11 Smaller points — **all accepted, with two adjustments**

* **`pert_counts.csv` per-target count column.** If it carries one and **any** value
  disagrees with the manifest's `cells_per_pert`, `check-data` **fails**, reporting both
  numbers — it never silently prefers one.
* **`--skip-check`** prints a prominent warning, records `checks_skipped: true` in the run
  config, and is never the default in any documented command or script.
* Gene-gene co-expression uses streaming randomized SVD, never a materialized gene × gene
  matrix.
* The `nowhere-measured` genes (541 on mini, ~10,000 on the server) are supervised only by
  the challenge controls at Phase 3; the summary reports how many genes are in that position.
* `ReplogleCells` reads `meta.json["source"]` (an absolute server path) and falls back to
  `source_rel`; on mini the fallback is used. `data_prep/` is not modified.
* `resources.max_ram_gb` drives chunk sizes for the co-expression blocks, the submission
  writer and the prediction loop.
* `check-data` hard-stops `all` before any training unless `--skip-check` is passed.

---

## 9. Deviations from `CLAUDE.md`, authorized at the M0 review

These four go into `DECISIONS.md` and into the final message, per §0 of `CLAUDE.md`.

| # | `CLAUDE.md` says | What we do | Why |
|---|---|---|---|
| 1 | §2, §7: the package is `vcc`, invoked `python -m vcc` | the package is **`vccp`**, invoked `python -m vccp` | avoids the import/PATH collision with the challenge's own `vcc` tool (§8.3) |
| 2 | §1: `configs/mini.yaml` has `device: auto` | `mini.yaml` has **`device: cpu`** | explicit beats inferred in the container; `server.yaml` keeps `auto` (§8.8) |
| 3 | §4.3: later phases train adapters and `δ` by default, the core is unfrozen only if the config says so | the config says so **for Phase 2** (`unfreeze_core: true`); Phase 3 stays frozen | Phase 2 has the data; Phase 3 sees only controls. Backed by the core-freeze ablation rather than asserted (§8.9) |
| 4 | §4.8 item 13: artifact includes `submission/prediction.vcc`, and `vcc prep` packaging when available | deliverable is `submission/prediction.h5ad`; `.vcc` is optional | the user packages the submission themselves (§8.1, §8.10). The packaging code path is still built |
