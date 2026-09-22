# Virtual Cell Challenge 2026 — prototype build specification

This file is the complete specification for this repository. Read all of it
before writing code, and re-read the relevant section before each milestone.

---

## 0. Priorities (in this order)

**Goal of the first build:** the user's method (section 4), implemented
completely, running from beginning to end without errors, producing
*reasonable* predictions (sanity checks, section 6.1), and a valid
submission. Winning the competition is not a goal. The user will present this
prototype and its results at a proposal defense, so every run also produces a
results summary with figures (section 5). Optimization comes later, designed
by the user after seeing the leaderboard.

1. **Every element of the method is real.** Each item in the element checklist
   (section 4.8) is implemented as specified, actually trained or run by
   `python -m vcc all`, and leaves the artifact the checklist names. No stubs,
   placeholders, `TODO`s, `NotImplementedError`s, or "simplified for now"
   versions in the delivered code path. If an element truly cannot be built as
   specified, implement the closest faithful version, and report it as a
   **deviation** in `DECISIONS.md` and in the final message — never silently.
   Do not add components the specification does not ask for.
2. **End to end, no manual steps.** One command runs everything on
   `mini_data/` on CPU (target: under 15 minutes) and the same command, with a
   different config, runs on the full data on the user's GPU server.
3. **Nothing sized by hand.** Never hardcode 18,533 genes, 300 perturbations,
   400 cells, 3 contexts, context names, or any path. Read them from the data,
   the manifest, and the config. `mini_data/` deliberately uses a different
   gene count so that hardcoding fails the tests.

**Before writing code**, write `PLAN.md`: for every section-4.8 checklist item,
the module that will implement it, the artifact it will produce, and the test
that will cover it. Keep `PLAN.md` updated as you go.

After the first submission the user will keep optimizing this prototype, so
favour clear modular code over clever code. Record every non-obvious choice in
`DECISIONS.md` (one short entry each: decision, reason, alternative).

### Timeline

- Validation round: contexts A, B, C, 300 perturbations; live leaderboard.
- Final test data released 2026-10-22: new contexts D, E, F and a **different**
  perturbation list. The pipeline must run on them with only a config change.
- **Validation-round submission deadline: 2026-09-30.** Time is short: build
  in the milestone order of section 9 so a complete, validated run exists as
  early as possible, and keep default training budgets small enough that
  `all` finishes on the server within about 12 hours on one GPU (all budgets
  configurable).

---

## 1. Environments

**Cloud (you, now):** this repository with `mini_data/`, CPU only. Install from
`environment.yml`; if conda is unavailable, pip-install the same packages and
CPU PyTorch. Every test and the full pipeline must pass here.

**Server (user, later):** Linux with an NVIDIA GPU, conda env `vcc`
(Python 3.11, anndata, scanpy, h5py, pyarrow, polars, scipy, scikit-learn,
PyTorch with CUDA, lightning). GPU memory is not yet known: make batch sizes,
model width and gene-chunk sizes configurable, default to settings that fit
within the GPU memory cap of section 1.2 (half of a 24 GB GPU unless the user
changes it), and use mixed precision when CUDA is available.

**Configs** (create both):

`configs/mini.yaml` — paths relative to the repo:
```yaml
data_root: mini_data          # contains ref/ and processed/
vcc_root: mini_data/vcc       # challenge release files
output_root: runs
device: auto                  # cpu in the cloud
```

`configs/server.yaml`:
```yaml
data_root: /data/han/projects/VCC/data
vcc_root: /data/share/han/VCC
output_root: /data/han/projects/VCC/runs
device: auto
```

On the server, HGNC is `ref/hgnc_complete_set.txt`; in `mini_data` it is
`ref/hgnc_subset.txt` (same columns). Accept either.

**Official scorer:** install `cell-eval2` (`pip install cell-eval2`; the base
install is CPU-only). Add it to `environment.yml`. Do not install its `[gpu]` or
`[gpudge]` extras: `[gpudge]` pulls its own `torch` and can break the model's
environment, and CPU scoring is enough for the rehearsal. Pin `de.backend`
explicitly (e.g. `"pdex"`) so scoring does not depend on which engine a host
happens to have.

**Packaging tool:** the challenge's `vcc` command-line tool (`vcc prep`,
section 8). The user installs it on the server; do not try to install it in
the cloud. Code must work with and without it on `PATH`.

### 1.1 Where the data lives

- **Only `mini_data/` is in this repository.** The full processed data is
  hundreds of GB and stays on the user's server at the `server.yaml` paths.
  You cannot access those paths; do not try, and do not treat their absence
  as an error to fix.
- **`mini_data/` mirrors the server layout exactly:** `mini_data/` has the
  same structure as the server's `data_root` (`ref/`, `processed/`), and
  `mini_data/vcc/` the same as `vcc_root`. Same file names, formats and
  columns; only fewer genes, cells, targets and cell lines. Code written
  against `mini_data` therefore runs on the server with only the config
  changed — so **one code path for both**: no `if mini:` branches, no
  mini-specific shortcuts, no assumptions that fit only small data.
- **Differences to expect on the server,** so nothing breaks there: millions
  of cells instead of thousands (stream, never load a whole Replogle file);
  18,533 genes; 300 targets per round and 400 cells each; possibly an extra
  Replogle context (`K562_essential`); HGNC file name as above. Size memory
  use by config, not by what fits `mini_data`.
- **Never modify, regenerate, download or commit data.** `mini_data/` is
  read-only input. Do not create a `data/` folder or placeholder data files.
  `.gitignore` must exclude `runs/` and any outputs; never commit model
  checkpoints, predictions or submissions.
- Because the full data cannot be tested here, `check-data` must give the
  user a clear pass/fail on the server **before** any training starts: every
  expected file present, columns and gene orders consistent, and the sizes it
  found (cells, genes, targets per source) printed for the user to confirm.

### 1.2 The server is shared — be a good neighbour

The server (192 cores, GPUs) is shared with the user's labmates. The pipeline
must never flood it: take the resources it needs and leave everything else
free. The user starts every session with:

```bash
# CPU courtesy — 192 cores, be a good neighbor
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1

# GPU
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Put exactly this block in `scripts/server_env.sh`, plus two lines the block
lacks: `POLARS_MAX_THREADS` and `RAYON_NUM_THREADS` (polars, used by
`cell-eval2`, otherwise starts one thread per core), both set to
`resources.cpu_threads` (default 4). The README tells the user to
`source scripts/server_env.sh` before any run.

Rules for the code:
- **Never override these variables.** Respect whatever the environment says.
  If one is unset when `python -m vcc` starts, set a conservative default with
  `os.environ.setdefault(...)` **before** importing numpy, torch, polars or
  scanpy (thread pools are fixed at import), and log the effective values at
  the start of every run.
- **CPU:** `torch.set_num_threads` and every worker pool (DataLoader
  `num_workers`, joblib, multiprocessing, cell-eval2's own parallelism) come from
  config: `resources.cpu_threads` (default 4) and `resources.num_workers`
  (default 2). No `n_jobs=-1`, no pool sized by `os.cpu_count()`, anywhere.
- **GPU:** use only the device(s) in `CUDA_VISIBLE_DEVICES`, addressed as
  `cuda:0` — never pick a physical index, never touch other GPUs, never
  reassign `CUDA_VISIBLE_DEVICES`. One GPU process at a time. Cap memory with
  `torch.cuda.set_per_process_memory_fraction(resources.gpu_memory_fraction)`
  (default 0.5), and size default batches to fit well inside that cap. Nothing
  may preallocate the whole GPU (no cupy or JAX memory pools; `cell-eval2`
  runs on CPU, section 1). Call `torch.cuda.empty_cache()` between stages so
  memory the pipeline no longer needs is returned to others. If a batch does
  not fit, fail with a clear message naming the config key to lower — never
  retry by grabbing more memory.
- **RAM:** stay under `resources.max_ram_gb` (default 64): derive chunk and
  block sizes from it, stream large files (section 3.3), and free big arrays
  as soon as a stage is done.
- **Disk:** write only under `output_root`; delete temporary files when a
  stage finishes.
- **Long runs:** README runs `all` as
  `nice -n 10 ionice -c3 python -m vcc all --config configs/server.yaml`,
  so interactive work by others keeps priority.
- **Default training budgets** are chosen to respect these limits and the
  ~12-hour target of section 0; anything heavier is opt-in via config.

Add a `resources:` block to both configs with these keys and defaults
(`device: cpu` for `mini.yaml`).

---

## 2. Repository layout

Existing — **do not change their behaviour**:
```
data_prep/     genes.py  harmonize_data.py  phase_data.py  challenge_prep.py  make_mini_data.py
mini_data/     small example of all processed inputs (section 3)
docs/          metrics.md, vcc2026-metrics-brief.pdf — the official scoring rules
environment.yml
CLAUDE.md
```
Reuse from `data_prep/phase_data.py`: `ReplogleCells` (per-cell loader) and
`log1p_cp10k` (the normalization). Import them; do not reimplement.

Create:
```
vcc/            the package (data/, priors/, models/, phases/, eval/, submit/, report/)
configs/        mini.yaml, server.yaml
tests/          pytest, run on mini_data
README.md       exact commands for the server, expected outputs and runtimes
scripts/        server_env.sh (section 1.2)
DECISIONS.md
```

---

## 3. Data contract

### 3.1 One gene coordinate system

All genes are **challenge genes in challenge order** (`vcc/gene_names.csv`).
`challenge_idx = i` means the i-th gene of that list in every file. Genes are
named by challenge symbols everywhere.

**Gene split** (`processed/phases/gene_split.csv`; columns `gene_symbol,
ensembl_id, challenge_idx, lincs_entrez, lincs_feature_space, split`):
- `panel` = LINCS landmark genes that are challenge genes (under 978)
- `rest` = every other challenge gene

Coverage differs by source. Replogle measures only part of the panel and
about 7,000 of the rest genes; the challenge measures all genes. About 10,000
challenge genes are measured nowhere except the challenge controls.

### 3.2 Normalization and units

- Replogle and challenge: `log1p(counts / library_size * 1e4)`, via `log1p_cp10k`.
  - Replogle `library_size` = `obs['UMI_count']` (all UMIs of the cell).
  - Challenge `library_size` = `obs['total_counts']` if present (mini data,
    where genes were subsetted), else the row sum over all genes.
- LINCS Level 3 = log2 normalized bulk expression. LINCS Level 5 = robust
  z-scores vs same-plate controls.
- **Control-SD units** bridge sources: `z = (x - ctrl_mean) / ctrl_std`,
  per gene and per context, with `ctrl_mean`/`ctrl_std` stored in each file's
  `var`. LINCS signatures are already in these units. Guard against
  `ctrl_std == 0` (floor it).

### 3.3 Files

Paths below are relative to `data_root` unless marked `vcc_root`.

**Reference**
- `ref/challenge_genes.tsv` — `symbol<TAB>ensembl`, no header, official order.
- `ref/challenge_perts.txt` — perturbation targets to predict, one per line.
- `ref/hgnc_*.txt` — HGNC table (`symbol`, `gene_group`, `ensembl_gene_id`,
  `prev_symbol`, `alias_symbol`, …).

**Challenge release** (`vcc_root`)
- `manifest.json` — `pert_col` ("target_gene"), `context_col` ("context"),
  `control_label` ("non-targeting"), `n_genes`, `cells_per_pert`,
  `per_context` (control_cells, ground_truth_cells, n_ntc_ids, …).
  Read `cells_per_pert` from here; do not assume 400.
- `gene_names.csv` (column `gene_name`), `pert_counts.csv` (column `target_gene`).
  `pert_counts.csv` is the authority on which perturbations the submission
  must contain; `check-data` verifies `ref/challenge_perts.txt` matches it.
- `context_<X>.h5ad` — control cells only, raw integer counts;
  `obs`: `target_gene` (= "non-targeting"), `context`, `ntc_id` (control guide).

**Phase 1 — LINCS** `processed/phases/phase1_lincs.h5ad`
- one row per `(cell_line, target_gene, time_h)`, CRISPR knockouts only
- `var` = panel genes (`challenge_idx`, `ensembl_id`, `lincs_entrez`)
- `layers`: `ctrl` (Level 3, same-plate controls), `pert` (Level 3, treated),
  `sig` (Level 5 mean), `gmt` ((#up-sets − #down-sets) / #signatures, in [−1, 1])
- `obs`: `cell_line, target_gene, time_h, n_pert_wells, n_ctrl_wells,
  ctrl_source, n_sigs, has_sig, n_gmt_signatures, has_gmt, cc_q75_median,
  is_challenge_pert`
- Note: LINCS perturbations are CRISPR **knockout**; the challenge is CRISPR
  **interference** (knockdown). Directions and affected genes transfer well;
  magnitudes and the target gene's own level do not.

**Phase 2 — Replogle** `processed/phases/phase2/<context>/`
(contexts: `K562_gwps`, `rpe1`; server may also have `K562_essential`)
- `genes.csv` — challenge genes measured here: `gene_symbol, challenge_idx,
  source_col, split, ctrl_mean, ctrl_std`
- `cells.parquet` — `row, target_gene, is_control, gem_group, library_size`
- `pseudobulk.h5ad` — one row per target plus `non-targeting`, mean
  log1p(CP10K); `obs`: `n_cells, is_control, is_challenge_pert`
- `meta.json` — source file, counts, normalization
- Cells are read on demand:
  ```python
  from data_prep.phase_data import ReplogleCells
  rc = ReplogleCells(".../phase2/K562_gwps")
  panel, rest = rc.controls(256, rng)      # (n, n_panel), (n, n_rest)
  panel, rest = rc.cells("ADNP", 256, rng)
  rc.targets                               # list of targets
  ```
- **Never load a full Replogle single-cell file into memory** (K562 genome-wide
  is ~2 million cells, 65 GB). Sample cells through `ReplogleCells`; use
  `pseudobulk.h5ad` for per-target means.

**Replogle source files** `processed/replogle/`
- `*_raw_singlecell.h5ad` — raw counts; `obs`: `target_gene, target_gene_orig,
  is_control, gem_group, UMI_count, …`; `var`: `challenge_idx` (−1 = not a
  challenge gene), `match`, …
- `*_bulk.h5ad` — per-perturbation rows (one per promoter, so a gene can have
  several rows); `obs['fold_expr']` = target expression in perturbed cells /
  control, i.e. **knockdown efficiency**; also `anderson_darling_counts`
  (≈ number of responsive genes). `raw_bulk` X = mean expression,
  `normalized_bulk` X = z-scores (may contain inf; mask non-finite values).

**Phase 3 — challenge** `processed/phases/phase3/`
- `genes.csv` — all challenge genes with `split`
- `controls_<ctx>.h5ad` — `X` = log1p(CP10K) (sparse), `layers['counts']` raw;
  `var`: `challenge_idx, split, ctrl_mean, ctrl_std`; `obs`: `library_size`, `ntc_id`, …
- `targets.csv` — one row per (context, target_gene) to predict, with
  `n_cells` and which training sources contain the target

**Coverage** `processed/pert_coverage.csv`, `processed/gene_coverage.csv`.

### 3.4 Facts about the full data (for design decisions)

- K562 genome-wide: ~2.0M cells, ~9,900 targets, ~8,200 genes;
  contains 272 of the 300 validation targets.
- RPE1: ~248k cells, ~2,400 targets; contains **none** of the 300 targets
  but shares many targets with K562 genome-wide (use for cross-context tests).
- Median knockdown `fold_expr`: 0.155 (K562), 0.088 (RPE1); 4–9% of guides weak.
- Challenge: ~20,000 UMIs per cell; Replogle K562 ~11,000–15,000.
- 28 validation targets are in no Replogle screen. The final test round has
  a different target list; coverage may differ.

---

## 4. The method

One **shared gene-token model** is updated by three phases. Genes are tokens,
so each phase can use a different gene set with the same network.

```
                 Gene vocabulary (all challenge genes, prior-initialized)
                                        |
                  Shared gene-token model (frozen core + phase adapters)
                 /                      |                       \
      Phase 1: LINCS  ------>  Phase 2: Replogle  ------>  Phase 3: challenge
      signatures from          panel -> rest mapping       adapt on controls only
      controls                        |                           |
                                  Rehearsal  --------------------> (decides what adapts)
                                                                  |
                                                             Submission
```

### 4.1 Gene vocabulary and priors

Each gene's embedding is

    embedding(g) = W · prior(g) + δ(g)

`prior(g)` concatenates fixed feature blocks, each reduced by PCA to a
configurable size (default 32), standardized, zero-filled where the source
does not cover g, with a per-block missing flag. `W` is a learned projection
shared by all genes. `δ(g)` is a free per-gene vector, initialized to zero,
L2-regularized.

Prior blocks, all built from files present in the data:
1. **Challenge control co-expression** — gene-gene correlation across all
   challenge control cells (every context) → PCA. Covers every gene; the most
   important block for genes measured nowhere else.
2. **Replogle control co-expression** — same, from Replogle control cells.
3. **LINCS signature co-variation** — genes as columns of `phase1 layers['sig']`.
4. **GMT co-membership** — genes as columns of `phase1 layers['gmt']`.
5. **HGNC gene groups** — multi-hot `gene_group` → truncated SVD.
6. **Perturbation phenotype** (target role only) — a target's knockdown
   response (pseudobulk delta in control-SD units, Replogle; `sig`, LINCS).
7. Optional: protein language model embeddings, only if the config points to
   an existing file. Never download at runtime.

Genes play two roles — measured feature and perturbation target — build a
feature-role and a target-role prior, and let the model combine them.

**Leakage rule:** any prior built from perturbation responses must be rebuilt
without held-out targets / held-out contexts when used in the rehearsal.
Priors for "pretend-unmeasured" genes in the rehearsal may use only the
rehearsal context's controls.

Checks (log them): nearest neighbours of a few well-known complex members
(ribosomal proteins, proteasome, mitochondrial respiratory chain) should be
each other; genes with similar target-role embeddings should have correlated
Replogle knockdown responses.

### 4.2 Shared gene-token model

- A cell (or bulk profile) is a set of `(gene embedding, value)` tokens over
  the **input** genes (the panel). Encode values with a small MLP added to the
  gene embedding.
- Compress with a Perceiver-style latent bottleneck (default 64 latents), so
  cost does not grow with the number of output genes.
- Condition on the perturbation with a **perturbation token**: the target's
  target-role embedding plus a learned "knockdown" type embedding.
- Condition on the context with a context vector computed from that context's
  control profile (e.g. pooled over control cells), applied via FiLM or adapters.
- Predict any output gene by **querying** its embedding against the latents.
  Output genes are processed in chunks.
- Keep it modest: first build should train within a few GPU-hours in total.

### 4.3 Guard against forgetting

- A **core** (value encoder, latent blocks, decoder) plus small **adapters**
  (LoRA or bottleneck) per phase and per context.
- Later phases train adapters and `δ` by default; the core is unfrozen only if
  the config says so.
- Options, all configurable: replay of a fraction of earlier-phase batches,
  and an L2-SP penalty pulling weights toward the previous phase's values.

### 4.4 Phase 1 — LINCS

- Input: control profile (`layers['ctrl']`, panel genes, per-gene
  standardized) + perturbation token (+ cell-line context).
- Heads: signature (`sig`, regression; `gmt`, masked where `has_gmt` is false)
  and perturbed profile (`pert`).
- Two-step as in the sketch: control + perturbation → signature;
  control + signature → perturbed. Train jointly.
- Split by **target gene** for validation (unseen perturbations), and report
  per-cell-line results too.

### 4.5 Phase 2 — Replogle

- The panel→rest mapping on **single cells**: panel values in, rest values
  out. Train on control cells and perturbed cells with the **same** weights
  (Siamese / shared network), so the mapping must hold in both states.
- Adapt the Phase 1 perturbation module to Replogle: control profile +
  perturbation token → perturbed panel (pseudobulk and sampled cells), in
  control-SD units.
- Loss only on genes measured in the context (masks from `genes.csv`).
- Hold out targets by gene for validation.

### 4.6 Rehearsal — Phase 3 on Replogle with the answers hidden

Runs Phase 3's procedure on a Replogle context whose perturbed cells are
hidden, then scores against them. Report, per variant:

1. **Controls-only mapping:** fit/adapt the panel→rest mapping on the held-out
   context's controls only; feed the **true** perturbed panel of held-out
   targets; score predicted rest genes (as changes from control).
   Compare with an **upper bound** (mapping also trained on perturbed cells)
   and a **floor** (predict no change in rest genes).
2. **Cross-context:** learn perturbation behaviour in one context, adapt to
   the other using only its controls, predict its perturbed cells for shared
   targets (K562 genome-wide ↔ RPE1). This is the closest analogue of the
   challenge.
3. **Unseen genes:** hide a random set of genes from all training except the
   rehearsal context's controls; score their perturbed predictions.

**Scoring.** Variant 2 produces full predicted cells, so it is scored with
the **six official metrics** exactly as the challenge scores them (section 6):
predicted cells in raw counts through the Phase 3 cell generator, truth = the
held-out context's real perturbed cells plus its real control cells, all in
counts. Variants 1 and 3 predict only part of the genes; score them with the
same metrics where a full profile can be assembled (true panel values plus
predicted rest), otherwise with per-gene correlation and error on the genes
involved. Cap the number of held-out targets by config (default 100) so DE
scoring stays fast.

Output: `rehearsal/report.json` + a readable summary, and a
`phase3_policy.json` that states which modules Phase 3 may adapt and the
generator calibration of section 4.7, chosen from these results. The upper bound and the no-change floor are reference
points for judging the method; they are never submitted.

### 4.7 Phase 3 — challenge

- Adapt, according to `phase3_policy.json`: context adapters and `δ` for genes
  first seen here, trained on the challenge controls (panel → rest, both halves
  observed). The perturbation module stays frozen.
- Predict for every (context, target) in `targets.csv`:
  1. control panel profile + perturbation token → predicted perturbed panel
  2. perturbed panel → predicted perturbed rest, via the adapted shared network
  3. **Knockdown prior** for the target gene's own expression: default
     `ctrl × median fold_expr` (Replogle), per-target `fold_expr` when the
     target is in Replogle bulk. Applies whether the target is panel or rest.
     Note: all six official metrics exclude each target's own gene (and
     discrimination excludes every target gene of the panel), so this prior
     does not move the score; it is kept because it is correct biology and
     part of the design.
- **Generate `cells_per_pert` distinct cells per target, in raw integer
  counts.** The official metrics compute DE on the submitted cells (Wilcoxon
  against the real control cells) and reward realistic cell-to-cell
  variation, so the generator is part of the method, not packaging:
  - For each (context, target), draw a **fresh, independent** sample of that
    context's control cells (with replacement if needed). Never reuse one
    block of cells across targets: the expression metric withholds its noise
    correction from such submissions.
  - Apply the predicted per-gene fold changes **in count space**, keeping
    integers: binomial thinning for decreases, scaling with stochastic
    rounding for increases. This keeps each cell's dispersion realistic;
    groups whose cells are too similar are penalized by the expression metric.
  - **Genes with no confident predicted change keep a fold change of exactly
    1**, so their predicted cells are exchangeable with controls and are not
    called differentially expressed. The DE metrics penalize calling too many
    genes (the Jaccard divides by the union) and calling too few (fidelity is
    scaled by yield, and reach is measured against all genes the reference
    calls significant). The expression metric scores a prediction of
    zero change at 1.0, so over-large or noisy predicted effects can score worse
    than predicting nothing.
  - Two calibration settings are therefore fitted on the rehearsal (variant 2)
    by maximizing the mean of the six official metrics, and written to
    `phase3_policy.json`: a **confidence threshold** below which a gene's
    predicted change is set to zero, and a global **effect-size scale**
    applied to the remaining predicted log fold changes.
  - Make the generator a swappable component with these settings in the config.
- The generated cells of every context go into **one** submission file
  (section 8). The control cells are inputs only: they are resampled to build
  predicted cells, but never written to the submission themselves.

### 4.8 Element checklist

Every item below is required. Each must be produced by `python -m vcc all`
under `output_root/<run_name>/`, on `mini_data` and on the server alike.

| # | Element (section) | Must do | Artifact |
|---|---|---|---|
| 1 | Gene vocabulary (4.1) | all challenge genes; `embedding = W·prior + δ` | `priors/prior_features.npz` |
| 2 | Prior blocks (4.1) | blocks 1–6 built (7 if configured), coverage per block | `priors/coverage.csv` |
| 3 | Two gene roles (4.1) | feature-role and target-role priors and embeddings | both in `priors/prior_features.npz` |
| 4 | Prior checks (4.1) | nearest-neighbour and response-correlation checks | `priors/checks.json` |
| 5 | Shared gene-token model (4.2) | one core used by all three phases; value encoder, latent bottleneck, gene-query decoder, perturbation token, context conditioning; any input/output gene subset | `checkpoints/core_phase{1,2,3}.pt` |
| 6 | Frozen core + adapters (4.3) | per-phase and per-context adapters; frozen parameters verified unchanged | `reports/frozen_check.json` |
| 7 | Guard against forgetting (4.3) | replay and L2-SP, on by default in Phases 2–3; Phase 1 validation re-scored after Phase 2 and after Phase 3 | `reports/forgetting.json` |
| 8 | Phase 1 (4.4) | control + perturbation → signature (`sig`, `gmt`) → perturbed; held-out targets | `phase1/metrics.json` |
| 9 | Phase 2 (4.5) | Siamese panel→rest on control **and** perturbed cells with shared weights; perturbation module adapted to Replogle; masked losses | `phase2/metrics.json`, with control and perturbed cells reported separately |
| 10 | Rehearsal (4.6) | all three variants: controls-only mapping (with upper bound and floor), cross-context K562 ↔ RPE1, unseen genes | `rehearsal/report.json`, `phase3_policy.json` |
| 11 | Phase 3 adaptation (4.7) | adapters and `δ` trained on each context's challenge controls per the policy; perturbation module frozen | `phase3/adapt_<context>.json` (loss before/after) |
| 12 | Phase 3 prediction (4.7) | panel via perturbation model, rest via adapted shared network, knockdown prior, `cells_per_pert` distinct cells | `predictions/model/` |
| 13 | Submission and validation (8) | one-file writer, validator, `vcc prep` packaging when available | `submission/prediction.h5ad`, `reports/validate.json`, `submission/prediction.vcc` (server) |
| 14 | Sanity checks (6.1) | all seven checks | `sanity/report.json` |
| 15 | Results summary (5) | tables and figures for the proposal defense | `reports/summary.md`, `reports/figures/` |

At the end of every `all` run, write `checklist.json`: for each item, `done`
or `deviation` (with the reason), and whether its artifact exists. A test must
run `all` on `mini_data` and assert that every artifact exists and every item
is `done` or an explicitly reported `deviation`. Include the checklist in the
final message to the user.

---

## 5. Results summary (for the proposal defense)

Every `all` run ends with a `report` stage that writes `reports/summary.md`
and PNG figures in `reports/figures/` (matplotlib; readable when projected:
large fonts, labelled axes, units, one message per figure). It collects,
without re-running anything:

- **Data used:** cells, targets and contexts per source; gene split sizes;
  how many challenge genes and targets each source covers.
- **Method:** a short description of each phase as built, with the
  checklist status (section 4.8) and any deviations.
- **Phase 1 and Phase 2 results:** held-out-target metrics; Phase 2 reported
  separately for control and perturbed cells; training and validation curves.
- **Rehearsal:** all three variants against the upper bound and the no-change
  floor, as a table and a bar chart; the six official metrics for the
  cross-context variant, on the leaderboard's 0–1 scale where available;
  predicted vs true change for a few held-out targets as scatter plots.
- **Forgetting:** Phase 1 validation score after Phase 1, 2 and 3.
- **Phase 3:** adaptation loss before and after, per context; the knockdown
  check (predicted vs control expression of each target gene) as a figure.
- **Sanity report** and **submission summary** (what was submitted, sizes).

Numbers on `mini_data` are not meaningful; the summary must say so when it is
produced from `mini_data`.

---

## 6. Evaluation

The challenge scores six metrics, computed by the public `cell-eval2` package
with its `vcc2026` preset. Full definitions: `docs/metrics.md` (sections 2.3,
3, 4.3, 4.5) and `docs/vcc2026-metrics-brief.pdf`. Read both before M4.

| metric (cell-eval2 name) | what it asks | better | "no change" scores |
|---|---|---|---|
| `pds_cosine` — perturbation discrimination | is each predicted effect closer (cosine) to its own real effect than to any other target's? rank within the panel | higher | 0.5 |
| `expr_mse_unbiased_capped_norm` — expression accuracy | squared error of predicted vs real profile, corrected for finite-cell sampling noise, divided by the real effect size; one ratio over the whole panel | lower | 1.0 |
| `de_wilcoxon_lfc_nmae` — DE log-FC accuracy | mean absolute error of log2 fold changes over the reference's significant genes, normalized | lower | ≈ 1.0 |
| `de_wilcoxon_direction_fidelity_yield_raw` — DE direction fidelity | of the genes the prediction calls significant, the share moving in the real direction, scaled by yield | higher | ≈ 0 |
| `de_wilcoxon_direction_reach_raw` — DE direction reach | ranking the reference's significant genes by the prediction's confidence, how deep directions stay ≥ 90% correct, as a fraction of the reference's calls | higher | ≈ 0 |
| `de_wilcoxon_sig_jaccard` — DE significance overlap | overlap of the two significant-gene sets over their union | higher | ≈ 0 |

Facts that matter for the method (the scorer implements them; do not
reimplement them):
- Input is **raw counts** on both sides. Expression and discrimination use a
  group-sum pseudobulk (`bulk_lognorm`); DE uses per-cell normalized values,
  a per-perturbation Wilcoxon test with BH correction, significance at
  adjusted p < 0.05, and genes kept only if the control expresses them
  (minimum 5 CPM).
- Predicted effects are measured against the **real** control
  (`control_source="real"`), for both discrimination and DE.
- All six exclude each target's own gene; discrimination excludes every target
  gene of the panel.
- The leaderboard rescales each metric so that **0 = the organizers'
  mean-response baseline** and **1 = a split-half replicate of the real data**,
  then averages the six.

**Local scoring.** Implement `vcc/eval/official.py` as a thin wrapper around
`cell_eval2.compute_metrics` with
`replace(EvalConfig.from_preset("vcc2026"), pert_col="target_gene")` and an
explicit `de.backend`. It is used by the rehearsal (section 4.6) and the
sanity checks (section 6.1), and returns the six values plus the
diagnostic counts `de_wilcoxon_nsig_counts_real` / `_pred`. Where the data
allow, also report the rehearsal on the leaderboard's 0–1 scale, using
`cell-eval2`'s own tools: `cell-eval2 baseline` for the 0 end and
`run --anchor` for the 1 end, built on the rehearsal context's real data.
Those two are reference points of the scorer only — never predictions, never
submitted, never used to train or choose the model. If they cannot be built on
a given dataset, report the six raw values alone and say so.

Do not reimplement the metrics. If `cell-eval2` cannot be installed in the
cloud environment, mark the scoring tests as skipped with a clear reason, keep
the wrapper, and flag it in the final message — the user's server has network
access and will run it.

### 6.1 Sanity checks — what "reasonable" means

A `sanity` stage runs automatically at the end of every `all` run, on the
predictions that go into the submission (and on the rehearsal predictions,
where the truth is known). Each check reports **pass / warn / fail** with the
measured value and the threshold. Every threshold is in the config with the
defaults below.

1. **Complete and valid.** Every rule of section 8: no NaN or infinite
   values; non-negative whole numbers; no cell above 1,000,000 counts; every
   (context, target) has exactly `cells_per_pert` cells; no control rows;
   gene order matches `gene_names.csv`; stored entries within the cap. Any
   failure here is a hard **fail**.
2. **Plausible magnitudes.** Per context, the median predicted library size is
   within 0.5–2× the median control library size, and per-gene mean
   expression stays within the range seen in that context's controls
   (allowing for the knocked-down gene and genes the model predicts to move).
3. **The knockdown is visible.** For each target whose gene is detectably
   expressed in controls (mean control CP10K above a configurable floor), the
   predicted mean expression of that gene is at most 0.7× control. Pass if
   this holds for at least 90% of such targets; list the targets that fail.
4. **Targets differ.** Excluding each target's own gene, predictions are not
   all the same profile: the median pairwise Pearson correlation of
   predicted mean changes from control across targets is below 0.95, and no
   two targets have identical predictions.
5. **Cells vary.** Within each (context, target), predicted cells are not
   copies of one average: the median per-gene variance across predicted cells
   is within 0.5–2× that of the context's control cells.
6. **Better than doing nothing.** On the rehearsal, where the truth is known,
   the official metrics beat their no-change values (section 6):
   `pds_cosine` > 0.5, `expr_mse_unbiased_capped_norm` < 1.0 and
   `de_wilcoxon_lfc_nmae` < 1.0.
7. **Reasonable number of DE calls.** On the rehearsal, the median over
   targets of (predicted significant genes / real significant genes), from
   the scorer's `nsig_counts` diagnostics, lies within 0.25–4. Near zero means
   the prediction is too timid to be scored by the DE metrics; far above means
   it floods them.

Output: `sanity/report.json` and a short readable summary printed at the end of
the run and written to `sanity/summary.txt`. A **fail** in check 1 stops the
pipeline before the submission is written. Other failures do not crash the
run, but they are printed prominently and `validate` refuses the submission
unless run with `--allow-warnings`. On `mini_data` the statistics are noisy,
so warnings are acceptable there; the checks must still run and report.

Tests: for each check, a unit test that feeds a deliberately broken
prediction (NaNs, wrong cell counts, target gene not reduced, all targets
identical, all cells identical, worse than no change, no DE calls at all) and
asserts the check catches it.

---

## 7. Engineering requirements

- CLI: `python -m vcc <stage> --config configs/<name>.yaml`, stages:
  `check-data, priors, phase1, phase2, rehearsal, phase3, predict,
  sanity, validate, package, report, all`. `all` runs everything in order and is resumable: finished
  stages are skipped unless `--force`.
- All outputs under `output_root/<run_name>/` (config, logs, checkpoints,
  reports, submission). Seeds fixed and logged.
- `check-data` verifies every file in section 3 exists with the expected
  columns and that all gene orders agree; clear error messages naming the file.
- Memory: stream Replogle through `ReplogleCells`; build predictions and the
  submission in chunks per context and target; never hold a dense
  cells × all-genes matrix for all targets at once.
- Tests (pytest, on `mini_data`, fast): data loading, gene alignment, priors
  shape and missing flags, one tiny training step per phase,
  rehearsal report produced, the validator rejects each violation of section 8
  (a control row, a construct id such as `ADNP-1`, a missing target, a wrong
  cell count, a non-integer or negative value, a cell above 1,000,000, explicit
  zeros, dense storage, wrong gene order), the official-scorer wrapper returns all six
  metrics on a tiny prediction, sanity checks catch broken predictions (6.1),
  submission passes the validator, and a test that
  fails if any module hardcodes the number of genes, perturbations, cells or
  contexts.
- Do not modify files in `data_prep/` or `mini_data/`. No network access at
  runtime.

---

## 8. Submission format

This section restates the challenge's submission guideline, which is the
highest authority on the submission. If any other document, including
`docs/`, disagrees with it, follow this section and record the difference in
`DECISIONS.md`.

**One file.** A single AnnData `.h5ad` holding the predictions for **all
contexts of the round**, packaged into one `.vcc` file for upload. Not one file
per context: `obs["context"]` separates them.

**obs** — exactly two required columns:
- `target_gene`: the perturbation's **gene symbol** (`ADNP`, never a construct
  id like `ADNP-1`), exactly as listed in `pert_counts.csv`.
- `context`: reused **exactly** as it appears in the control files'
  `obs["context"]` (e.g. `A`, `B`, `C`).
- **No control rows.** No `non-targeting` cells at all: every row is one of the
  perturbations to predict. The control cells are model inputs only and must
  never be copied into the prediction.

**var** — the index is the gene name: all genes of `gene_names.csv`, in that
exact order (18,533 in the validation round).

**X** — raw counts:
- non-negative, whole numbers, finite;
- no cell totalling more than **1,000,000** counts;
- exactly the perturbations of `pert_counts.csv`, `cells_per_pert` cells each
  (400 in the validation round), in **every** context — 360,000 rows ×
  18,533 columns in the validation round;
- **stored sparse**, with at most **4,750,000,000 stored entries** in total
  (about 13,200 per cell at this shape). Explicitly stored zeros count toward
  the cap: call `eliminate_zeros()` on every block, and never store a dense
  array.

**Building it without running out of memory.** The validation file holds on
the order of 2×10⁹ stored entries. Assemble it block by block (per context and
target) and write the sparse matrix to disk incrementally — for example by
appending CSR blocks to an on-disk sparse dataset — rather than concatenating
everything in RAM. With this many entries `indptr` must be 64-bit. Integer
counts may be stored as an integer dtype.

**Packaging** — the challenge's `vcc` command-line tool:

    vcc prep prediction.h5ad -g gene_names.csv --perts pert_counts.csv -o prediction.vcc

`vcc prep` validates the gene set, context labels, perturbation labels, cell
counts and the raw-counts requirement; `--dry-run` checks without writing. The
user installs `vcc` on the server. Detect it at runtime: if it is on `PATH`,
the `validate` stage runs `vcc prep --dry-run` and a `package` stage writes the
`.vcc`; if it is not (as in the cloud), skip both with a clear message. On
`mini_data` it may reject the file because the example is deliberately smaller
than the real round; treat that as a warning there, never as a pass condition.

**Our own `validate` stage** checks everything above before `vcc prep` runs:
gene names and order equal `gene_names.csv`; the set of `(context,
target_gene)` pairs equals every context × every target of `pert_counts.csv`,
each with exactly `cells_per_pert` rows; no control rows and no labels outside
the list; context labels identical to the control files'; values
non-negative, integral and finite; every cell total ≤ 1,000,000; stored
entries ≤ 4,750,000,000 with no explicit zeros; sparse storage. It writes
`reports/validate.json`.

**Local scoring vs. submission.** The submission has no control rows, but the
scorer needs the real control cells. For local scoring on the rehearsal the
`cell-eval2` wrapper takes the real controls from the real side; if
`cell-eval2` requires a control group in the prediction object too, the
wrapper adds the real control cells to its **in-memory copy for scoring only**
— never to a file that is submitted.

---

## 9. Milestones

Commit after each. From M1 on, keep `pytest` passing; from M5 on, keep
`python -m vcc all --config configs/mini.yaml` passing.

- **M0 — plan:** `PLAN.md` mapping every section-4.8 item to its module,
  artifact and test.
- **M1 — data layer:** configs, loaders for every file in section 3,
  `check-data`, tests.
- **M2 — gene vocabulary and priors:** all prior blocks, both gene roles, the
  embedding module, the prior checks of section 4.1.
- **M3 — shared model, Phase 1 and Phase 2:** the gene-token core, adapters,
  the forgetting guard, Phase 1 training, Phase 2 training.
- **M4 — rehearsal and Phase 3:** the three rehearsal variants,
  `phase3_policy.json`, Phase 3 adaptation, prediction with the knockdown
  prior and cell generation.
- **M5 — submission and reporting:** one-file submission writer, validator,
  `vcc prep` packaging when available, sanity checks, results summary,
  `checklist.json`, full `all` run on mini.
- **M6 — hand-off:** README with server commands and expected runtimes,
  DECISIONS.md, final message.

### Definition of done

- `pytest` passes and `python -m vcc all --config configs/mini.yaml` completes
  on CPU, producing a validated submission, a rehearsal report, a sanity
  report with no **fail** in check 1, and a results summary.
- `checklist.json` lists every section-4.8 item as `done`, or as a `deviation`
  that is also explained in `DECISIONS.md`; the corresponding test passes.
- README explains, for the server: `source scripts/server_env.sh`, the
  environment setup check (including that `vcc` and `cell-eval2` are
  installed), the `all` command, the `vcc prep
  --dry-run` check and the packaging command, how to run stages separately, where outputs go, how to switch to
  the final-test round (new `vcc_root`, new target list), and what to look at
  in the rehearsal and sanity reports before submitting.
- Final message to the user: the section-4.8 checklist with the status of
  every item, every deviation from this specification and why, rehearsal
  numbers on mini (and that mini numbers are not indicative of real
  performance), known limitations, and the exact server commands.
