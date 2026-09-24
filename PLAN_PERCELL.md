# Per-cell perturbation module — design

**Status: agreed and scheduled (§14). P0 `can-fit` on `mini_data` and on the
server; P1 landed (`vccp/train/ot.py`); P2 landed (the delta term, the type
vocabulary, the `core_scratch` arm); P3 landed (`perturbation_mode: percell`,
the per-pair delta, and the coupling premise check that gated it).**

The current perturbation module predicts one mean effect per (context,
target). This replaces it with one that maps **a cell** to **that cell
perturbed**, supervised by entropic optimal transport between drawn control
cells and drawn perturbed cells.

Read §1 for why, §2 for the method end to end, §7 for cross-assay
transfer, §11 for what could go wrong, and §13 for what was decided.

---

## 1. Why the current module has to change

Two separate problems, and the second is the reason for this document.

**It does not fit.** Phase 2 on the full data, validation *and* training:

| arm | perturbation ÷ no-change (val) | (train) | pearson |
|---|---|---|---|
| `core_unfrozen` | 1.003 | 1.036 | 0.161 |
| `core_frozen` | 1.088 | 1.121 | 0.073 |

A ratio of 1.0 means "no better than predicting that nothing happened". At
**1.036 on its own training targets**, the module cannot fit data it has
already seen. Everything downstream inherits that: the rehearsal's upper
bound — which saw the held-out screen's perturbed cells — behaved like the
no-change floor, and the first leaderboard submission scored −0.131 with DE
direction fidelity at −0.70.

**It answers the wrong question.** The specification asks for a model over
`(gene embedding, value)` tokens of a cell (§4.2). What was built pools the
control cells into one profile per context and predicts one mean response:

```python
# phase2.py, make_step — the same profile for every target, the pooled mean as truth
baseline   = tensors.control_panel.unsqueeze(0).expand(len(chosen), -1)
bulk_truth = torch.stack([tensors.pseudobulk[t] for t in chosen])
```

Per-cell heterogeneity in the submission comes entirely from resampled
control cells, not from the network. That was my decision, taken for the
measurement reason in DECISIONS.md D27 — in control-SD units the mean of *n*
cells carries variance 1/n, which at these cell counts was several times the
size of the response, so a per-cell baseline made the prediction worse than
predicting nothing. Stabilising the baseline fixed the measurement and
discarded the thing the model was supposed to learn.

The fix is not to un-pool naively — that reintroduces exactly the noise
problem. It is to change what the module is supervised *against*.

---

## 2. The method, end to end

```
Model 1 — LINCS (bulk: 688 paired wells, 955 panel genes)
  control profile + target token → signature (sig, gmt) → perturbed profile
                                   [the pair is OBSERVED]

Model 2 — Replogle (single cells; panel ⊂ 955, rest ≈ 7,000 of 17,578)
  control cell (B × 955) + target token ─── OT pairing ──→ perturbed cell (B × 955)
         │                                                        │
         │        shared gene-query decoder (same weights)        │
         ▼                                                        ▼
  control rest (B × N_rest)                           perturbed rest (B × N_rest)
         └──────────── delta consistency penalty ──────────────────┘
                  (their difference = the observed rest change)

Challenge
  Phase 3 adapt on that context's controls (context adapters + δ for genes
  first seen here); then
  control cell + target token → Model 2 (perturbation module frozen) → perturbed cell
```

Three changes from what is built: the perturbation input becomes a **cell**
rather than a pooled profile (§3), its target comes from **OT** rather than a
pseudobulk mean (§4), and the two decoder arms gain a **delta consistency**
term (§5). Everything else — the gene vocabulary, the core, the adapters,
the rehearsal, the generator, the submission path — is unchanged.

### 2.1 Three facts about the data that constrain this

**LINCS has no cells.** `phase1_lincs.h5ad` is `(688, 955)`: 688 rows, each a
bulk well average over 6–235 wells (`obs.n_pert_wells`, `n_ctrl_wells`).
Model 1 can only ever be profile-level. The cell-level claim starts at
Model 2, and the defense write-up must say so.

**The arrow is well-posed exactly where there are no cells.**

| | has cells? | is the pair (control → perturbed) observed? |
|---|---|---|
| LINCS (Model 1) | no — bulk wells | **yes** — `layers['ctrl']` / `layers['pert']`, same plate |
| Replogle (Model 2) | yes — ~2M cells | **no** — destructive assay |
| challenge | yes — controls only | no; that is the task |

Replogle cell *i* (control) and cell *j* (perturbed) are different cells:
*i* was never perturbed and *j* was never observed unperturbed. So
`control cell + target token → that cell perturbed` has **no `y` for its
`x`**. It is a distribution transport problem, not supervised regression,
and the pairing rule has to be named. That is the gap that produced the
failure — the current code resolves it by collapsing both sides to a pooled
mean. **Filling in the pairing rule is the whole fix**; nothing downstream
of the perturbation arrow was the broken part.

**The rest genes narrow and then widen.**

| | panel | rest |
|---|---|---|
| challenge (server) | 955 | 17,578 |
| challenge (`mini_data`) | 955 | 1,028 |
| Replogle K562_gwps (mini) | 716 measured | 414 measured |
| Replogle rpe1 (mini) | 812 measured | 432 measured |
| Replogle (server) | part of 955 | ≈ 7,000 |

So the decoder is *trained* on ~7,000 rest genes and *required* on 17,578;
roughly 10,000 challenge genes are observed nowhere except the challenge
controls. A fixed-width output head cannot span that — the decoder must stay
**query-based** (a gene's embedding in, its value out), with the
prior-initialized vocabulary carrying the unseen genes. That is the existing
design and it is load-bearing.

### 2.2 Why panel first, then rest

Perturbation supervision exists only for the 955 panel genes (LINCS) and for
the ~7,000 Replogle rest genes. Expanding first and perturbing second would
demand perturbation supervision on genes that have none. Perturbing the
panel and then expanding confines the hard problem to where the data is.
This is the built order and it is kept.

### 2.3 What stays from CLAUDE.md

Two things a cell-level sketch tends to drop, both of which stay:

* **Phase 1's `sig`/`gmt` heads and the two-step** (control + perturbation →
  signature; control + signature → perturbed) — §4.4 and checklist item 8,
  artifact `phase1/metrics.json`.
* **Phase 3 adaptation** on each context's challenge controls — checklist
  item 11. Unavoidable: the context is new and ~10,000 of its genes are
  first seen there.

---

## 3. What the module becomes

Today:

```
control pseudobulk (1 × 955)  +  target token  →  perturbed panel (1 × 955)
```

Proposed:

```
control cell (B × 955)  +  target token  →  that cell perturbed (B × 955)
```

The network barely changes. `predict_perturbed_panel` already takes a batch
of profiles and a per-row target index; it is fed one pooled row today. The
head already predicts a **change** added to the input, which is the right
parameterization here: a perturbed cell is mostly its own control, and the
residual is what the module has to learn.

What changes is the **loss**, the **batch shape**, and the **generator**.

---

## 4. The loss: entropic OT with a barycentric target

A perturbed cell has no matched control cell. Supervision has to construct
one.

Per step, for one target `t` in one screen:

1. draw `B` control cells and `B` cells perturbed for `t`, in control-SD
   units over the panel — `ContextTensors.draw_controls` /
   `draw_perturbed` already do this;
2. cost `C[i,j] = ‖control_i − perturbed_j‖² / n_panel` (scaled by the gene
   count so `ε` means the same thing on any panel);
3. entropic OT coupling `P = Sinkhorn(C, ε)` with uniform marginals, in the
   log domain, ~50 iterations. **Detached**: the coupling builds the target,
   it is not part of the model, and differentiating through Sinkhorn buys
   nothing here and biases the gradient;
4. barycentric target `T_i = Σ_j P̂_ij · perturbed_j`, where `P̂` is `P`
   row-normalized — the coupling-weighted average of the perturbed cells
   that control cell *i* was matched to;
5. loss `= mean‖predicted(control_i, t) − T_i‖²`, masked to the genes the
   screen measures and divided by its own no-change baseline, as today.

### Why entropic and not hard 1-to-1

At 12,000 UMIs the per-cell Poisson scatter is ~25× the perturbation
response we measured (variance 1.0 against 0.04 in control-SD units). A hard
assignment under that noise is close to arbitrary, and an L2 loss against an
arbitrarily chosen single cell fits noise. The barycentric average is the
variance-reduced version of the same idea.

### Why matching on "nuisance" is the point — and what P3 measured

The cost is dominated by sequencing depth and cell cycle, not by the
perturbation. That is meant to be the mechanism, not a flaw: pairing like
with like on those factors should mean the residual difference between
paired cells is closer to the perturbation than the difference between a
random control and a random perturbed cell. This is what CellOT and its
relatives exploit, and it is what pooling throws away.

**P3 measured it, and the claim does not hold at the concentrated end.**
`residual_reduction` is `E‖T_i − c_i‖² ÷ E‖pooled mean − c_i‖²`, so below 1
means pairing brought the target nearer:

| ε | effective partners | `target_spread` | `residual_reduction` |
|---|---|---|---|
| 0.005 | 3.2 | 0.57 | **1.35** |
| 0.01 | 8.0 | 0.38 | **1.17** |
| 0.02 | 34.6 | 0.15 | 0.98 |
| 0.05 | 117.5 | 0.02 | 0.94 |

A target averaged over three cells carries about a third of a cell's
sampling noise, and that is more than the pairing removes. The
control-against-control null traces the same curve, so this is noise
geometry, not anything about perturbation.

**The two wants pull opposite ways.** `target_spread` wants a small ε — a
large one hands every cell the same target and the per-cell loss becomes the
pooled loss. `residual_reduction` wants a large one. `phase2.ot_epsilon`
therefore defaults to **0.02**, where pairing measurably stops adding noise
while the targets still vary, and P5's sweep settles it (DECISIONS.md D61).

**Why this does not sink the redesign.** Noise in the target inflates the
loss without moving its minimum — the expected squared error against
`true + noise` is still minimized at `true` — and the reason for per-cell
supervision is that the model should be *asked* a per-cell question, which
§4.2 specifies and the pooled formulation never asks. But the
variance-reduction argument above, taken on its own, is not supported by
this measurement, and it should not be offered at the defense as though it
were.

### ε spans the whole design space

* `ε → ∞`: uniform coupling, every control matched to everything equally,
  `T_i` → the perturbed pseudobulk. **This is today's loss.**
* `ε → 0`: hard assignment, `T_i` → one perturbed cell.
* in between: local averages.

So `phase2.ot_epsilon` is one config key that moves continuously from the
current formulation to hard OT, the pooled arm stays alive as a control, and
the rehearsal can *measure* where the optimum sits rather than us guessing.

### `epsilon` is a fraction of the cost, and the window is narrow

Measured during P1 on a realistic batch — 256 against 256 cells over 955
genes in control-SD units, where the mean cost comes out at ~2.0:

| `epsilon` (× mean cost) | Sinkhorn iterations | effective partners per cell |
|---|---|---|
| 0.005 | did not converge in 300 | 4.5 |
| 0.01 | 36 | 15.9 |
| 0.05 | 3 | 209.5 |
| 0.2 | 2 | 252.8 |
| 1.0 | 2 | 255.5 |
| ∞ | — | 256 (pooled) |

Three things follow, and all three are now in the code:

1. **`epsilon` is expressed as a fraction of the batch's mean cost**, not as
   an absolute distance. The entire pooled-to-hard transition happens between
   0.005 and 0.05 of it — too narrow a window, and too dependent on a
   screen's noise level, for an absolute value to land in reliably on every
   screen.
2. **The swept grid moves to `{0.005, 0.01, 0.02, 0.05, ∞}`.** The grid
   originally proposed (§13.2, absolute) put three of its five values in the
   region where the coupling is already pooled, which would have spent three
   quarters of cycle C measuring the same thing.
3. **Below ~0.01 nothing converges affordably.** That is a property of the
   problem, not of the iteration ceiling. The barycentric target from a
   partly converged coupling is still a usable weighted average, so the
   solver reports its residual (`marginal_drift`) rather than pretending; a
   run at the concentrated end is visibly approximate instead of silently so.

---

## 5. The delta consistency penalty

The two decoder arms share weights (§4.5, `phase2.py:279-292`), but nothing
constrains the **difference** between them — and that difference is the only
quantity the six official metrics ever score. Each arm's MSE is dominated by
the baseline expression level, so the mapping can look healthy on both arms
while getting the rest-gene *change* badly wrong.

The OT coupling fixes this for free. It is computed on the panel, but it
pairs *cells*, so the same `P̂` applies to those cells' rest values:

```
T_i^rest = Σ_j P̂_ij · perturbed_rest_j                     # the paired truth, rest genes
observed change_i   = T_i^rest − control_rest_i
predicted change_i  = decode(perturbed panel_i) − decode(control panel_i)

delta loss = mean‖predicted change_i − observed change_i‖²  ÷  mean‖observed change_i‖²
```

masked to the genes the screen measures, and normalized by its own
no-change baseline so 1.0 means "predicted no change in the rest genes",
the same convention as everywhere else.

**Which perturbed panel feeds the decoder** is a config key,
`phase2.delta_through_prediction`:

* `false` (default): the decoder is fed the **OT-paired true** perturbed
  panel. The term then trains the mapping only, on changes, and keeps
  decoder error out of the perturbation module's gradient. This is exactly
  the quantity rehearsal variant 1 measures.
* `true`: the decoder is fed the **predicted** perturbed panel, training the
  composition end to end — which is what Phase 3 actually runs at inference.
  Worth measuring, but not first: it lets a bad perturbation prediction
  corrupt the mapping.

In pooled mode there is no coupling, so the term falls back to group means
(mean perturbed rest − mean control rest against the same for the decoded
arms). That keeps both modes runnable and makes the term independently
measurable before OT lands.

Weight: `phase2.loss_delta`, default 0.5 alongside `loss_mapping`.

### What P2 built, and the one number that makes it readable

The group-mean form is in `phase2.make_step`, and the evaluation reports

| key | what it says |
|---|---|
| `map_delta_ratio_to_no_change` | the change error, against predicting no change |
| `map_delta_noise_floor_ratio` | what a **perfect** predictor would score |

The floor exists because an observed change is a difference of two sampled
means and carries noise of its own. That noise inflates the ratio without
moving the loss's minimum — the expected squared error against `true + noise`
is still minimized at `true` — so the loss is sound and only the *reading*
would mislead. At the training draw of 16 cells a perfect predictor would
score around 0.76; `phase2.delta_eval_cells` is therefore 128 for scoring,
where the floor is nearer 0.28. It is estimated directly, from two
independent control draws whose mean difference is pure sampling noise of the
same size.

A ratio at the floor means the mapping has learned everything these cells can
show it. A ratio at 1.0 means it has learned nothing about the change,
whatever the per-state MSEs say.

---

## 6. Considered and dropped: an autoregressive gene decoder

An scGPT-style generative decoder over the rest genes was considered for
both arms of Model 2 and is **not** being built. Two readings, both answered:

**Strictly gene-by-gene.** The submission is 3 × 300 × 400 = 360,000 cells ×
17,578 rest genes = **6.3 × 10⁹ sequential decode steps**, each a forward
pass over a growing context. That is not affordable in the ~12-hour budget
of §0, by orders of magnitude.

**scGPT's actual scheme** — iterative masked refinement, all masked genes
predicted in parallel each round, K ≈ 3–10 rounds — *is* affordable, at K×
the current decoder. But it buys nothing that can be scored here:

* `pds_cosine` and `expr_mse_unbiased_capped_norm` operate on a **group-sum
  pseudobulk**. Gene-gene dependence within a cell averages out and does not
  change the pseudobulk.
* All four DE metrics are a **per-gene Wilcoxon test**. The statistic for
  gene *g* depends only on gene *g*'s marginal across cells. Cross-gene
  dependence within a cell does not enter it.

So none of the six metrics can see the correlation structure autoregression
exists to produce. K× the compute, zero expected score.

The gene-query decoder already in place *is* the generative head — "given a
gene's embedding, predict its value" — simply non-autoregressive, and
parallel over genes, which is what makes 17,578 output genes tractable.

**If cross-gene realism is wanted later** (for the biology, not for the
score), the cheap route is stochastic **latents**: a conditional VAE over the
64-latent bottleneck. Sampling the latent per cell induces correlated
variation across all genes at once, at essentially no extra cost, and keeps
the decoder parallel. Deferred until OT is measured.

---

## 7. Cross-assay transfer: leveraging Model 1 in Model 2

**The challenge is already a different assay from Replogle** — different lab,
different protocol, ~20,000 UMIs against Replogle's 11,000–15,000. So
cross-assay transfer is not preparation for hypothetical future data; it is
the task being scored, and the pipeline has never measured it.

### 7.1 The principle: split by what a new assay's controls reveal

| what varies across assays | visible in its controls? | mechanism |
|---|---|---|
| depth, batch, dynamic range, cell state | **yes** | amortize — compute it from the controls |
| perturbation modality (KO vs KDi vs drug) | **no** — controls look identical either way | a discrete type token |
| effect magnitude for that modality | no | a fitted per-assay scale |

Amortize what the controls show; use a discrete token for what they do not;
fit a scale for what neither gives. That split decides the three changes
below, and it is the line the defense can be argued along.

### 7.2 What is already right

The context vector is **amortized, not learned per context**:

```python
# core.py:225-227
pooled  = self.value_encoder(context_profile).mean(dim=-2)
latents = self.film(latents, self.context_encoder(pooled))
```

A brand-new context needs *zero* fitted parameters — the first row of the
table is already solved, and it is why §4.7's Phase 3 works at all. Phase 2
also warm-starts from the Phase 1 core, adds per-phase adapters, and is held
back by L2-SP and Phase 1 replay (§4.3). The transfer machinery exists.

### 7.3 The gap: one type token for every assay

```python
core.py:146   self.knockdown_type = nn.Parameter(torch.zeros(dim))   # ONE vector
core.py:209   pert = self.vocabulary(target_idx, role=TARGET) + self.knockdown_type
```

A single constant is added to every perturbation token in every phase, so
LINCS CRISPR **knockout** and Replogle/challenge CRISPR **interference** are
indistinguishable to the model. CLAUDE.md §3.3 states that directions and
affected genes transfer between them but *magnitudes and the target gene's
own level do not* — and the architecture currently has nowhere to put that
distinction. Phase 1's knockout evidence is therefore absorbed into Phase 2
as though it were knockdown evidence.

### 7.4 Change 1 — a type vocabulary, in the gene vocabulary's shape

```
type(assay) = base + δ_assay          δ initialized to zero, L2-regularized
```

The `W·prior + δ` pattern of §4.1, one level up. `base` is trained by every
phase and carries what all knockdown-like perturbations share — **that is
Model 1's contribution to Model 2, as an explicit named quantity** rather
than an assumption buried in a warm start. `δ_assay` absorbs what is
specific to knockout, and is penalized toward zero so the two cannot quietly
diverge until nothing transfers at all. A new assay is one new `δ` row,
initialized from the nearest existing type.

This is a faithful reading of §4.2's "a learned knockdown-type embedding"
rather than an added component: a type embedding with exactly one type is
the degenerate case.

**Built in P2.** `core.knockdown_type` becomes `pert_type_base` plus a
`pert_type_delta` row per assay, penalized toward zero by
`train.type_delta_l2`. The assay is named by `phase1.pert_type`
(`crispr_ko`), `phase2.pert_type` and `phase3.pert_type` (both `crispri` —
they are the same modality, and that sharing *is* the transfer). Every call
site that builds a perturbation token must pass one; the model raises rather
than defaulting, because a silent default would let a knockout be treated as
a knockdown, which is the exact failure the vocabulary exists to prevent.

One property worth knowing, found by a test: the blocks normalize tokens, so
a type offset that is **constant** across the embedding's dimensions is
removed before the latents ever see it. A type row speaks through its
direction, never through a uniform shift.

Not oversold: the module currently *under*-predicts, so importing oversized
knockout magnitudes is not the active failure. This is a correctness fix and
an assay-generalization mechanism, not a predicted leaderboard gain.

### 7.5 Change 2 — a per-assay effect scale

The third row of the table already has a knob: the generator's
`effect_scale`, fitted on the rehearsal (§4.7). But it is **one global number
fitted on Replogle and applied to the challenge** — an unstated cross-assay
assumption sitting in the middle of the submission path. Making it per-assay,
fitted per rehearsal dataset and recorded in `phase3_policy.json` alongside
the assay it came from, turns a hidden assumption into a reported number.
Nearly free, and it is the honest form of what is already happening.

### 7.6 Change 3 — measure what Phase 1 is actually worth

Phase 2 has two arms, `core_unfrozen` and `core_frozen`, and **both are
warm-started from Phase 1**. There is no from-scratch arm, so Phase 1's
contribution has never been measured. With the module sitting at 1.036 it is
entirely possible that Phase 1 contributes nothing, and we would not know.

A third arm, `core_scratch`, initialized from the priors without the Phase 1
checkpoint. One config flag, one extra arm in `train_arm`, and the
three-phase story in the defense stops being a claim and becomes a number.
It is the cheapest high-value item in this document, and it runs in the same
server cycle as change 1 — so **one cycle answers both transfer questions**.

If the scratch arm matches the warm-started arms, changes 1 and 7.7 need
rethinking before anything is built on top of them, which is why this comes
first rather than last.

**Built in P2**, behind `train.phase1_contribution_ablation` (default on).
`phase2/metrics.json` gains `phase1_contribution`: a per-metric
`scratch − warm` difference on ratios where lower is better, so a positive
number means Phase 1 helped. The summary prints it as its own section.

**The comparison is not symmetric, and the report says so.** A warm arm gives
`replay_fraction` of its steps to Phase 1's objective and the scratch arm
gives none, so the scratch arm gets *more* training on Phase 2's own
objective. `n_phase2_steps` is recorded per arm. The bias runs against Phase
1, so a small positive contribution is a stronger result than it looks, and a
small negative one is weaker. Equalizing it by giving the scratch arm replay
too would put back through the side door exactly what the ablation removes.

### 7.7 Change 4 — a cross-assay rehearsal variant

Rehearsal variant 2 is cross-**context**, same assay. The cross-**assay**
analogue, on data already on disk:

> Train the perturbation module on LINCS only. Adapt to a Replogle screen
> using **its controls only**. Predict its perturbed panel for the targets
> the two share, and score.

That is literally "a new assay arrives, adapt with controls only, measure" —
an extension of variant 2 rather than a new component, and a more honest
analogue of the challenge than variant 2 is, for the reason at the top of
this section. It produces **evidence, not score**: it cannot improve the
submission, so it is built behind a config flag
(`rehearsal.cross_assay: false`) and never slows a deadline run.

---

## 8. What changes, file by file

| file | change |
|---|---|
| `vccp/train/ot.py` | **new.** Log-domain Sinkhorn (~40 lines), cost matrix, barycentric projection. Pure tensor code, no new dependency, runs on the GPU beside the batch. |
| `vccp/phases/phase2.py` | `make_step`: the perturbation half draws `ot_batch_cells` control and perturbed cells per target for `ot_targets_per_step` targets, builds the coupling, and takes the loss against the barycentric target. The same coupling feeds the §5 delta term. `perturbation_mode: pooled \| percell` selects the old path. |
| `vccp/config.py` | `phase2.perturbation_mode`, `ot_epsilon`, `ot_batch_cells`, `ot_targets_per_step`, `ot_iterations`, `loss_delta`, `delta_through_prediction`. |
| `vccp/predict/run.py` | per-cell prediction: each drawn control cell goes through the perturbation module and the mapping individually; the result is a fold-change **matrix** (cells × genes), not a vector. |
| `vccp/predict/generator.py` | `generate_counts` accepts a per-cell fold-change matrix as well as a vector; thresholding and scaling apply elementwise. The knockdown exemption becomes a column mask. |
| `vccp/rehearsal/variants.py` | the cross-context variant runs both modes so the report compares them directly. |
| `vccp/phases/adapt.py`, `phase3.py` | unchanged — the mapping is already per cell and the policy already forbids adapting the perturbation module. |
| `vccp/models/core.py` | `knockdown_type` becomes a type vocabulary, `base + δ_assay` (§7.4), with the δ penalty beside the gene vocabulary's. |
| `vccp/phases/phase2.py` (2) | a third arm, `core_scratch`, initialized from the priors without the Phase 1 checkpoint (§7.6). |
| `vccp/rehearsal/variants.py` (2) | the cross-assay variant, LINCS → Replogle, behind `rehearsal.cross_assay` (§7.7). |
| `vccp/predict/run.py` (2), `phase3.py` | `effect_scale` resolved per assay rather than globally, and the assay it was fitted on recorded in `phase3_policy.json` (§7.5). |
| `vccp/phases/phase1.py` | **unchanged.** LINCS is 688 bulk wells; there are no cells to be per-cell about (§2.1). Phase 1 stays profile-level, keeps its `sig`/`gmt` heads and its two-step, and the shared core absorbs both. |

---

## 9. Prediction and the generator

Today, per (context, target): one forward pass, one fold-change vector,
applied to 400 resampled control cells.

Proposed: draw the 400 control cells first, push all 400 through the
perturbation module **as one batch**, map each to the rest genes, and form
400 fold-change vectors — each cell against **its own** control values. The
generator then applies row *i* to cell *i*.

Cost: the same number of forward passes as today in batch terms (batch 400
instead of batch 1), so wall clock should rise by a small factor, not by
400×. Memory is governed by the existing `model.gene_chunk`. I will measure
it on `mini_data` before the server run rather than assert it.

A consequence worth stating plainly: the submitted cells stop being
"resampled control cells with a shared multiplier" and become 400 distinct
model outputs. That is the point, and it is also what makes the expression
and discrimination metrics able to reward the model rather than the
resampling.

---

## 10. Step 0 — the capacity check

The module could not fit its **training** targets (1.036). OT changes what it
is supervised against, not whether it can learn, so before building any of
the above:

> Train the current pooled module on **8 targets from one screen** with
> validation disabled, for enough steps to overfit, and report the training
> ratio.

`scripts/capacity_check.py` implements it. It reports three numbers per
checkpoint — the error ratio, the correlation, and **the size of the
predicted change against the size of the true one**, which is the number
that catches a module that looks like it is training while converging to
predicting nothing.

**Result on `mini_data`** (8 targets from K562_gwps, 600 steps, every
parameter trainable, warm-started from the Phase 1 core):

| step | ratio ÷ no-change | pearson | \|pred\| ÷ \|true\| |
|---|---|---|---|
| 1 | 2.530 | −0.07 | 1.18 |
| 200 | 0.745 | +0.51 | 0.51 |
| 400 | 0.154 | +0.93 | 0.86 |
| 600 | **0.040** | **+0.98** | **1.00** |

**Verdict: can-fit.** The module drove training error to 4% of no-change and
predicted changes of the right magnitude and direction. So the architecture,
the perturbation token, the conditioning and the optimiser are all capable
of representing a response — the failure at 1.003 / 1.036 on the real
training targets is about signal and supervision, not about the module.

**Result on the server** (8 targets from K562_essential, 2,000 steps, 35
seconds on one GPU):

| step | ratio ÷ no-change | pearson | \|pred\| ÷ \|true\| |
|---|---|---|---|
| 1 | 9.671 | +0.01 | 3.20 |
| 500 | 0.078 | +0.960 | 0.936 |
| 1000 | 0.034 | +0.993 | 0.987 |
| 2000 | **0.000** | **+1.000** | **1.000** |

**Verdict: CAN-FIT**, and more decisively than on mini — complete
memorization, with the predicted change the exact size of the true one. The
architecture, the perturbation token, the context conditioning and the
optimiser are all sound on the real panel with the real screens.

**What this does and does not establish.** Memorizing 8 targets proves the
module has somewhere to *put* a response — it can store the answer in the
target-role embedding, which is a per-target parameter. It rules a broken
module **out**. It does not rule generalization **in**, and it was never
meant to: it is a negative control that had to pass before the redesign was
worth starting. It passed, so P1 onward proceeds.

(The `Memory Efficient attention defaults to a non-deterministic algorithm`
warning in that log is expected: `runtime.set_determinism` uses
`warn_only=True`, so attention warns rather than raising, and the run stays
reproducible everywhere else.)

---

## 11. What could go wrong

* **The module learns the identity.** With a per-cell input and a per-cell
  target that is "a nearby cell", predicting `output = input` scores well.
  The head already predicts a change, so I will report the change magnitude
  against the no-change baseline every step — the same diagnostic that
  exposed the current failure.
* **ε is hard to set.** Mitigated by it spanning the pooled formulation at
  one end; the rehearsal sweeps it.
* **~~Distances concentrate, and the coupling carries no information.~~**
  **Measured in P3 and the premise holds.** This was the most serious risk in
  the document. P1 found that in 955 dimensions the pairwise costs between
  *independent noise* vectors span 1.69 to 2.47 around a mean of 2.04 — every
  control cell nearly equidistant from every perturbed one, a uniform
  coupling whatever `epsilon` says, and OT silently degenerating into the
  pooled objective it exists to replace.

  `scripts/coupling_check.py` asked the question on real cells before any
  training was built on the answer. On `mini_data` (K562_gwps, 256 cells a
  side, 716 panel genes):

  | | cost spread | noise alone | `target_spread` @ ε 0.005 | library r |
  |---|---|---|---|---|
  | control vs perturbed | **0.246** | 0.053 | **0.57** | 0.54–0.74 |
  | control vs control (null) | 0.244 | 0.053 | 0.52 | 0.60–0.81 |

  Real cells carry about five times the cost spread that noise would give,
  the barycentric targets vary 57% as much as the perturbed cells themselves,
  and the pairing tracks sequencing depth strongly — the mechanism §4 claims.
  Verdict `informative`.

  **The caveat the check also shows.** The control-against-control null is
  nearly identical to the perturbed arm. That is expected — the coupling is
  built before any prediction, so it can only match on nuisance — but it
  means the method's value rests entirely on the argument that removing
  nuisance variance sharpens the residual, not on the coupling finding the
  perturbation. Worth running on the server before P5:

  ```bash
  python scripts/coupling_check.py --config configs/server.yaml
  ```
* **The coupling matches on depth and the residual is depth.** This is the
  intended mechanism, but if library size dominates completely the module
  may learn a depth correction and nothing else. Diagnostic: the correlation
  between predicted change and library-size difference, reported per step.
* **The delta term fights the level term.** A mapping tuned for changes may
  drift on absolute level, which the rest-gene predictions still need.
  Both are reported separately in `phase2/metrics.json`, and `loss_delta`
  is the knob.
* **Prediction gets slow or large.** Measured on mini before the server.
* **It does not help.** Entirely possible. The rehearsal's cross-context
  variant, on the leaderboard scale, is the arbiter, and the pooled arm
  stays in the codebase as the control.

---

## 12. How we will know

The comparison is `cross_context`, both modes, on the leaderboard scale
(0 = organizers' baseline, 1 = split-half replicate) — the reading that
disagreed with my calibration objective and turned out to be the one the
leaderboard agreed with.

Today's numbers to beat, per dataset (method arm, default generator
settings): **+0.027** (K562_essential), **−0.001** (K562_gwps), **−0.226**
(rpe1). Plus the Phase 2 training ratio: anything ≥ 1.0 means the module
still is not learning, whatever the rehearsal says.

---

## 13. Decisions taken

1. **Step 0 first.** Done on mini and on the server: `can-fit` (§10). The server confirmation
   runs in parallel with P1–P2 and gates P3.
2. **`ε` sweep** is `{0.005, 0.01, 0.02, 0.05, ∞}` as a **fraction of the
   batch's mean cost**, revised from the absolute grid first proposed after
   P1 measured where the transition actually sits (§4). The rehearsal reports
   each, with `effective_partners` beside it so the sweep is read in terms of
   what the coupling did rather than what the config asked for.
3. **The pooled mode stays**, as the control arm and as a fallback. It costs
   one config branch, and `perturbation_mode` is a setting, not a code fork
   (§1.1's "one code path" is about mini vs server, not about a measured
   A/B).
4. **No autoregressive decoder** (§6). The gene-query decoder stays; a
   stochastic-latent VAE is the deferred option if cross-gene realism is
   wanted later.
5. **The delta consistency term is added** (§5), defaulting to the
   OT-paired-true-panel form.
6. **A perturbation-type vocabulary** replaces the single shared type
   constant (§7.4). Lands with P3, in the same file OT is already rewriting.
7. **A `core_scratch` arm** measures Phase 1's contribution (§7.6). Lands
   first, with P2, because its answer can change 6 and 7.7.
8. **`effect_scale` becomes per-assay** (§7.5), fitted per rehearsal dataset.
9. **A cross-assay rehearsal variant**, opt-in (§7.7). Evidence for the
   defense, not score for the leaderboard, so it is the one item allowed to
   slip past the deadline.

## 14. Schedule

**Validation-round deadline: 2026-10-15** (extended from 09-30). The final
test round follows on **2026-10-22** with new contexts and a different
target list, which CLAUDE.md requires the pipeline to run on with only a
config change — so the freeze below is real, not a formality.

**The binding constraint is server cycles, not coding time.** §1.2 allows one
GPU process at a time, so server runs are strictly serial, and each costs
about a day of turnaround once launching and reporting back are counted. The
plan needs five.

| dates | code, verified on `mini_data` | server cycle |
|---|---|---|
| **Sep 24–27** | **P1** `train/ot.py` + tests · **P2** delta term, group-mean form, pooled mode · **#1** type vocabulary · **#3** `core_scratch` arm | **A** — capacity check, then Phase 2 with three arms and the delta term |
| **Sep 28–Oct 2** | **P3** per-cell mode + per-pair delta · **P4** per-cell prediction and generator | **B** — Phase 2, per-cell against pooled |
| **Oct 3–7** | **P5** rehearsal A/B harness · **#2** per-assay `effect_scale` · **#4** cross-assay variant, opt-in | **C** — rehearsal, ε swept, mode chosen |
| **Oct 8–11** | integration and fixes; **code freeze Oct 11** | **D** — full `all`, submission, validate, package |
| **Oct 12–15** | buffer only | **E** — second submission with whatever D taught us |

### Why this order

**Cycle A is the highest-value cycle and it comes first**, because #3 answers
a question that changes the rest of the plan: if Phase 2 from scratch matches
Phase 2 warm-started, Phase 1 contributes nothing and §7.4 and §7.7 need
rethinking *before* they are built on rather than after. The same cycle says
whether the delta term alone moves the rest-gene change ratio — a result
that arrives before OT lands and de-risks everything downstream.

#1 moves earlier than first proposed for the same reason: it is ~20 lines in
`core.py` and `phase2.py`, which P3 rewrites anyway, and pairing it with #3
means one server run answers both transfer questions.

### Exit conditions

| | content | exit condition |
|---|---|---|
| **P0** | capacity check on 8 targets | ✅ `can-fit` on mini **and** on the server (§10) |
| **P1** | `train/ot.py` + unit tests (ε → ∞ reproduces the pooled target; ε → 0 approaches hard assignment; marginals uniform) | ✅ 16 tests green; ε made relative and the grid revised (§4) |
| **P2** | the delta term, group-mean form, pooled mode; `core_scratch` arm; type vocabulary | ✅ Phase 2 reports `map_delta_ratio_to_no_change` with its noise floor, and `phase1_contribution` per metric |
| **P3** | `phase2.perturbation_mode: percell` + the per-pair delta term | ✅ premise checked on real cells first — `informative` (§11); both modes scored on both questions |
| **P4** | per-cell prediction and generator | `all` green on mini, runtime and memory measured |
| **P5** | rehearsal A/B, ε swept, per-assay scale | a table of leaderboard-scale numbers, both modes |
| **P6** | server run, submission | a leaderboard number to compare against −0.131 |

### Three things that could break it

1. **The ε sweep can eat cycle C.** Five ε values × three rehearsal datasets
   × two modes × three variants is a combinatorial trap. Bound it to the
   **cross-context variant, method arm, one dataset, 100 held-out targets** —
   five runs, not ninety — then confirm the winner on the other two datasets.
2. **Turnaround latency.** Five serial cycles in 21 days leaves no room for a
   cycle that sits unlaunched for three days. This is the likeliest way the
   schedule slips, and it is not something code can fix.
3. **Oct 22.** Whatever is in the repo at the Oct 11 freeze has to be a
   working pipeline for the final round, not a half-landed refactor.

**If OT does not beat pooled in cycle C**, we submit the pooled arm carrying
the delta term and #1–#3. That is what the pooled mode stays in the codebase
for (§13.3), and it is why no cycle depends on OT succeeding.
