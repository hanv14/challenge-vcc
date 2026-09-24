# Per-cell perturbation module — design

**Status: agreed. P0 answered on `mini_data`; implementation not started.**

The current perturbation module predicts one mean effect per (context,
target). This replaces it with one that maps **a cell** to **that cell
perturbed**, supervised by entropic optimal transport between drawn control
cells and drawn perturbed cells.

Read §1 for why, §2 for the method end to end, §10 for what could go wrong,
and §12 for what was decided.

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

### Why matching on "nuisance" is the point

The cost is dominated by sequencing depth and cell cycle, not by the
perturbation. That is the mechanism, not a flaw: pairing like with like on
those factors means the residual difference between paired cells is closer
to the perturbation than the difference between a random control and a
random perturbed cell. This is what CellOT and its relatives exploit, and it
is precisely what pooling throws away.

### ε spans the whole design space

* `ε → ∞`: uniform coupling, every control matched to everything equally,
  `T_i` → the perturbed pseudobulk. **This is today's loss.**
* `ε → 0`: hard assignment, `T_i` → one perturbed cell.
* in between: local averages.

So `phase2.ot_epsilon` is one config key that moves continuously from the
current formulation to hard OT, the pooled arm stays alive as a control, and
the rehearsal can *measure* where the optimum sits rather than us guessing.

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

## 7. What changes, file by file

| file | change |
|---|---|
| `vccp/train/ot.py` | **new.** Log-domain Sinkhorn (~40 lines), cost matrix, barycentric projection. Pure tensor code, no new dependency, runs on the GPU beside the batch. |
| `vccp/phases/phase2.py` | `make_step`: the perturbation half draws `ot_batch_cells` control and perturbed cells per target for `ot_targets_per_step` targets, builds the coupling, and takes the loss against the barycentric target. The same coupling feeds the §5 delta term. `perturbation_mode: pooled \| percell` selects the old path. |
| `vccp/config.py` | `phase2.perturbation_mode`, `ot_epsilon`, `ot_batch_cells`, `ot_targets_per_step`, `ot_iterations`, `loss_delta`, `delta_through_prediction`. |
| `vccp/predict/run.py` | per-cell prediction: each drawn control cell goes through the perturbation module and the mapping individually; the result is a fold-change **matrix** (cells × genes), not a vector. |
| `vccp/predict/generator.py` | `generate_counts` accepts a per-cell fold-change matrix as well as a vector; thresholding and scaling apply elementwise. The knockdown exemption becomes a column mask. |
| `vccp/rehearsal/variants.py` | the cross-context variant runs both modes so the report compares them directly. |
| `vccp/phases/adapt.py`, `phase3.py` | unchanged — the mapping is already per cell and the policy already forbids adapting the perturbation module. |
| `vccp/phases/phase1.py` | **unchanged.** LINCS is 688 bulk wells; there are no cells to be per-cell about (§2.1). Phase 1 stays profile-level, keeps its `sig`/`gmt` heads and its two-step, and the shared core absorbs both. |

---

## 8. Prediction and the generator

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

## 9. Step 0 — the capacity check

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

That is what makes the per-cell OT redesign worth building rather than a
bug hunt. **It still needs confirming on the server data**, where the
targets are 4,968 rather than 51 and the panel is the real one:

```bash
source scripts/server_env.sh
python scripts/capacity_check.py --config configs/server.yaml --targets 8 --steps 2000
```

P1 and P2 are low-risk and independent of the verdict, so they proceed in
parallel with that run. If the server comes back `cannot-fit` or
`collapses`, P3 onward stops until it is understood.

---

## 10. What could go wrong

* **The module learns the identity.** With a per-cell input and a per-cell
  target that is "a nearby cell", predicting `output = input` scores well.
  The head already predicts a change, so I will report the change magnitude
  against the no-change baseline every step — the same diagnostic that
  exposed the current failure.
* **ε is hard to set.** Mitigated by it spanning the pooled formulation at
  one end; the rehearsal sweeps it.
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

## 11. How we will know

The comparison is `cross_context`, both modes, on the leaderboard scale
(0 = organizers' baseline, 1 = split-half replicate) — the reading that
disagreed with my calibration objective and turned out to be the one the
leaderboard agreed with.

Today's numbers to beat, per dataset (method arm, default generator
settings): **+0.027** (K562_essential), **−0.001** (K562_gwps), **−0.226**
(rpe1). Plus the Phase 2 training ratio: anything ≥ 1.0 means the module
still is not learning, whatever the rehearsal says.

---

## 12. Decisions taken

1. **Step 0 first.** Done on mini: `can-fit` (§9). The server confirmation
   runs in parallel with P1–P2 and gates P3.
2. **`ε` sweep** starts at `{0.01, 0.05, 0.2, 1.0, ∞}` in cost units (the
   cost is normalized per gene, so these are comparable across panels), with
   the rehearsal reporting each.
3. **The pooled mode stays**, as the control arm and as a fallback. It costs
   one config branch, and `perturbation_mode` is a setting, not a code fork
   (§1.1's "one code path" is about mini vs server, not about a measured
   A/B).
4. **No autoregressive decoder** (§6). The gene-query decoder stays; a
   stochastic-latent VAE is the deferred option if cross-gene realism is
   wanted later.
5. **The delta consistency term is added** (§5), defaulting to the
   OT-paired-true-panel form.

## 13. Milestones

| | content | exit condition |
|---|---|---|
| **P0** | capacity check on 8 targets | ✅ `can-fit` on mini; server run outstanding |
| **P1** | `train/ot.py` + unit tests (ε → ∞ reproduces the pooled target; ε → 0 approaches hard assignment; coupling rows sum to 1) | `pytest` green |
| **P2** | the delta consistency term, group-mean form, in pooled mode | Phase 2 reports a rest-gene *change* ratio; measured against today |
| **P3** | `phase2.perturbation_mode: percell` + the per-pair delta term, both modes training on mini | training ratio reported for both |
| **P4** | per-cell prediction and generator | `all` green on mini, runtime and memory measured |
| **P5** | rehearsal A/B, ε swept | a table of leaderboard-scale numbers, both modes |
| **P6** | server run, submission | a leaderboard number to compare against −0.131 |
