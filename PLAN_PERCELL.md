# Per-cell perturbation module — design for review

**Status: proposal. No code written yet.**

The current perturbation module predicts one mean effect per (context,
target). This proposes replacing it with one that maps **a cell** to **that
cell perturbed**, supervised by entropic optimal transport between drawn
control cells and drawn perturbed cells.

Read §1 for why, §7 for what could go wrong, and §9 for the three decisions
I need from you.

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
control cells into one profile per context and predicts one mean response.
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

## 2. What the module becomes

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

## 3. The loss: entropic OT with a barycentric target

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

## 4. What changes, file by file

| file | change |
|---|---|
| `vccp/train/ot.py` | **new.** Log-domain Sinkhorn (~40 lines), cost matrix, barycentric projection. Pure tensor code, no new dependency, runs on the GPU beside the batch. |
| `vccp/phases/phase2.py` | `make_step`: the perturbation half draws `ot_batch_cells` control and perturbed cells per target for `ot_targets_per_step` targets, builds the coupling, and takes the loss against the barycentric target. `perturbation_mode: pooled \| percell` selects the old path. |
| `vccp/config.py` | `phase2.perturbation_mode`, `ot_epsilon`, `ot_batch_cells`, `ot_targets_per_step`, `ot_iterations`. |
| `vccp/predict/run.py` | per-cell prediction: each drawn control cell goes through the perturbation module and the mapping individually; the result is a fold-change **matrix** (cells × genes), not a vector. |
| `vccp/predict/generator.py` | `generate_counts` accepts a per-cell fold-change matrix as well as a vector; thresholding and scaling apply elementwise. The knockdown exemption becomes a column mask. |
| `vccp/rehearsal/variants.py` | the cross-context variant runs both modes so the report compares them directly. |
| `vccp/phases/adapt.py`, `phase3.py` | unchanged — the mapping is already per cell and the policy already forbids adapting the perturbation module. |
| `vccp/phases/phase1.py` | **unchanged.** LINCS is one row per (cell line, target, time); there are no cells to be per-cell about. Phase 1 stays profile-level and the shared core absorbs both. |

---

## 5. Prediction and the generator

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

## 6. Step 0 — a capacity check before any of this

The module cannot fit its **training** targets (1.036). OT changes what it is
supervised against, not whether it can learn. Before building any of the
above:

> Train the current pooled module on **8 targets from one screen** with
> validation disabled, for enough steps to overfit, and report the training
> ratio.

* **It reaches ≪ 1.0** → the module can learn; the problem is signal, scale
  or optimisation, and OT is worth building.
* **It stays ~1.0** → there is a bug or a capacity limit in the perturbation
  path, and OT would inherit it. Fix that first.

Half a day, decisive either way, and it stops us building on a broken
foundation. I would not skip it.

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
the perturbation token and the optimiser are all capable of representing a
response — the failure at 1.003 / 1.036 on the real training targets is
about signal or supervision, not about the module.

That is what makes the per-cell OT redesign worth building rather than a
bug hunt. **It still needs confirming on the server data**, where the
targets are 4,968 rather than 51 and the panel is the real one:

```bash
python scripts/capacity_check.py --config configs/server.yaml --targets 8 --steps 2000
```

---

## 7. What could go wrong

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
* **Prediction gets slow or large.** Measured on mini before the server.
* **It does not help.** Entirely possible. The rehearsal's cross-context
  variant, on the leaderboard scale, is the arbiter, and the pooled arm
  stays in the codebase as the control.

---

## 8. How we will know

The comparison is `cross_context`, both modes, on the leaderboard scale
(0 = organizers' baseline, 1 = split-half replicate) — the reading that
disagreed with my calibration objective and turned out to be the one the
leaderboard agreed with.

Today's numbers to beat, per dataset (method arm, default generator
settings): **+0.027** (K562_essential), **−0.001** (K562_gwps), **−0.226**
(rpe1). Plus the Phase 2 training ratio: anything ≥ 1.0 means the module
still is not learning, whatever the rehearsal says.

---

## 9. Decisions I need from you

1. **Step 0 first?** I recommend yes — half a day, and it tells us whether
   OT is worth building at all.
2. **`ε` default.** I would start the sweep at `{0.01, 0.05, 0.2, 1.0, ∞}`
   in cost units (the cost is normalized per gene, so these are comparable
   across panels), with the rehearsal reporting each.
3. **Does the pooled mode stay?** I recommend yes, as the control arm and as
   a fallback — it costs one config branch, and `perturbation_mode` is a
   setting, not a code fork (§1.1's "one code path" is about mini vs server,
   not about a measured A/B).

## 10. Milestones

| | content | exit condition |
|---|---|---|
| **P0** | capacity check on 8 targets | a number, and a go/no-go |
| **P1** | `train/ot.py` + unit tests (ε → ∞ reproduces the pooled target; ε → 0 approaches hard assignment; coupling rows sum to 1) | `pytest` green |
| **P2** | `phase2.perturbation_mode: percell`, both modes training on mini | training ratio reported for both |
| **P3** | per-cell prediction and generator | `all` green on mini, runtime and memory measured |
| **P4** | rehearsal A/B, ε swept | a table of leaderboard-scale numbers, both modes |
| **P5** | server run, submission | a leaderboard number to compare against −0.131 |
