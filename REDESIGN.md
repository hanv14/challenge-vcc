# REDESIGN — the final round of the Virtual Cell Challenge 2026

Written 2026-10-01, from `REDESIGN_PROMPT.md`, DECISIONS.md D1–D123 and the
measurements of D120–D123. Every number is cited to a D-entry, to
`docs/metrics.md`, or to an upload you reported. Where a number cannot be
known yet, the section says so and names the measurement that will give it.

**Status: approved 2026-10-01.** Built so far: the K562 lookup (rank 2, D124), off by default.
Measured since (UPLOADS.md): rank 1 closed, the model's specific part removed (D125);
the raw lookup transfers, +0.0098, the first upload above 0 (D126); per-gene shrinkage
refuted (D127). Since: L6, shared 1.5 with the raw lookup, +0.0189 (D129); the
lookup proxy passes its gate on the round's targets and picks soft SNR (D130).
Then: L7 (shared 1.0) +0.0286 and L8 (shared 0.5) **+0.0369** (D131, D134); SNR
lost in both forms, closing per-entry denoising (D132, D135).
`CLAUDE.md` stays the specification; §7 lists the changes I recommend to it.

---

## 0. Summary

1. **The leaderboard's 0 is an oracle.** It is the context's own mean
   perturbation response, computed from the hidden truth and emitted through
   resampled controls (docs §6a). "Nothing has beaten the baseline" is the
   expected state. Our uploads sit at −0.014 to −0.032 because they are
   mostly one profile shared by every target, close to that oracle (D117).
   **Anything above 0 must come from target-specific effects that hold in a
   cell line we never saw perturbed**, or from emitting the mean better than
   the organizers' baseline emits it.
2. **The offline instrument we relied on was broken twice.** The rehearsal
   read the wrong column (D120). On the right column, rpe1 does not track the
   uploads, member by member (D121): its targets are essential genes with
   strong responses, while the challenge's are not (D122).
3. **The instrument from here is the validation leaderboard itself.** It is
   open and uploads are unlimited. It scores the challenge's own lab and
   protocol, in three cell lines new to us, on 300 targets that overlap the
   final list. The scorer is deterministic and our generator is seeded, so
   two uploads that differ in one component measure that component exactly on
   that truth. Uploads become paired experiments (§5).
4. **The method becomes a composition of simple, testable parts:** a
   context mean response, a target-specific part, and the existing count
   generator, each with its own scale. The gene-token model stays as one
   source among several, and is kept only where an upload shows it earns its
   place. The rehearsal's Phase 2 retraining leaves the final round's
   critical path.
5. **First experiments need no code:** five uploads by `predict.override_*`
   on `runs/server_shared` (§5.3). The first experiment that needs code is the
   K562 lookup, the single most decisive test of whether anything
   target-specific transfers (§4, rank 2).

---

## 1. The metrics, mathematically

Notation: for target *p* in one context, a predicted group of `cells_per_pert`
(400) cells, the real group of about 461 cells (D122), and the real controls
(18,400). The six members exclude the target's own gene; discrimination
excludes every target gene of the panel. Two spaces are used:

* **group pseudobulk** `b_p = log1p(5e4 · P_p / ΣP_p)`, where `P_p` is the
  group's summed counts (docs §1.2). It depends only on the group's total
  counts, not on how they are spread across cells;
* **per-cell DE**: per-cell normalized expression, a Wilcoxon test of the
  group against the real controls, BH per target, significance at
  p_adj < 0.05, genes kept only if the control reaches 5 CPM. The fold change
  is the ratio of arithmetic means (docs §1.4).

### 1.1 The six members

| member | depends on | maximized by | oracle mean (0 end), val A | replicate (1 end), val A | leaderboard shape |
|---|---|---|---|---|---|
| `pds_cosine` | group sums: δ̂_p = b̂_p − b_ctrl, ranked by cosine among all real δ_q | δ̂_p pointing where δ_p differs from the other targets | 0.530 (B 0.528, C 0.510) | ≈ 0.98 (floor −b/(r−b) = −1.17) | linear, no floor |
| `expr_mse_unbiased_capped_norm` | group sums; Σ_p (squared error − sampling credit) / Σ_p (real effect − sampling) | b̂_p = E[b_p], with realistic dispersion | 0.986 | unknown on val (rpe1: 0.070) | **clamped to [0, 1]** |
| `de_wilcoxon_lfc_nmae` | predicted log2FC on the **reference's** significant genes only | the right magnitude where the truth is significant; under uncertainty, the posterior median (shrinkage) | ≈ 0.96 (no change ≈ 1.0) | ≈ 0.35 | linear to −6 |
| `..._direction_fidelity_yield_raw` | k / max(n_pred, N_conf): right-signed calls over the larger of our calls and the truth's | exactly as many calls as the truth (N_conf ≈ 343 on val A), all right-signed | ≈ 0.5 | ≈ 0.82 (floor −1.58) | linear, no floor |
| `..._direction_reach_raw` | the reference's significant genes ranked by **our** p-values; deepest depth whose signs are ≥ 90% right | our most significant genes are those whose sign we are surest of | 0.042 | 0.904 | linear, no floor |
| `de_wilcoxon_sig_jaccard` | overlap of our significant set with the truth's, over their union | the truth's set, at the truth's size | 0.033 | 0.43 | linear, no floor |

(Sources: docs §2.3, §3, §4.3, §4.5, §6, §6.0, §6.1; val A values are the
docs' measurements on the official bundles; cell-eval2 0.16.0's catalog for
the shapes. The val B/C ends differ; the leaderboard does not publish them.)

What follows for the method:

* **pds is scale-free.** Cosine ignores the size of δ̂_p, so any correct
  target-specific *direction* helps, however small, as long as it survives
  the resampling noise of a 400-cell group. A part shared by all targets adds
  no discrimination and dilutes it. Every upload scored −0.004 to −0.016 here:
  no discrimination at all.
* **nmae only looks where the truth is significant.** A change predicted on
  other genes costs nothing here. On the significant genes, the
  expected-error-minimizing prediction is shrunk toward 0 when uncertain.
* **fid and Jaccard want the truth's number of calls.** Too few calls and fid
  collapses (v6 −1.34, v8 −0.83). Too many and Jaccard and fid's precision
  fall. The number of calls is set by the size of our predicted changes
  against cell-to-cell dispersion.
* **reach wants our most confident genes to be right in sign.** We control the
  ranking: a gene's p-value in our cells is set by the size of its predicted
  change. So **the size of a predicted change should rise with our confidence
  in its sign**, not only with the expected effect. A mean-response
  prediction ranks generic genes first, and their sign is wrong for many
  targets, which is why the baseline reads 0.042 against a replicate's 0.904.
  It is the largest span of any member.
* **expression has no downside on the leaderboard** (clamped at 0, D113's
  open item closed in D120). All six uploads read exactly 0: at or worse than
  the oracle. Its upside needs target-specific accuracy on the strongest
  perturbations, because it is a ratio of sums.

### 1.2 Finite cells and count noise

* **Our DE power is slightly below the truth's.** 400 predicted cells against
  461 real ones, both tested against the same 18,400 controls: about √(400/461)
  ≈ 0.93 of the reference's z-statistic for the same true effect. For the
  same size of predicted change, we call fewer genes than the truth does.
* **Our cells are resampled real controls, changed in count space** (binomial
  thinning, stochastic rounding). That keeps Poisson-scale dispersion but adds
  none of a response's own heterogeneity: knockdown that varies between
  cells, shifts in cell-cycle state. A Wilcoxon test sees distribution shape,
  so for the same mean change our cells can be more or less significant than
  real ones. The oracle ladder's `own_target` arm measures that loss (§5.2).
* **Each target gets a fresh control draw** (§4.7, #348). In pds that is
  independent noise per target; in expression it earns the sampling credit.

### 1.3 Where the points are: the decomposition

The leaderboard places every prediction relative to an oracle, so the
decomposition the brief asks for reads:

| prediction | leaderboard score | source |
|---|---|---|
| no change (control resample) | about −0.3 to −0.5, estimated from the members' floors; "almost nothing" scored −0.242 (v6) | docs §6, D113 |
| our shared-dominated uploads | −0.014 (v4), −0.032 (shared 2 / specific 1) | D113, D118 |
| the oracle mean, emitted by the organizers | **0**, by definition | docs §6a |
| the oracle mean emitted *better* (sharper calls) | unknown; upper end small. v5 already beat the baseline on fid (+0.009) and reach (+0.004) with an oversized shared profile | oracle ladder `own_mean_x2`; val uploads |
| + target-specific effects that transfer | unknown; this is the open question | oracle ladder `own_mean+src_res`; the K562 lookup upload |
| a half-depth split-half replicate | 1; a perfect submission ≈ 1.4–1.6 | docs §6c-bis |

**Mean response versus target-specific.** The mean response is worth the
distance from no change to 0, about 0.3–0.5, and we already have nearly all
of it. **The whole 0 → 1 gap is target-specific**, by construction. How much
of it can transfer to a new cell line cannot be read from the data we have,
because no dataset gives a new cell line on challenge-like targets (§3). It is
measured three ways, from most to least direct:

1. the K562 lookup on the validation leaderboard: 272 of the 300 validation
   targets, the challenge's own cells;
2. the oracle ladder on rpe1: essential targets, a cell line new to K562;
3. LINCS across ten lines: landmark genes, knockouts.

---

## 2. The problem, precisely

**Input at prediction time, per context:** about 18,400 control cells in raw
counts (about 20,000 UMIs each), the target list, `cells_per_pert`, and
everything learned before. **Output:** for each target, 400 cells in raw
counts over the 18,533 genes.

**What the controls alone identify:** baseline expression and depth per gene;
dispersion; which genes can be called at all (the 5 CPM gate); co-expression
modules in this cell line; the 46 control guides' spread.

**What must transfer:**

* **the context's mean response.** Not in the controls. Transferred from other
  lines (Replogle K562 or RPE1, LINCS), possibly adjusted by control
  expression. LINCS gives ten lines to test whether a control-based
  adjustment helps (measurement 4).
* **each target's specific response in this context.** Not in the controls.
  Transferred from K562 genome-wide (272/300 validation targets), LINCS
  (82/300), or from similar targets. Its sign, its affected genes, and its
  size can each transfer to different degrees.
* **the size of responses in this lab and protocol.** Deeper cells than
  Replogle (about 20,000 against 11,000–15,000 UMIs), different knockdown
  strengths. Calibrated on the validation leaderboard; carried to D/E/F as an
  assumption.

**Identifiability limits.**

* About 10,000 genes are measured in no perturbation dataset (D12). Their
  response can only be inferred through their co-expression with measured
  genes in the new context's controls, which assumes the response follows
  the modules. That cannot be tested before the final round, and only
  indirectly on the leaderboard.
* Without perturbed cells in the new context, no data separates "this target
  acts differently here" from "our estimate is noisy". Only population-level
  transfer rates are identifiable (how well K562 predicts another line, on
  average), not per-context ones.
* The 28 validation targets in no Replogle screen (D122) have only priors and
  LINCS (if covered).
* The final round's targets overlap the validation 300, but how many, and how
  well they are covered, is unknown until the release.

---

## 3. Data inventory against the problem

| source | lines | targets | genes | what it can teach | limit |
|---|---|---|---|---|---|
| Replogle K562 genome-wide | K562 | 9,865; **272/300** validation targets | ~8,200 | each challenge target's response, in one line | one line; 172 cells per challenge target |
| Replogle K562 essential | K562 | 2,057; 0/300 | ~8,200 | a second K562 measurement of 2,053 shared targets: within-line reproducibility | essential genes only |
| Replogle RPE1 | RPE1 | 2,393; 0/300 | ~8,000 | the only second cell line with single cells: 2,390 targets shared with K562 genome-wide | essential genes; 72 cells per target |
| LINCS CRISPR-KO, Level 5 | 10 lines × ~5,100 targets, 9 × ~530 | 5,156; 82/300 | 978 landmarks | **cross-line transfer at scale**: 4,349 targets in at least 10 lines | knockout not knockdown; bulk; landmark genes only |
| challenge controls | A, B, C (D, E, F later) | — | 18,533 | baseline, dispersion, co-expression in the context | no perturbation |
| validation leaderboard | A, B, C | 300 | 18,533 | **the effect of any change, on the challenge's own data** | aggregates only; three contexts; possible overfitting |

(All counts from D122.)

**How the data teaches cross-cell-line transfer.** Three sources, all imperfect:

* **K562 ↔ RPE1 on 2,390 shared targets:** how much of a target-specific
  response survives a change of cell line, at single-cell depth. Essential
  targets only, so it probably overstates what non-essential targets keep.
* **LINCS's ten large lines:** with nine other lines to average over, how well
  a target's response in a new line is predicted, and whether averaging lines
  beats using one. Bulk landmark genes, and knockout.
* **The validation leaderboard:** the K562 lookup's effect on the challenge's
  own cells. Aggregate only, but the only measurement in the target regime.

**No training cell line is near A, B or C** (D122): control profiles
correlate 0.32–0.42 with K562 and RPE1, and no LINCS line stands out.
Nothing licenses weighting one source line over another by similarity yet.

**Likely final-round coverage.** The final list overlaps the validation 300.
K562 genome-wide covered 272/300 (91%) of the validation list, and the
validation targets are non-essential genes with weak-to-moderate responses
in K562 (Anderson–Darling median 5, D122). If the final list is drawn the same
way, expect similar coverage. Unknown until the release; the data prep logs
"challenge perturbations present: n/N" per source (`FINAL_ROUND.md` §2).

---

## 4. Candidate approaches, ranked by expected gain per unit cost

A prediction for target *t* in context *c*, in log2 fold change per gene:

    lfc(c, t) = a_m · m(c)  +  a_s · w(t) · s(t, c)

* `m(c)` is the mean-response estimate;
* `s(t, c)` is the target-specific estimate and `w(t)` its per-target
  confidence;
* `a_m` and `a_s` are global scales;
* the count generator then emits cells.

The current pipeline is this with `m` = the model's shared part,
`s` = the model's own deviation from it, `w` = 1, `a_m` = 2 and `a_s` = 1
(D117, D118).

| rank | approach | hypothesis | expected effect, and why | cheapest decisive experiment | cost |
|---|---|---|---|---|---|
| 1 | **Size of the shared part, alone and with the specific part** | The score is an nmae–fid trade-off set by the size of the change. The model's specific part costs on both (D118). | From −0.032 toward −0.01…0: v4, 98% shared at a similar size, scored −0.014 (D117). | 5 override uploads, no code (§5.3) | 0 GPU; ~1.5 h server time per upload |
| 2 | **K562 lookup for the target-specific part** | A covered target's K562 residual (its change minus K562's mean change) points the right way in a new line often enough to help pds and reach. | Unknown; this is the key unknown. If the residual corrected for noise correlates ≥ 0.3 across lines, pds and reach gain; if ≈ 0, they lose and we stop here. | Offline: the oracle ladder's `own_mean+src_res` arms (CPU ~3 h). On the leaderboard: `m` + α · K562 residual, α ∈ {0.25, 0.5, 1}, paired against `m` alone | ~1 day of code (a lookup source in `predict`), then 3–4 uploads |
| 3 | **Size rises with sign confidence** | reach and fid reward calling first the genes whose sign we are surest of. | reach spans 0.04 → 0.90 and every upload is at its baseline: the largest unexploited member. Even the mean response's genes could be ordered by how consistent their sign is across targets and screens. | Per-gene sign consistency from K562 (fraction of targets moving each gene the same way), used to rescale `m`; paired upload | ~½ day of code |
| 4 | **A call budget per target** | fid and Jaccard want about as many calls as the truth makes; the truth makes more for stronger targets. | Moderate: fid and Jaccard together moved 0.07–1.4 across uploads. Scaling each target's change to its K562 strength (Anderson–Darling, `fold_expr`) moves its call count with the truth's. | Paired upload: per-target scale ∝ K562 strength, against a constant scale | ~½ day |
| 5 | **The mean response from other lines, adjusted by the controls** | A context's mean response is partly predictable from its control profile. | Only if measurement 4's LINCS leave-one-line-out shows control-based prediction beating the plain cross-line average. The current shared part is already near the oracle (D117), so the upside is small. | Measurement 4 (CPU ~45 min), then one upload | — |
| 6 | **LINCS cross-line target effects** | The average of a target's residual over nine lines transfers better than one line. | Applies to 82/300 targets and landmark genes, so a small share of the score; worth it only if measurement 4 shows multi-line averaging clearly beats one line. | Measurement 4 | ~1 day of code |
| 7 | **Similar targets for the uncovered** (kNN over target-role priors) | Targets with similar annotations have similar residuals. | 28/300 targets: at most about 9% of any target-specific gain. | Offline, on K562: predict held-out targets' residuals from their neighbours' | ~1 day |
| 8 | **Low-rank target × context factorization** | Target effects are shared, with context-specific loadings. | The loadings of a new context must come from its controls alone, which is rank 5's question. With two single-cell lines, not identifiable; with LINCS, a landmark-only version. | After measurement 4 | days |
| 9 | **Gene-token model improvements** | — | Its specific part costs points (D118); its per-cell mapping is at ridge level (D102–D108). Not worth effort before ranks 1–4. Kept as a source of `m`. | — | — |

Floors every candidate must beat, on the leaderboard and offline:

* **no change**;
* **the model's shared part alone** (rank 1);
* **the K562 mean response alone**: one no-model mean, the `src_mean` arm offline;
* **K562 lookup unshrunk**: the `src_lookup` arm offline.

The generator (count model, dispersion, how changes are applied) stays as it
is until the ladder's `own_target` arm measures what it loses. If an oracle
change emitted through it scores far below the replicate, the generator
becomes a candidate; until then nothing says it is the bottleneck.

---

## 5. Evaluation protocol

### 5.1 The rule for an upload

Uploads are unlimited, so their cost is our turnaround: about 1–2 hours of
server time each (`predict`, `sanity`, `validate`, `package`). The rule is
about evidence, not budget.

1. **Every upload is a paired experiment.** It differs from a named parent
   upload in **one** component, with the same seed. The hypothesis and the
   expected sign are written down before the upload.
2. **The noise floor is measured first.** One upload identical to the parent
   but for the generator seed (`--seed 3`). Its distance from the parent, per
   member, is the leaderboard's own noise for our submissions. A change is
   adopted only if it beats the parent by more than **twice** that.
3. **Per-context checks only where they decide something.** Neither the `vcc`
   tool nor the website reports per-context scores (confirmed 2026-10-06), so
   an upload gives only the average over A, B and C. A change is adopted on
   its overall gain: more than twice the seed noise, about 0.004 (P0). For the
   one or two decisions that matter most (the denoiser, the final recipe), a
   context-isolated pair measures it per context: the candidate in one
   context, the parent in the other two. At 2 uploads a day that is used
   sparingly; offline checks carry the rest.
4. **Prefer rules to constants.** A scale that is right for A/B/C may not be
   right for D/E/F. Settings are expressed relative to quantities the new
   context provides (the model's own shared estimate, the controls'
   dispersion, a target's K562 strength), so the same rule recomputes for a
   new context.
5. **Every upload is logged** in `UPLOADS.md`: run, commit, parent, the one
   change, the six members, the overall score. Every conclusion goes in
   DECISIONS.md, whichever way it comes out.

**Overfitting risk, stated.** We would tune on 300 targets × 3 contexts and be
judged on new contexts and a partly new target list. Rules 3 and 4 limit it;
nothing removes it. After the test release, the same paired protocol runs on
the test-phase leaderboard. This plan assumes that leaderboard is the final
ranking (your answer did not settle whether part of the test set is hidden);
if part is hidden, the test-phase probes should only choose between
settings already shown to work on the validation leaderboard.

### 5.2 Offline instruments, and what they are for

* **The oracle ladder** (measurement 3): mechanism, not settings.
  - Does `own_mean` through our generator reach the organizers' 0 (a
    generator check)?
  - Does ×2 beat ×1?
  - Do K562 residuals add anything on top of the true mean in a new line?
  - What does an oracle per-target change lose in the generator?
  
  Error bars from a 1,000-resample target bootstrap; arms paired on the same
  resamples.
* **Transfer structure** (measurement 4): how much residual survives a change
  of line, corrected for noise, by effect size; whether LINCS averaging helps;
  whether controls predict a line's mean response.
* **The rehearsal** no longer chooses the generator setting. On its fixed
  column it measures a regime unlike the challenge (D121). It stays runnable,
  off the final round's critical path.

### 5.3 First batch: five uploads, no code

All from `runs/server_shared` (shared 2 / specific 1, −0.0318), changing only
the generator setting through `predict.override_*` (D115). Effect scale 0.01
stands for "specific part off": `shared_scale` needs a positive effect
scale (`predict/generator.py`).

| probe | threshold / effect / shared | change against the parent | hypothesis |
|---|---|---|---|
| P0 | 0.05 / 1.0 / 2.0, `--seed 3` | generator seed only | the noise floor (rule 2) |
| P1 | 0.05 / 0.01 / 2.0 | specific part off | the model's specific part costs points (D118); expect +0.005 to +0.02 |
| P2 | 0.05 / 0.01 / 1.5 | P1 with a smaller shared part | brackets the shared optimum from below |
| P3 | 0.05 / 0.01 / 3.0 | P1 with a larger shared part | brackets it from above; v5's nmae −0.50 says too large costs fast |
| P4 | 0.05 / 0.5 / 2.0 | specific part halved | if P1 wins, whether some of the specific part is worth keeping |

What each outcome changes:
* If P1 beats the parent by more than twice P0's noise, the model's
  target-specific part is dropped, and rank 2 (the lookup) becomes the only
  target-specific candidate.
* P1–P3 fix `a_m` for the shared-only baseline.

---

## 6. Architecture and milestones

### 6.1 The architecture

```
 controls(c) ──► context stats (depth, dispersion, detectable genes, CP10K mean)
                                   │
 sources ──► m(c): mean response   │    s(t, c), w(t): target-specific, confidence
   · Phase 2 model (existing)      │      · K562 lookup residual, strength
   · K562 / RPE1 mean (new, small) │      · LINCS cross-line average (if earned)
   · LINCS adjustment (if earned)  │      · model's own deviation (if earned)
                                   ▼
                composer: lfc = a_m·m + a_s·w·s, sized by sign confidence
                                   │
              count generator (existing) ──► submission writer, validator,
                                             sanity, packaging (existing)
```

A **source** is anything that returns, for a context and a target list, one
change per gene on the challenge gene axis, plus a confidence. Each is
computed once per round and stored; the composer and the generator run in
minutes. So every upload after the first is a composer setting, not a
retraining, and the final round's run shrinks to: data prep, priors and
Phase 2 for the model source, the sources' tables, predict.

**Keep:**
* the data layer and `check-data`;
* the priors and Phase 2, as the model source;
* the count generator;
* the submission writer, validator, sanity checks and packaging;
* the official-scorer wrapper and the scale (fixed in D120);
* provenance and determinism;
* `predict.override_*`;
* the measurement scripts.

All are validated by uploads that `vcc prep` accepted.

**Off the critical path, not deleted:**
* the rehearsal's Phase 2 retraining (about 5–7 h, and its instrument is not
  validated, D121);
* Phase 1's weight transfer (measured not to help, D90–D91);
* per-cell mode and OT (D100);
* Phase 3 adaptation (neutral, D112);
* the core-freeze ablation.

They stay as configurable options. The checklist reports them as
deviations.

### 6.2 Milestones

**T** is the test-set release, **2026-10-22**: three weeks from today. The
test-phase deadline is not yet known; the last row runs to it.

| when | what | output |
|---|---|---|
| Oct 1–3 | P0–P4 uploads (§5.3); measurements 3 and 4 on the server | the noise floor, `a_m`, whether the model's specific part goes; D124+ |
| Oct 2–6 | Code: the source/composer split; the K562 lookup source; the per-gene sign-confidence and per-target strength factors (ranks 2–4) | tests; mini `all` passing |
| Oct 6–13 | Paired uploads for ranks 2, 3, 4, then 5–7 if measurement 4 earns them | the validation-best recipe, with per-member evidence |
| Oct 13–19 | Final-round path: `configs/server_final.yaml`, a fast `all` with the rehearsal off the critical path (≈ 5–6 h); **a full rehearsal on the validation release reproducing the validation-best upload bit for bit**; `FINAL_ROUND.md` rewritten | a tested procedure |
| Oct 19–21 | Buffer; the validation-best recipe frozen and tagged | — |
| Oct 22 (T) | data prep (a few hours) and the run; **safe upload** = the validation-best recipe | the first test upload within about 24 h of the release |
| Oct 23 … deadline | the paired protocol on the test-phase leaderboard, choosing between settings already supported on validation; noise re-measured once | the final submission |

The estimate of about 5–6 h for the final run is unmeasured. It is the sum of:
* data prep (a few hours, D119);
* priors (20–60 min, README);
* Phase 2 at 12,000 steps (72 min, D107), without the ablation arm;
* the sources' tables (minutes to an hour, reading K562 cells);
* predict, sanity, validate and package (about 1–2 h).

It is measured during the full rehearsal, Oct 13–19.

---

## 7. Recommended changes to CLAUDE.md

Proposed, not made. Until you adopt them, each departure is a recorded
deviation (DECISIONS.md and `checklist.json`).

1. **§6: the leaderboard's 0 is an oracle.** Say that the mean-response
   baseline is computed from the hidden truth (docs §6a), that expression is
   clamped to [0, 1], that nmae's floor is linear to −6, and that the other
   four are unfloored. Replace the "no change scores ≈ 0" column with each
   member's leaderboard behaviour (§1.1 here). Name `from_replicate` as the
   leaderboard's column (D120).
2. **§4.7: drop "genes with no confident change keep a fold change of exactly
   1" as the governing rule.** Its rationale, that expression punishes large
   effects, does not hold on the leaderboard. Replace it with: sizes follow
   sign confidence, and each target's number of calls should track the
   truth's. The generator itself is unchanged.
3. **§4.6: the rehearsal is a mechanism instrument, not the calibrator.**
   Calibration is by paired validation-leaderboard uploads (§5.1 here). The
   rehearsal's three variants stay available, off the final round's critical
   path.
4. **§4 and §4.8: the method is a composition of sources** (§6.1 here). The
   gene-token model is one source. Checklist items 5–9 and 11 (shared core,
   frozen core and adapters, forgetting guard, Phases 1–3) become optional
   components, reported as present or off; new items cover the sources, the
   composer and `UPLOADS.md`.
5. **§0: the goal is the final-round score**, the prototype requirements
   (every element real and run by `all`) applying to whatever the composer
   uses.
6. **§1.1: name the instrument.** The validation leaderboard is a permitted
   instrument for choosing settings, under §5.1's rules.

---

## 8. What I need from you

1. Approval of the direction: sources plus composer, the validation
   leaderboard as the instrument, the rehearsal off the critical path. Or what
   to change.
2. The test-phase deadline (the release is 2026-10-22).
3. ~~Per-context scores~~: not reported by the `vcc` tool or the website
   (2026-10-06); rule 3 of §5.1 was changed accordingly.
4. Run P0–P4 and measurements 3–4. The commands are in the session message
   that accompanies this file.
