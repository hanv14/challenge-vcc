# DECISIONS.md

One entry per non-obvious choice: the decision, the reason, and the
alternative that was not taken. 44 entries, M0 to M6.

**Deviations from `CLAUDE.md`** are marked **DEVIATION** below, listed in
`PLAN.md` §9 and stated in the final message. All four were authorized by the
user at the M0 review; nothing deviates silently:

* **D1** — The package is `vccp`, not `vcc` (CLAUDE.md §2, §7)
* **D2** — `mini.yaml` sets `device: cpu` (CLAUDE.md §1)
* **D3** — `train.unfreeze_core` defaults true for Phase 2 (CLAUDE.md §4.3)
* **D4** — `prediction.vcc` is optional (CLAUDE.md §4.8 item 13)

---

## M0 — plan

### D1. The package is `vccp`, not `vcc` — **DEVIATION** (CLAUDE.md §2, §7)

**Decision.** The Python package is `vccp/` and every command is
`python -m vccp <stage> --config …`. The challenge's own packaging tool is
invoked only as the executable found by `shutil.which("vcc")`.

**Reason.** `CLAUDE.md` asks for a package importable as `vcc` while the
challenge ships a command-line tool of the same name. On the server, a `vcc`
Python package installed by the challenge would make `python -m vcc` ambiguous:
run from outside the repository it resolves to theirs, run from inside it
shadows theirs for anything that imports it. Renaming costs one letter now and
is very painful later. Approved by the user at the M0 review.

**Alternative.** Keep the name and guard at startup by asserting
`vcc.__file__` resolves under the repository root. Rejected: it detects the
collision rather than removing it, and does nothing about our package shadowing
theirs.

### D2. `mini.yaml` sets `device: cpu` — **DEVIATION** (CLAUDE.md §1)

**Decision.** The mini config pins `device: cpu`; `server.yaml` keeps
`device: auto`. `device` stays a single top-level key.

**Reason.** `CLAUDE.md` §1's snippet shows `device: auto` for mini while §1.2
says the `resources` block should carry `device: cpu` for mini — and `device`
is not among the `resources` keys §1.2 lists. One key, explicit value, no
reliance on `auto` happening to find no CUDA in the container. Approved by the
user at the M0 review.

**Alternative.** A second `resources.device` key overriding the top-level one.
Rejected: two keys for one setting is exactly the ambiguity that produced the
question.

### D3. `train.unfreeze_core` defaults true for Phase 2 — **DEVIATION** (CLAUDE.md §4.3)

**Decision.** The core is trainable in Phase 2 and frozen in Phase 3, both
configurable. Phase 2 additionally trains a second, frozen-core arm, and Phase 1
validation is scored under both; the comparison is reported in
`reports/forgetting.json` and the rehearsal report.

**Reason.** §4.3's stated default freezes the core in every later phase, which
leaves the forgetting guard (§4.8 item 7) with almost nothing to act on. The
user's judgement: Phase 2 has the data to support training the core and is where
the guard is worth having, while Phase 3 sees only control cells, where updating
the core would risk the perturbation knowledge Phases 1–2 built. The ablation
exists so the default rests on a measurement rather than on that judgement.

**Alternative.** Keep the core frozen everywhere and report that the guard is
inactive by construction. Rejected by the user at the M0 review.

### D4. `prediction.vcc` is optional — **DEVIATION** (CLAUDE.md §4.8 item 13)

**Decision.** The deliverable is `submission/prediction.h5ad` in the format of
§8. The `vcc prep` code path is still built and still runs when the tool is on
`PATH`, but the `.vcc` is not required for a run to be complete.

**Reason.** The user packages the submission themselves with the challenge's
tool on the server. The tool cannot be installed in the cloud, so requiring the
`.vcc` would make every cloud run incomplete for a reason unrelated to the
method.

### D5. The calibration objective is normalized against the no-change points

**Decision.** The generator's confidence threshold and effect-size scale are
fitted on rehearsal variant 2 by maximizing the mean of the six official
metrics *after* mapping each so that its documented no-change value is 0 and
better is positive; where the baseline and replicate anchors can be built on the
rehearsal data, the leaderboard's own 0 = baseline / 1 = replicate scaling is
used instead. The objective actually used is recorded in `phase3_policy.json`.

**Reason.** §4.7 says "maximizing the mean of the six official metrics", but two
of the six are lower-is-better and all six live on different scales, so their
raw mean is not a meaningful quantity to maximize.

**Alternative.** Maximize the raw mean as written. Rejected: it would trade a
point of `sig_jaccard` against a point of `lfc_nmae` as though they were
commensurable, and would reward raising an error metric.

### D6. Sanity check 2 bounds the moved genes as well as the unmoved ones

**Decision.** Check 2 has two halves: genes the model did not predict to move
must have a predicted mean inside a bootstrap range of the context's control
per-gene means; genes it did predict to move must satisfy
|log2 FC vs control| ≤ `sanity.max_abs_log2fc` (default 10). The target gene is
excluded from both. Both halves report the violating fraction and the worst
offenders.

**Reason.** §6.1's check 2 exempts "genes the model predicts to move", which as
written exempts every gene that could fail it. The genes below the confidence
threshold are copies of control cells by construction, so a check restricted to
them tests nothing. Raised by the user at the M0 review.

### D7. Rehearsal arms share their control resample and generator seed

**Decision.** Within a rehearsal variant, the method, the upper bound and the
floor draw the same control cells and use the same generator settings, seeded
per `(variant, target)`. The official metrics for variants 1 and 3 are keyed
`official_panel_assisted`, never under variant 2's key.

**Reason.** The three arms exist to be compared; if they differ in their control
resample as well as in their predicted fold changes, the comparison measures
sampling noise too. And variants 1 and 3 are fed the *true* panel values, so
their official metrics are not end-to-end performance and must not be able to be
read as though they were. Both raised by the user at the M0 review.

### D8. The knockdown prior records its source and respects an expression floor

**Decision.** `fold_expr` is per-target from Replogle bulk where available
(median over the gene's promoter rows), otherwise the pooled median over every
Replogle bulk file present. The resolved source is recorded per target
(`pooled` or `per_target:<screen>`) and shown in the knockdown figure. The prior
is applied only where the target gene is detectably expressed in that context's
controls, above `phase3.knockdown.min_control_cpm`.

**Reason.** The medians differ by screen (0.155 K562, 0.088 RPE1) and the
challenge contexts are not identified with any Replogle cell line, so the number
used is a transfer assumption and should be visible as one. Applying a fold
change to a gene that is not expressed in controls would manufacture a change
out of noise. Raised by the user at the M0 review.

---

## M1 — data layer

### D9. `cell-eval2` is pinned to 0.16.0

**Decision.** `environment.yml` pins `cell-eval2==0.16.0`, CPU-only, with no
`[gpu]` or `[gpudge]` extras. The eval wrapper will introspect the installed
`EvalConfig` rather than assume attribute names, and will log the installed
version into every report that uses it.

**Reason.** `docs/metrics.md` and the brief describe 0.15.0 with
`rule_version 3`, but 0.16.0 is the only version PyPI publishes. The docs
themselves record metric semantics moving *within* a version several times
(#172, #271, #343, #348, #351), so the version a report was produced under is
part of the result.

**Alternative.** Leave it unpinned. Rejected: two runs on different days would
not be comparable, and the docs show that is not hypothetical.

### D10. Tests work on a symlink mirror, and a fixture proves `mini_data` is untouched

**Decision.** Tests that need a broken input build a tmp tree of symlinks
pointing at `mini_data` and replace or remove links in it. A session-scoped
autouse fixture snapshots every file's size and mtime under `mini_data` and
fails the session if anything changed.

**Reason.** `mini_data/` is read-only input (§1.1). While writing these tests I
deleted a real input file by calling `.resolve()` on a mirror path before
unlinking, which followed the link out of the mirror; it was restored from git.
The helpers now refuse any path that is not a symlink in the mirror, and the
fixture catches the whole class of mistake rather than that one instance.

**Alternative.** Copy `mini_data` (43 MB) per test. Rejected: slower, and it
would not have caught the bug — a copy is writable, so the mistake would have
been invisible.

### D11. Stages are registered as they are built

**Decision.** The CLI's stage list contains only implemented stages; `all` runs
those. An unregistered name is rejected with the list of the ones that exist.

**Reason.** §0 forbids stubs and `NotImplementedError`s in the delivered code
path. A stage that exists but does nothing is exactly that. The list grows with
each milestone and is complete at M5.

**Alternative.** Register every stage of §7 now, raising "not implemented" for
the unbuilt ones. Rejected: that is the stub §0 rules out, and it makes `all`
report a failure that is not one.

### D12. `check-data` reports a second gene-coverage number

**Decision.** Alongside `gene_coverage.csv`'s count, `check-data` reports how
many challenge genes Phase 1 or Phase 2 actually measures, and how many are
therefore reachable only through the challenge controls.

**Reason.** `gene_coverage.csv` counts a gene as covered when any source file
holds it, including LINCS genes outside the panel — 1,722 of 1,983 on mini. What
constrains the method is narrower: 1,442 genes are measured by a phase we train,
so 541 have no supervised perturbation response anywhere. That second number is
the one that matters for the priors and for reading the results, and on the
server it is about 10,000.

**Alternative.** Report only the coverage file's number. Rejected: it is
optimistic in a way that would mislead when reading the summary.

### D13. Replogle `ctrl_std` is floored at 1e-3

**Decision.** Control-SD units divide by `max(ctrl_std, 1e-3)`, in both the
Replogle and the challenge loaders.

**Reason.** §3.2 requires guarding against `ctrl_std == 0`; a gene with no
variance in a context's controls would otherwise produce infinite z-scores that
propagate into the priors and the loss.

**Alternative.** Drop zero-variance genes. Rejected: they are still genes the
submission must predict, and dropping them would make the gene axis
context-dependent.

---

## M2 — gene vocabulary and priors

### D14. A source that exists per screen becomes one block per screen

**Decision.** `replogle_coexpr` and `pert_phenotype_replogle` produce one
sub-block per Replogle screen (`replogle_coexpr:K562_gwps`,
`replogle_coexpr:rpe1`, …), each with its own coverage mask and missing flag,
rather than one pooled block.

**Reason.** The screens measure different gene sets, and principal directions
from two different decompositions are not in the same basis — pooling them
would put unrelated numbers in the same coordinate and call it a feature. A
gene measured in two screens gets features from both; a gene measured in one
gets that one and a missing flag for the other. It also means the server's
third screen (`K562_essential`) widens the prior automatically.

**Alternative.** Restrict the block to the genes every screen measures.
Rejected: it throws away coverage for genes measured in only one screen, which
is most of them.

### D15. Co-expression never forms a gene-gene matrix

**Decision.** Blocks 1 and 2 stream cells from disk in `priors.cell_block_size`
blocks and take a randomized SVD of the standardized data matrix, whose right
singular vectors are the principal directions of the gene-gene correlation.

**Reason.** A gene-gene correlation matrix is 18,533² on the server — 1.4 GB
before its eigendecomposition, and far more during it. The randomized method
holds only `(n_cells x k)` and `(n_genes x k)`, and `priors.coexpr_max_cells`
caps the first.

**Alternative.** Materialize the correlation matrix within `max_ram_gb`.
Rejected: it fits only just, and would not survive a larger control cohort.

### D16. Which blocks serve which role

**Decision.** Co-expression, HGNC groups and (if configured) the PLM block
serve **both** roles. The knockdown phenotype blocks serve the **target** role
only. Nothing serves the feature role alone, so the target-role prior is the
wider of the two.

**Reason.** §4.1 asks for a feature-role and a target-role prior. How a gene
behaves and what it belongs to describe it either way; what its knockdown does
to everything else describes it only as a target. Each role has its own `W`,
so a shared block can still be read differently by the two.

**Alternative.** Give each block to exactly one role. Rejected: it would deny
the perturbation token any knowledge of what the target gene *is*, which is
most of what is known about a target the screens never touched.

### D17. Response profiles are standardized per target before decomposition

**Decision.** In the knockdown-phenotype blocks, each target's response vector
is z-scored across genes before the decomposition, so two targets are alike
when their responses point the same way rather than when they are the same
size.

**Reason.** §3.3: LINCS is CRISPR knockout, the challenge is interference —
"directions and affected genes transfer well; magnitudes and the target gene's
own level do not". Encoding magnitude into the target-role prior would encode
the part that does not transfer.

**Alternative.** Keep the raw magnitude. Rejected for the reason above; the
magnitude is re-fitted where it belongs, by Phase 2 and by the generator's
effect-size scale.

**Amended at the M2 review.** Z-scoring alone loses the distinction between a
strong specific response and a near-null one — and most knockdowns do very
little, so a null response divided by its own tiny norm becomes a
confident-looking random direction. Each phenotype sub-block therefore also
carries **scalar magnitude and reliability features** in the same block, under
the same missing flag: `log1p_magnitude` plus the source's own quality signals
(Replogle `fold_expr`, `log1p_anderson_darling`, `log1p_n_cells`; LINCS
`cc_q75`, `log1p_n_sigs`, `log1p_n_rows`). The direction basis is learned from
the weighted responses — weight `m / (m + floor)`, halving at
`priors.phenotype_magnitude_floor` times the source's median magnitude — and
then **every** target is projected into it, so a null response lands where it
lands in a basis it did not help choose, rather than bending the basis toward
its own noise. `priors/checks.json` reports the fraction of targets below the
floor per sub-block.

### D18. The neighbour check runs per HGNC group, not per configured pattern

**Decision.** `priors.check_families` holds name fragments, but the check
evaluates every HGNC gene group matching a fragment **separately**, and skips
any with fewer than three members present, reporting the reason.

**Reason.** The first version pooled every group matching a pattern, which put
"Mitochondrial complex I assembly complex" and "Mitochondrial complex IV:
cytochrome c oxidase subunits" in one family and scored 0.0x enrichment. Those
are different machines and their subunits have no reason to be neighbours —
the check was asking a question with no right answer. Per group, on mini_data:
Proteasome 159x, Mitochondrial respiratory chain complex assembly factors
196x, Large ribosomal subunit biogenesis complex 17x, 3 of 3 scored groups
enriched, and the four small complexes explicitly skipped.

**Alternative.** Hand-list the complexes. Rejected: it would be a hardcoded
biology table that mini_data and the server would disagree about.

### D19. The missing flag is 1.0 for missing

**Decision.** Each block contributes its `block_dim` standardized columns plus
one flag column that is **1.0 where the block does not cover the gene** and
0.0 where it does; uncovered genes' feature columns are exactly zero.

**Reason.** A standardized feature of zero is the block's mean, not an absence.
Without the flag the model cannot tell "this source says this gene is typical"
from "this source has never seen this gene" — and on the server the second is
true of roughly 10,000 genes.

**Alternative.** Impute the missing values. Rejected: an imputed
co-expression profile for a gene measured nowhere is a fabrication, and the
free per-gene `delta` is the mechanism the specification gives for that case.

---

## M2 review — amendments

### D20. The magnitude check reports the depth confound beside the floor

**Decision.** The magnitude report carries, per sub-block, the RMS
percentiles and the correlation between response magnitude and the source's
depth signal (`n_cells` for Replogle, `n_sigs` for LINCS), plus a caveat
sentence.

**Reason.** On `mini_data` the fraction below the floor is 0.0% for both
Replogle screens, which reads as "every knockdown did something". It is not:
magnitude correlates **−0.75** with cell count on K562_gwps and −0.44 on
rpe1, because a pseudobulk over 8 cells is noisy and noise has a magnitude.
The floor fraction alone would have been read as biology. The depth signal is
a feature of the same block, so the model can discount magnitude where depth
is low; the report is what tells a human to expect that.

**Alternative.** Depth-correct the magnitude before reporting it. Not taken
at M2: it is a modelling choice rather than a diagnostic, the model has the
information to do it, and inventing a correction now would hide the raw
number the user asked to see.

### D21. Two kinds of context exclusion, because variant 2 needs both

**Decision.** `PriorScope` has `exclude_contexts` (hide a source entirely)
and `exclude_response_contexts` (hide only its perturbation responses, keep
its control cells). The three rehearsal variants' scopes are constructed in
`vccp/priors/scope.py`.

**Reason.** Rehearsal variant 2 learns perturbation behaviour in one screen
and adapts to the other *using only its controls* (§4.6). One exclusion set
cannot express that: hiding the screen entirely would remove the controls the
variant is built on, and hiding nothing would leak the responses it is
supposed to predict. Defining the scopes next to the rule keeps what the
priors promise and what the rehearsal asks for from drifting apart.

### D22. The rebuild plan is declared, and tested against a real rebuild

**Decision.** `priors/leakage.py` classifies every block as rebuilt, dropped
or reused under each variant's scope, and that plan goes into
`priors/checks.json`. `tests/test_leakage.py` builds the priors under each
scope and asserts the outcome matches.

**Reason.** The plan has to be readable before the rehearsal runs, but a
declaration nobody checks is just a comment. Building all six scopes inside
the `priors` stage would triple its cost for a result that does not change
between runs; testing them once does the same job. The classification depends
only on what *kind* of thing a scope hides, not on which targets it picks, so
the plan is exact even though the rehearsal chooses its held-out sets later.

### D23. Per-block sampling seeds, and a layout hash

**Decision.** Each block derives its own seed from
`sha256(run seed, scope key, block name)` rather than drawing from one shared
generator. `coverage.csv` and `prior_features.npz` record every block's name,
width, column labels, per-role offsets, coverage and seed, and
`PriorFeatures.layout_hash()` digests all of it.

**Reason.** With a shared generator, which cells a block sampled depended on
how many blocks ran before it, so adding a block changed an unrelated one's
sample. And the block set is not fixed: the server's `K562_essential` adds
two blocks and shifts every offset after them, so a checkpoint trained
against one layout must be able to refuse another. `load()` verifies the
stored hash against the layout it holds.

### D24. The size scan gained an explicit, auditable exemption

**Decision.** `# not-a-size: <reason>` on a line exempts it from the
hardcoded-size scan. A test caps the number of exemptions and requires each
to carry a reason.

**Reason.** The scan flagged `np.percentile(rms, [5, 25, 50, 75, 95])` — 50
is on the banned list as mini_data's `cells_per_pert`. Weakening the scan
would have cost more than the annotation. Making the exemption explicit and
greppable means it is a claim a reviewer can see and disagree with, rather
than something the scanner quietly allows.

---

## M3 — the shared model, Phases 1 and 2

### D25. Every head predicts a change, and starts at exactly zero

**Decision.** The output heads' final linear layers are zero-initialized, so
an untrained model predicts no change at all. Phase 1's perturbed-profile head
and Phase 2's perturbation module both add their output to a control baseline
rather than predicting an absolute level.

**Reason.** A perturbed profile is mostly its own control, and pushing that
through a 64-latent bottleneck spends the whole model on copying its input —
measured: the decoder reaches only 0.23 Pearson trying to reproduce a random
input it was just given. The bottleneck exists for the *response* (§4.2), and
control-SD units (§3.2) already express everything as a change. Zero-init then
makes "this perturbation did nothing" the starting point, which is also what
the challenge's metrics reward a prediction for not knowing (§4.7: a gene with
no confident change keeps a fold change of exactly 1) — an untrained head
cannot start out worse than the no-change baseline. On mini this moved
Phase 1's held-out `pert_pearson` from 0.047 to 0.100.

**Alternative.** Predict absolute levels and let the model learn the identity
part. Rejected on the measurement above.

**Consequence, accepted.** At exactly zero the chain rule gives zero, so for
one step no gradient flows *through* a head to what feeds it. The head's own
gradient lifts it off zero immediately, so this is a one-step delay, not a
blockage; the tests that check gradients flow through a head nudge it first
and say why.

### D26. Both Phase 2 losses are divided by their own no-change baseline

**Decision.** The mapping loss and the perturbation loss are each divided by
what predicting no change scores on them, measured on the training split.

**Reason.** They are in the same units but at different scales: the mapping
predicts single cells (variance ~1.0 in control-SD units) while the
perturbation module predicts a pseudobulk mean over cells (variance ~0.04).
With equal weights the mapping was twenty times the size of the perturbation
term and simply drowned it. Normalized, both read as a fraction of the
variance there was to explain, `loss_mapping` and `loss_perturbation` mean
what they say, and the scales will differ again on the server.

### D27. The perturbation module's baseline is the stored control pseudobulk

**Decision.** The control profile fed to the perturbation module — its input
and the baseline its predicted change is added to — is the screen's control
pseudobulk, not a fresh draw of control cells.

**Reason.** A bug found by checking whether the module beat predicting no
change on its own *training* targets, which it did not. The mean of n cells in
control-SD units carries noise of variance 1/n: at 8 cells that is 0.129,
against a pseudobulk response of 0.037 in K562_gwps — three and a half times
the signal, injected by the baseline before the model predicted anything, and
matching the observed error ratio of 2.7 almost exactly. With the stored
pseudobulk (variance 0.00000 by construction) the module goes below the
no-change line on both splits.

**What caught it.** Reporting `pert_mse_no_change` next to `pert_mse`, and
then the train/validation split of the same ratio. A raw MSE would have looked
like a small number and hidden this.

### D28. Output genes are subsampled per training step

**Decision.** `train.output_genes_per_step` decodes a fresh random subset of
the output genes each step (default 2,048; 256 on mini). Evaluation always
uses every gene.

**Reason.** The loss is a mean over genes, so a subsample estimates it without
bias, and the decoder's cost is linear in the number of genes decoded. On the
server the alternative is decoding 18,533 genes per step. On mini it took
Phase 1 from 822 ms/step to 340.

**Exception.** Phase 1's signature is decoded over the whole input set even
when the loss scores a subsample, because step 2 consumes it as a per-gene
input channel — a partial signature would change what step 2 sees.

### D29. Only targets that are challenge genes condition the perturbation module

**Decision.** Phase 1 trains on the LINCS rows whose target is a challenge
gene (177 of 688 on mini); Phase 2 splits its target pools, using every
target's cells for the Siamese mapping and only challenge-gene targets for the
perturbation module (36 of 142 training targets in K562_gwps).

**Reason.** The perturbation token is the target's target-role embedding, and
the vocabulary covers the challenge genes (§3.1). A target outside that axis
has no embedding, so its rows would train the model to ignore the
perturbation token. The mapping needs no token, so it keeps all the cells.
Both counts are reported in the phase metrics; on the server the overlap is
far better (272 of 300 validation targets are in K562_gwps).

### D30. A `masked_mse` with a per-row mask counts entries, not rows

**Decision.** The mask is broadcast to the error's shape before its entries
are counted.

**Reason.** A bug. Phase 1's `gmt` loss masks by row, so the mask arrives as
(B, 1) against a (B, G) error; counting the mask's own entries divided a sum
over B x G terms by B, inflating that loss by the number of genes. Found by a
test whose expected value I had to work out by hand.

---

## M4 — rehearsal, Phase 3, prediction

### D31. A target with no embedding is predicted as no change, not skipped

**Decision.** If a perturbation in `pert_counts.csv` is not a challenge gene,
the vocabulary has no target-role embedding for it, so no perturbation token
can be formed. That target still gets its `cells_per_pert` cells — drawn from
the context's controls with a fold change of exactly 1 everywhere — and the
prediction index lists it under `targets_without_token`.

**Reason.** The submission must contain every perturbation of
`pert_counts.csv` in every context (§8); failing the run, or omitting the
target, would make the whole submission invalid because of one label. A
no-change prediction is the honest answer for a perturbation the model was
given no way to represent, and §4.7 notes the metrics treat no change as
neutral rather than as an error.

**Alternative.** Raise. Rejected: it turns a one-label data surprise in the
final round into a total failure, and it is exactly the kind of thing that
surfaces at 2am before a deadline. The label is reported instead.

### D32. A rehearsal-scoped prior rebuild lands in the run's own layout

**Decision.** `build_priors(..., reference_layouts=...)` places a scoped
rebuild into the layout of the run's own priors: a block the scope leaves
with nothing to read keeps its columns, zero-filled, with its missing flag
set for every gene. The layout hash covers only structure — block names,
widths, offsets and column labels — not coverage counts or sampling seeds.

**Reason.** The rehearsal rebuilds the priors without the held-out targets
and contexts (§4.1's leakage rule). A dropped block shifts every offset after
it, so the Phase 1 core — trained against the run's priors — could not be
loaded into the rehearsal's arm at all, and the rehearsal would measure a
model that never saw Phase 1. Zero-filling is what §4.1 already prescribes
for a source that does not cover a gene, applied to a source that now covers
nothing; the scope's effect then shows up as a set missing flag rather than
as a silent shift.

**Alternative.** Train the rehearsal's arm from scratch. Rejected: it would
measure a different method from the one that is submitted.

### D33. The scorer never asks for more threads than polars has

**Decision.** `eval/official.py` requests `min(resources.cpu_threads,
polars.thread_pool_size())`.

**Reason.** `cell-eval2` gathers with polars, whose pool size is fixed at
import from `POLARS_MAX_THREADS` — by `scripts/server_env.sh` on the server,
by the conservative default `vccp/__main__.py` sets when the environment is
silent. Asking for more than the pool holds raises ("The number of threads
must be between 1 and 1"), and raising the pool to fit the config would
override what the environment said, which §1.2 forbids.

### D34. The leaderboard's 0–1 scale is attempted, and its refusal is reported

**Decision.** `eval/scale.py` builds both ends of §6's scale with
`cell-eval2`'s own tools — the generic-response baseline for 0, the
split-half replicate anchor for 1, packaged as a *real bundle* — on the
cross-context rehearsal dataset, and places every arm on it with
`score_metrics(..., real_bundle=...)`. Where `cell-eval2` refuses the pair,
its own message is recorded as the reason and the six raw values stand alone.

**Reason.** §6 asks for the scale "where the data allow" and for a statement
where they do not. An assertion that the data do not allow it, written
without trying, is a guess: on `mini_data` the pair builds on one of the two
screens and is refused on the other, for a specific reason
(`expr_distance_unbiased` summing non-positive over the reference panel) that
no amount of reasoning would have produced. The competition's own anchor rule
(`cell_eval2.competition`) supplies the split count and base seed, so the
scale is the leaderboard's rather than ours.

**Alternative.** Shell out to the `cell-eval2` CLI as §6's wording suggests.
Rejected: the Python entry points are the same code, and a subprocess would
have to be parsed rather than read.

### D35. The knockdown gate measures mean CP10K, not `ctrl_mean`

**Decision.** "Detectably expressed in this context's controls" is evaluated
as the per-gene **mean CP10K** over the held control cells, computed as a
matrix–vector product so no cells × genes float array is materialized.

**Reason.** `ctrl_mean` is the mean of `log1p(CP10K)`, which the zeros pull
far below the mean CP10K the specification names (§6.1 check 3, §4.7). Gating
on it applied the knockdown prior to 5 of 30 targets per context on
`mini_data`; on the quantity the specification actually names, 29 of 30.

### D36. Phase 1 validation is re-scored after Phase 3, so `forgetting` runs last but one

**Decision.** The `forgetting` stage runs after `phase3` and before
`predict`.

**Reason.** Checklist item 7 asks for Phase 1 validation re-scored "after
Phase 2 and after Phase 3". The stage scores every core checkpoint that
exists, so running it after Phase 3 is what makes the Phase 3 column real.

---

## M5 — submission, sanity, summary

### D37. Sanity check 2's first half is judged in count space

**Decision.** "Genes the model did not predict to move stay within the range
seen in that context's controls" is measured on **mean counts**, against the
sampling range of a draw of `cells_per_pert` control cells
(`sanity.control_mean_sigmas` standard errors). Not on normalized values.

**Reason.** Those cells *are* control cells, gene by gene — the generator
leaves an unmoved gene bit-for-bit alone. But moving other genes changes each
cell's library size, so in CP10K every untouched gene shifts a little. That
compositional shift is arithmetic, not a magnitude problem, and on a small
gene panel it was enough to flag 42% of the untouched genes.

### D38. The measured knockdown is exempt from the generator's two settings

**Decision.** `apply_settings` takes an `exempt` mask, and Phase 3 marks the
target's own gene where the knockdown prior was applied. The confidence
threshold and the effect scale do not touch it.

**Reason.** Both settings are calibrated against the *model's* uncertainty.
`fold_expr` is measured. At the scale the rehearsal chose on `mini_data`
(0.5), a weak guide's knockdown to 0.85 of control became 0.92 — no longer a
knockdown at all, and sanity check 3 duly failed for 22 of 74 targets. With
the exemption it is 74 of 74. None of the six metrics scores a target's own
gene, so this moves no score; it is correct biology, which is why §4.7 asks
for it and why check 3 exists.

### D39. Every log fold change is floored at a detectability level

**Decision.** `predict.cpm_floor` (default 0.01 CP10K) floors both sides of
every log ratio the pipeline forms — the model's predictions, the rehearsal's
measured truth and the sanity checks' re-measurement.

**Reason.** With a bare guard epsilon of 1e-6, a gene sitting at 0.05 CP10K
that the model switches off reads as a **15 log2** drop, and one that simply
draws no count in 50 cells reads as −14. Those numbers describe the epsilon
and the sparsity of the data, not the prediction: in count space both mean
"this gene is off", which is a change of a fraction of one count. Flooring at
a level fifty times below a single count in a 20,000-UMI cell makes "off" a
finite statement. On `mini_data` the largest predicted effect fell from
|log2 FC| ≈ 20 to ≈ 2.6, and sanity check 2 went from flagging half the moved
genes to flagging none.

**Alternative.** Raise `sanity.max_abs_log2fc` until the check stops
complaining. Rejected: the check was right and the quantity was wrong.

### D40. Sanity check 4 excludes every target gene, not each row's own

**Decision.** The change matrix has the column of *every* target gene removed
before the pairwise correlations and the duplicate test.

**Reason.** §6.1 says "excluding each target's own gene". Excluding only the
row's own leaves each row differing from its twin exactly at the genes being
knocked down, so two identical predictions are never detected as identical.
It is also what the discrimination metric does: it excludes every target gene
of the panel (§6).

### D41. `mini.yaml` accepts sanity warnings through the config, not a branch

**Decision.** `sanity.accept_warnings` is a config key, true in `mini.yaml`
and false in `server.yaml`. `--allow-warnings` still works and is what the
README tells the user to reach for on the server.

**Reason.** §6.1 says `validate` refuses a warned submission unless
`--allow-warnings`, and §9 says `python -m vccp all --config configs/mini.yaml`
must complete. On `mini_data` warnings are expected, so both cannot hold
unless the acceptance is a setting. Making it a config key keeps one code
path (§1.1 forbids an `if mini:` branch) and leaves the gate real where the
numbers mean something. A **fail** still stops the run under either config.

### D42. The submission is written by `validate`, after `sanity`

**Decision.** There is no `submit` stage. The `validate` stage assembles
`submission/prediction.h5ad` from the prediction blocks and then checks the
file it wrote.

**Reason.** §7 lists the stages and `submit` is not among them, while §6.1
requires that a failure of sanity check 1 stop the pipeline *before the
submission is written*. Writing inside `validate`, which runs after `sanity`,
is the only order that satisfies both. Sanity check 1 runs §8's rules over
the prediction blocks; `validate` runs the same rules over the assembled
file, through the same code.

---

## M6 — hand-off

### D43. The environment check is a script, not a stage

**Decision.** `scripts/check_env.py`, over `vccp/envcheck.py`, rather than a
`check-env` stage of the CLI.

**Reason.** §7 fixes the stage list and `check-env` is not in it, and the
check answers a question about the *machine* rather than about a run: it
writes nothing, has no artifact, and is most useful before the run directory
exists at all. Keeping it out of `STAGES` also keeps `all` exactly the twelve
stages §7 names.

### D44. The server runtimes in the README are labelled as estimates

**Decision.** The README gives measured numbers for `mini_data` and
explicitly labelled **estimates** for the server, with the quantity that
drives each stage and the five knobs to turn if a run overruns.

**Reason.** The full data was never reachable from this environment (§1.1),
so a server timing here would be a guess dressed as a measurement. What can
honestly be given is the shape of the cost — that the rehearsal is dominated
by about 43 scoring calls, each a Wilcoxon DE over `rehearsal.max_targets`
targets — and where to look for the real numbers, which is `log.txt`.

### D45. `check-data` reads the matrices, not only the metadata

**Decision.** `check-data` reads blocks of every `X` and every layer of every
`.h5ad` the pipeline opens (`checks.probe_matrices`, on by default), and
`scripts/probe_data.py` does the same on demand. Large files are sampled;
small ones are read in full.

**Reason.** A real run on the server died inside the `priors` stage with
`Can't synchronously read data (filter returned failure during read)` — HDF5
saying a compressed chunk would not decode — after eight minutes of work and
naming no file. Every earlier check had passed, because they read `obs`,
`var` and shapes, and a truncated matrix is perfectly consistent with all of
them. §1.1 asks `check-data` to give a clear pass/fail *before* training
starts; that is only true if something has touched the data.

**Alternative.** Verify checksums against the source instead. Better where
the source is reachable, but it does not exist here and would not catch a
missing compression plugin. The probe catches both, and reports which of the
two it is.

### D46. An adapter is created on its base layer's device and dtype

**Decision.** `LoRALinear.add_adapter` creates its two parameters with the
`device` and `dtype` of `self.base.weight`, not on the default device.

**Reason.** A bug, found by the first GPU run. Every stage does
`build_model(...).to(device)` and *then* `model.add_adapter(...)` — which is
the point of an adapter — so the new parameters were created on the CPU while
the rest of the model was on `cuda:0`, and the first forward pass died with
`Expected all tensors to be on the same device ... mat2 is on cpu`. It could
not show on `mini_data`, where the default device *is* the right one, which
is exactly why it survived M1–M6.

**Test.** PyTorch's `meta` device is a real device that exists without a GPU,
so `tests/test_model.py` moves a model to `meta`, adds adapters, and asserts
every parameter and buffer shares one device. The old code path gives
`{'meta', 'cpu'}` there. An audit for the same shape — a tensor created
without a device outside `__init__` — found no other instance.

### D47. The scorer's thread cap is the smallest pool, not just polars

**Decision.** `eval/official.py` caps its thread count at
`min(resources.cpu_threads, polars pool, NUMBA_NUM_THREADS)`, and `score()`
retries once single-threaded if a pool refuses the count anyway.

**Reason.** D33 capped against polars alone. `cell-eval2` also runs its
Wilcoxon test through **numba**, and `scripts/server_env.sh` — following
§1.2's mandated block — sets `NUMBA_NUM_THREADS=1` while
`POLARS_MAX_THREADS` follows `cpu_threads=4`. The two disagreeing is the
normal case on the server and impossible in the cloud, where every variable
defaults to 1. So the first server run asked numba for 4 threads, got
`ValueError: The number of threads must be between 1 and 1`, and **every one
of the rehearsal's scoring calls failed** — 21 arms across three variants.
The rehearsal produced no evidence at all, the generator was never
calibrated, and sanity checks 6 and 7 had nothing to read.

The retry exists because that trade is never worth making: scoring slowly
beats not scoring, and a rehearsal is the only evidence there is about
whether a submission is any good.

**Test.** A subprocess with `NUMBA_NUM_THREADS=1` and `POLARS_MAX_THREADS=4`
— the server's own combination — asserts the cap comes out at 1, and a real
scoring under those variables returns all six metrics. Neither could have
failed on a machine where the variables agree, which is why M1–M6 missed it.

### D48. The "not indicative" caveat follows the data, and matches a path component

**Decision.** `Config.on_mini_data` is the single place that asks the
question, and it matches a whole path component named `mini_data` rather than
the substring "mini" anywhere in the path. The rehearsal report and the
forgetting report now ask it instead of carrying the caveat unconditionally.

**Reason.** Both reports stated "on mini_data these numbers are not
indicative of real performance" on *every* run, including the server's. That
line sits at the bottom of `rehearsal/summary.txt`, which is exactly the file
taken to a proposal defense — a caveat that is false is worse than none,
because it discards a real result. Found by reading a server summary.

The component match, rather than a substring, came from the test for this:
pytest's own `tmp_path` for a test whose name contains "mini" matched the
substring rule. A server directory that happens to contain those letters is
real data.

### D49. Runs are deterministic by default

**Decision.** `train.deterministic` (default true) asks torch for
deterministic kernels (`warn_only`, so an op without one warns rather than
failing at hour three), turns TF32 off for matmuls and cuDNN, and fixes
`CUBLAS_WORKSPACE_CONFIG` — set in `vccp/__main__.py` beside the thread
variables, because cuBLAS reads it once when CUDA initializes.

**Reason.** Two `predict` runs on the server, from the same checkpoint with
the same settings and the same seeds, produced different submissions. §7
asks for seeds fixed and logged, and they were — but seeds fix which cells
are drawn and which way a coin lands, not the order a CUDA matmul reduces
in. Those last bits move the predicted values, and a confidence threshold
applied afterwards turns that into thousands of genes moving or not moving.
A prototype whose submission changes between identical runs cannot be
optimized against: no comparison between two configurations means anything
if the noise floor is unknown.

**Alternative.** Leave it off and treat the drift as part of the noise.
Rejected: the drift is invisible, it is not reported anywhere, and the whole
point of the next phase of work is comparing configurations.

### D50. A stage whose inputs were rewritten is not finished

**Decision.** `sanity` and `validate` record the newest prediction block's
mtime in their stage marker, and `package` records the submission's; a stage
whose inputs are newer than that reruns instead of being skipped. `validate`
and `package` additionally refuse outright when the artifact they would
describe is older than the predictions.

**Reason.** Resume keyed on the config hash alone. A rerun of `predict`
changes no config, so `sanity`, `validate` and `package` were all silently
skipped, and `prediction.vcc` on disk described a set of prediction blocks
that had since been overwritten. Nothing looked wrong: the validator report
said PASS, the checklist said 15/15, and every file existed. It took
comparing two targets byte for byte against the assembled file to see it.

A submission is uploaded once and scored for weeks. One built from
predictions that no longer exist is worse than no submission, precisely
because nothing about it looks wrong.

### D51. Every run stamps the code that produced it

**Decision.** `vccp/provenance.py` records the git commit, the branch,
whether the tree was dirty and which files differed, plus the versions of
the packages whose numbers reach the output. It goes into `config.yaml`,
`checklist.json` and `reports/summary.md`, and one line of it is logged at
the start of every run.

**Reason.** The run directory recorded the settings, the generator and the
data — everything except which code read them. A `.vcc` on a server six
weeks after the fact is unattributable without it, and the user is about to
submit several versions and compare their scores. It can never fail a run: a
tree that is not a git checkout reports what it can.

## P1 — per-cell OT

### D52. `ot_epsilon` is a fraction of the cost, not an absolute distance

**Decision.** The OT regularization strength is expressed as a fraction of
the batch's mean transport cost, resolved inside `paired_target`; the solver
`sinkhorn_log` keeps an absolute value. The swept grid is
`{0.005, 0.01, 0.02, 0.05, ∞}`.

**Reason.** Measured on a realistic batch (256 × 256 cells, 955 genes,
control-SD units) the mean cost is ~2.0 and the whole pooled-to-hard
transition happens between 0.005 and 0.05 of it. An absolute value would have
to be retuned for every screen whose cells are noisier or quieter, and the
grid first proposed — `{0.01, 0.05, 0.2, 1.0, ∞}` absolute — put three of its
five points in the region where the coupling is already pooled.

**Alternative.** Resolve `epsilon` against a per-screen constant estimated
once, rather than per batch. Cleaner statistically, since the effective
`epsilon` would then not vary step to step, but it needs a calibration pass
and the per-batch variation is small next to what the cell draw itself
contributes. Revisit in P3 if the training curves show it.

### D53. The OT solver reports its residual rather than asserting convergence

**Decision.** `sinkhorn_log` stops at a marginal drift of 1e-4 or at
`DEFAULT_ITERATIONS` (200), whichever comes first, and
`coupling_diagnostics` reports the drift it finished at.

**Reason.** At the concentrated end of the sweep (`epsilon` ≲ 0.01) Sinkhorn
does not converge in any affordable number of iterations — 300 was not enough
at 0.005, against 36 at 0.01 and 3 at 0.05. That is a property of entropic OT
near the hard-assignment limit, not something more iterations fix. A partly
converged coupling still gives a usable weighted average, so the run should
continue and say so rather than fail or silently pretend.

**Alternative.** Raise the ceiling until everything converges (unaffordable
inside a training step), or refuse to run below 0.01 (would remove the
interesting end of the sweep before measuring it).

### D54. `cost_spread` is reported because it tests the premise of the method

**Decision.** `coupling_diagnostics` reports the transport cost's standard
deviation over its mean, and P3's first output is that number on real drawn
cells, before any training.

**Reason.** In high dimension pairwise distances concentrate. On 955 genes of
independent noise the costs span 1.69–2.47 around 2.04, so every control cell
is nearly equidistant from every perturbed cell and the coupling is uniform
whatever `epsilon` says — OT would degenerate into the pooled objective it
exists to replace, and would do so silently. Real cells differ in depth and
cell state, which is the structure the pairing exploits, but that is an
expectation rather than a measurement. A spread near zero on real data means
the premise fails and the approach should be abandoned rather than tuned.

**Alternative.** Discover it from the training curves instead. Much more
expensive: a degenerate coupling looks exactly like a correctly configured
pooled run, so it would cost a server cycle to distinguish.

## P2 — the delta term, the type vocabulary, the Phase 1 ablation

### D55. The delta consistency term is normalized per batch, and its floor is reported

**Decision.** `phase2.loss_delta` (default 0.5) adds a squared error on the
*change* between the mapping's two arms, divided by the batch's own observed
change. Evaluation reports `map_delta_ratio_to_no_change` beside
`map_delta_noise_floor_ratio` — what a perfect predictor would score at
`phase2.delta_eval_cells` cells, estimated from two independent control
draws.

**Reason.** The two Siamese arms share weights but nothing constrained their
difference, and that difference is the only quantity the six official metrics
score; each arm's own MSE is dominated by the baseline expression level. An
observed change is a difference of two sampled means, so it carries noise
that inflates the ratio without moving its minimum — which makes the loss
sound and the *reading* misleading unless the floor is reported with it. At
16 cells a perfect predictor would score around 0.76; at 128 it is nearer
0.28.

**Alternative.** Measure a per-context normalizer once at load time, as
`mapping_no_change` and `perturbation_no_change` are. Steadier, but it needs
a calibration pass over cells at startup, and the per-batch value is
detached, so the varying denominator reweights steps rather than biasing
them. Revisit if the training curves are noisy.

### D56. The perturbation type is a vocabulary, `base + delta[assay]`

**Decision.** `knockdown_type`, one constant shared by every phase, becomes
`pert_type_base` plus a per-assay `pert_type_delta` row, penalized toward
zero by `train.type_delta_l2`. The assay is named by `phase1.pert_type`
(`crispr_ko`), `phase2.pert_type` and `phase3.pert_type` (both `crispri`),
and every call site that builds a perturbation token must pass one or the
model raises.

**Reason.** LINCS is CRISPR knockout and the challenge is CRISPR
interference. CLAUDE.md §3.3 says directions and affected genes transfer
between them but magnitudes and the target's own level do not — and the
model had nowhere to put that distinction, so Phase 1's knockout evidence
entered Phase 2 as though it were knockdown evidence. `base` is trained by
every phase and is what LINCS contributes to Replogle, as a named quantity
rather than an assumption inside a warm start; `delta` absorbs what is
specific to one assay. Phase 2 and Phase 3 share a row on purpose: they are
the same modality, and that sharing is the transfer. A new assay is one new
row from config, never a code change.

**Alternative.** Leave the single constant and let the adapters carry the
difference. They cannot: adapters are per phase, so nothing would remain
shared, and a new assay would need a new adapter trained on perturbed data
it does not have.

**Why raising is better than defaulting.** A default would let a caller
silently treat a knockout as a knockdown, which is precisely the failure the
vocabulary exists to prevent, and it would be invisible.

### D57. A third Phase 2 arm measures what Phase 1 contributes

**Decision.** `train.phase1_contribution_ablation` (default true) trains
`core_scratch`: no warm start from the Phase 1 checkpoint, no replay of its
objective, no L2-SP toward its weights. `phase2/metrics.json` gains
`phase1_contribution`, a per-metric `scratch − warm` difference, and the
summary prints it.

**Reason.** Both existing arms are warm-started, so nothing in the pipeline
had ever said whether Phase 1 contributes anything. With the perturbation
module at 1.036 on its own training targets it is entirely possible that it
contributes nothing, and the three-phase design would then rest on an
assumption at the proposal defense.

**The comparison is not symmetric, and the report says which way.** Two
settings pull against each other: a warm arm gives `train.replay_fraction` of
its steps to Phase 1, while the scratch arm's whole budget is
`train.ablation_steps_fraction` of the main arm's. At the default fraction of
1.0 the scratch arm ends up with more Phase 2 steps; on `mini.yaml`, where
the fraction is 0.5, it ends up with fewer (75 against 113). So the direction
cannot be stated once and for all — `n_phase2_steps` carries both counts and
`budget_favours` names the arm the difference helps, and the summary reads
the result accordingly.

*(Corrected after P4: this entry first claimed the bias always favours the
scratch arm, which is only true when `ablation_steps_fraction` is 1.0.)*

**Alternative.** Give the scratch arm replay too, to equalize the budgets.
That would put back through the side door exactly what the ablation removes.

## P3 — per-cell supervision

### D58. The coupling premise is checked on real cells before it is built on

**Decision.** `vccp/diagnostics/coupling.py` and `scripts/coupling_check.py`
measure, on real drawn cells and before any per-cell training,
whether the OT coupling carries information. Four numbers per `epsilon`,
with a control-against-control null arm, and a verdict of `informative`,
`degenerate` or `pairs-on-noise`.

**Reason.** P1 measured the pairwise costs between independent 955-dimensional
noise vectors at 1.69–2.47 around a mean of 2.04. If real cells concentrated
like that, every coupling would be uniform whatever `epsilon` said, the
barycentric target would collapse to the perturbed pseudobulk, and per-cell
training would *be* pooled training — silently, because the two are
indistinguishable from the outside. That would cost a server cycle to notice
and a second one to diagnose.

**The decisive number is `target_spread`**, not `cost_spread`: how much the
barycentric targets vary between control cells, against how much the
perturbed cells themselves vary. It is what the training step actually sees.
`library_size_pearson` is the second: §4 claims the coupling pairs like with
like on sequencing depth and cell state, so a pairing that does not track
depth is not the mechanism the method assumes.

**Result on `mini_data`** (K562_gwps, 256 cells a side, 716 panel genes):
cost spread **0.246** against **0.053** for noise alone, `target_spread`
**0.57** at `epsilon` 0.005, and `library_size_pearson` 0.54–0.74. Verdict
`informative`. The concentration risk is real for noise and does not bite on
real data.

**One caveat the report does not hide.** The control-against-control null arm
is nearly identical (spread 0.244, `target_spread` 0.52, pearson 0.60–0.81).
That is expected — the coupling is built before any prediction, so it can only
match on nuisance — but it means the method's value rests entirely on the
argument that removing nuisance variance sharpens the residual, not on the
coupling finding the perturbation itself.

### D59. `perturbation_mode` defaults to `pooled` until the rehearsal says otherwise

**Decision.** `phase2.perturbation_mode` selects `pooled` or `percell`, and
the default stays `pooled` through P3 and P4. P5's rehearsal A/B decides
whether it flips.

**Reason.** The specification asks for a per-cell model and the user's intent
is per-cell, but a default that has not been measured is a guess. `pooled` is
the arm with a leaderboard number behind it, poor as that number is. The
modes are a setting rather than a fork, so flipping the default after P5
costs one line.

### D60. Both modes are scored on both questions

**Decision.** `evaluate_per_context` reports
`pert_mse_ratio_to_no_change` (the pooled question: control pseudobulk in,
target pseudobulk out) **and** `pert_percell_ratio_to_no_change` (the
per-cell question: a cell in, its OT-paired truth out) for every arm,
whichever mode trained it, with `ot_effective_partners` beside them.

**Reason.** Two arms each scored only on their own objective say nothing
about each other. An arm that wins one metric and loses the other is telling
us something specific, and that is the reading P5 needs. `ot_effective_partners`
is reported with them because a coupling that came out uniform makes the two
metrics the same measurement, and that has to be visible rather than
inferred.

**Alternative.** Compare the two modes only on the rehearsal's leaderboard
metrics. That stays the arbiter, but it arrives a cycle later and cannot say
*why* an arm lost.

### D61. `ot_epsilon` defaults to 0.02, because pairing below that adds more noise than it removes

**Decision.** The default OT regularization is 0.02 of the batch's mean cost,
not the 0.01 first proposed, and `coupling_check` reports
`residual_reduction` so the trade-off is visible rather than assumed.

**Reason.** §4 claims that pairing brings each control cell's target nearer to
it, so the residual the model must explain is closer to the perturbation.
Measured on real cells that claim fails at the concentrated end:

| `epsilon` | effective partners | `target_spread` | `residual_reduction` |
|---|---|---|---|
| 0.005 | 3.2 | 0.57 | **1.35** |
| 0.01 | 8.0 | 0.38 | **1.17** |
| 0.02 | 34.6 | 0.15 | 0.98 |
| 0.05 | 117.5 | 0.02 | 0.94 |

A target averaged over three cells carries roughly a third of a cell's
sampling noise, and that is more than the pairing removes — so at 0.005 the
paired target sits 1.35x as far from its control cell as the pooled mean
does. The control-against-control null shows the same curve, which confirms
it is noise geometry rather than anything about perturbation.

**The two wants pull opposite ways**, and that is now stated rather than
discovered later: `target_spread` wants a small `epsilon`, since a large one
hands every cell the same target and the per-cell loss becomes the pooled
loss; `residual_reduction` wants a large one. 0.02 is where pairing measurably
stops adding noise while the targets still vary. P5's sweep is what settles it.

**This does not invalidate per-cell supervision**, and the report says why:
the loss's minimizer is unaffected by noise in the target (the expected
squared error against `true + noise` is still minimized at `true`), and the
reason for the redesign is that the model should be *asked* a per-cell
question, which §4.2 specifies and the pooled formulation never asks. But the
variance-reduction argument in §4, taken alone, is not supported by this
measurement, and the plan now says so.

**Alternative.** Keep 0.01 and let the sweep find it. Cheap in code, but it
would have spent a server cycle at a setting already measured to be adding
noise.

## P4 — per-cell prediction

### D62. The per-cell fold change is taken against the cell, not the population

**Decision.** `sd_units_to_log2fc` gains a `baseline_sd`. The pooled path
leaves it out, so the ratio is against the context's control mean as before;
the per-cell path passes the drawn cells themselves.

**Reason.** The generator multiplies *that cell's own counts*. A fold change
measured against the population mean would apply the cell's deviation from
that mean a second time, so a cell already above average for a gene would be
pushed further above it for no reason the model asked for.

**What it also buys.** The model's head predicts a change *added to its
input*, so a per-cell prediction is the cell's own baseline plus a delta.
Taking the ratio against that same baseline means a constant delta is a
constant shift in log space, which preserves the cell-to-cell spread the
drawn cells brought with them. Against the pooled mean it would not: the
folds would compensate for each cell's deviation and pull every cell toward
one profile, which is exactly what sanity check 5 exists to catch.

### D63. A gene with no counts in a drawn cell keeps a fold change of exactly 1

**Decision.** In per-cell mode, `log2fc[control_counts[rows] == 0] = 0`
before the generator sees it.

**Reason.** A multiplicative generator cannot move a zero — `0 x fold` is 0
whatever the fold — so the value there is set by `predict.cpm_floor` and by
how faint the control was, not by the prediction. With a *pooled* baseline
almost no gene sits at zero and this never mattered; with a per-cell baseline
most genes in a given cell do, so without it the reported effect sizes and
`max_abs_log2fc` would be dominated by arithmetic about cells nothing can
happen to.

**It changes no output**, only what is reported and what the generator's two
calibration settings are fitted against.

**The limitation it makes visible.** Neither mode can turn a gene on in a
cell with no counts for it; the count-space generator is multiplicative by
construction (§4.7). An additive or zero-inflated generator would be a
different component, and the generator is deliberately swappable.

### D64. The draw happens before the prediction, and the rows are passed on

**Decision.** `generate_cells` takes an optional `rows`; in per-cell mode
`run_predict` samples them, predicts on those cells, and hands the same rows
to the generator.

**Reason.** The model has to be shown *these* cells to answer for them, so the
draw cannot stay inside the generator. Passing the rows is what keeps row `i`
of the fold-change matrix matched to cell `i`; letting the generator draw
again would silently pair each cell with another cell's prediction, and
nothing downstream would notice.

§4.7's requirement is untouched: the draw is still fresh and independent per
target, it has just moved one call earlier.

### D65. P4's measurement: per-cell costs 2.3x and does not collapse the cells

**Measured**, `python -m vccp all` on `mini_data`, CPU, both modes, back to
back and otherwise identical:

| | wall clock | sanity | check 5 variance ratio (band 0.5–2.0) |
|---|---|---|---|
| `pooled` | 706 s | 5 pass, 2 warn, 0 fail | 0.94 / 0.95 / 0.95 |
| `percell` | 1653 s | 5 pass, 2 warn, 0 fail | 0.89 / 0.89 / 0.90 |

**2.34x the wall clock**, which is the estimate cycle C needs: per-cell
prediction pushes `cells_per_pert` rows through the encoder per target
instead of one, and Phase 2 pushes `ot_batch_cells x ot_targets_per_step`
rather than `batch_size` pseudobulk rows.

**Check 5 answers the risk D62 was reasoning about.** Per-cell fold changes
are taken against each cell's own baseline, and the concern was that they
would cancel the drawn cells' spread and pull every cell onto one profile.
They do not: the variance ratio moves from ~0.95 to ~0.89, well inside the
band, and no target falls outside it. The head predicting a change *added to
its input* is what preserves it, as D62 argued.

The two warnings are checks 6 and 7 in both modes, which is expected on
`mini_data` and is why `mini.yaml` sets `sanity.accept_warnings`.

**Nothing here says per-cell is better.** At 150 Phase 2 steps on 1,983 genes
the model barely trains: `pert_percell_ratio_to_no_change` is 1.00 in both
modes and `map_delta_ratio_to_no_change` is 1.00 against a noise floor of
0.24, so neither objective has been learned at all. P4 was testing that the
path runs, costs what it should, and does not break the generator. P5's
rehearsal is what compares them.

## P5 — the A/B that decides the default

### D66. The mode sweep is a rehearsal stage, off by default

**Decision.** `rehearsal.mode_sweep` runs the cross-context variant once per
configuration — `pooled`, then `percell` at each `ot_epsilon` — on
`rehearsal.mode_sweep_contexts` screens (1), scored with the six official
metrics on the leaderboard scale. Default off.

**Reason.** The default `perturbation_mode` has to rest on a measurement
(D59), and the measurement has to be the one the leaderboard agrees with —
which the calibration objective was not, the last time the two were compared.
Everything outside the configuration is held fixed: the same held-out screen,
the same targets, the same control draw, the same seed. The difference
between rows is then the configuration and nothing else.

Off by default because each row retrains a Phase 2 arm: it is a deliberate
measurement, not something every `all` run should pay for. One screen and a
five-point grid is §14's budget of five runs rather than ninety.

**The `floor` row costs no training** and is the reference every other row is
read against. A configuration that does not beat predicting no change has not
earned a submission, whatever it scores relative to the others.

### D67. Both modes predict through the same code

**Decision.** `vccp/predict/percell.py` holds the per-cell prediction core,
and both `predict/run.py` and `rehearsal/variants.py` call it.
`variants.predict_targets` dispatches on `cfg.phase2.perturbation_mode` for
both arms of every variant.

**Reason.** An A/B whose two sides predict differently measures the
difference between two prediction routines rather than between two trained
models. Before this the rehearsal always predicted from the pooled control
profile, so a per-cell-trained model would have been scored through the
pooled path and the comparison would have said nothing.

**The per-cell path draws exactly the rows `common.build_cells` will draw** —
same variant, same target, same seed, through `common.control_rows_for` —
so row `i` of the prediction meets cell `i` of the generated block. That also
keeps D7 intact: every arm still scores on the same control draw, so the
comparison between arms is still about the fold changes alone.

### D68. The calibration records which assay it was fitted on

**Decision.** The calibration carries `fitted_assay`, `applied_assay` and
`transfers_across_assays` into `phase3_policy.json`.
`rehearsal.calibrate_per_dataset` (default off) additionally fits the
settings on every scorable screen and reports the spread.

**Reason.** The generator's two settings are fitted on Replogle and applied
to the challenge, which is a different lab and protocol — an unstated
cross-assay assumption sitting in the middle of the submission path. The
provenance costs nothing and makes it a line anyone reading the policy can
see. The per-screen refit costs a full grid of official scorings per screen,
which is the slow part of the rehearsal, so it is opt-in; when it runs, a
wide `spread` means the settings are a property of the screen they were
fitted on rather than of the method, and carrying them to the challenge is
the weakest link in the submission path.

**A field named for the wrong thing is worse than no field.** The first
version reported `transfers_across_assays`, computed by comparing
`phase2.pert_type` with `phase3.pert_type` — which both read `crispri`, so it
came out **false** about a risk that is entirely real. A matching modality is
not the same assay: Replogle and the challenge differ in lab, protocol and
sequencing depth (~20,000 UMIs against 11,000–15,000), and nothing in the
model represents that. The comparison is now named `modality_differs` for
what it actually compares, and `carried_across_assays` is recorded
**unconditionally** beside it with a note saying why. The spread in
`per_dataset` is the only measurement of whether it matters.

## P6 — hand-off

### D69. A shared scorer failure is reported once, not once per row

**Decision.** When every row of the mode sweep fails to score with the same
message, the summary says so once and names the reason, instead of repeating
the scorer's message per row.

**Reason.** Found on the `mini_data` sweep: four configurations each carried
the same 500-character `expr_mse_unbiased_capped_norm` refusal, which filled
the table and buried the one thing worth knowing — that all four *ran*, and
the failure is a property of the dataset rather than of any configuration.
A row that fails on its own still gets its own line, because that one *is*
about that arm.

### D70. `configs/server_sweep.yaml` rather than instructions to edit a config

**Decision.** The A/B ships as its own config, identical to `server.yaml`
except for the `rehearsal.mode_sweep` block, writing to `runs/server_sweep/`.

**Reason.** The sweep's whole claim is that the only thing differing between
rows is the configuration being measured. A hand-edited config is the easiest
way to break that claim without noticing, and a separate `run_name` keeps the
measurement from overwriting the run a submission came from — which is
exactly the confusion that made the earlier threshold-and-scale mystery so
slow to resolve.

## The cross-assay variant

### D71. The cross-assay variant hides every screen's responses, not just one

**Decision.** `cross_assay_scope` sets `exclude_response_contexts` to **all**
Replogle screens, where `cross_context_scope` sets it to the held-out one.

**Reason.** The arm's whole claim is that it learned what a perturbation does
from LINCS alone. A prior block built on any single-cell screen's responses
— the Replogle perturbation-phenotype block of §4.1 — would be that
knowledge arriving by another route, and the variant would be measuring
something weaker than it says. Their control cells stay visible, which is
what "adapt using only its controls" means, and LINCS's own blocks stay
because it is the training assay here.

**The mapping is part of the point.** LINCS is panel-only
(`phase1_lincs.h5ad` is 688 x 955), so a Phase-1-only model has never seen a
rest gene and has no panel→rest mapping at all. The controls-only adaptation
is what builds it — precisely the thing the challenge allows and the
submission depends on, measured here for the first time.

### D72. The variant scores the same weights through two type rows

**Decision.** Besides `method` and `upper_bound`, the cross-assay variant
scores `method_source_modality`: the same model, asked as
`phase1.pert_type` (CRISPR knockout) instead of `phase2.pert_type` (CRISPR
interference).

**Reason.** This is the experiment that tests the type vocabulary of D56, and
nothing else in the pipeline does. In this arm `delta[crispri]` was never
trained, so asking as CRISPRi reaches `base` alone, while asking as knockout
reaches `base + delta[crispr_ko]` — the offset Phase 1 actually learned.
**The gap between the two arms is what that knockout-specific offset is worth
when carried to a knockdown screen**, which is exactly the question CLAUDE.md
§3.3 raises when it says directions transfer between the two but magnitudes
do not.

If the two arms score the same, the type vocabulary is doing nothing here and
§7.4 is a correctness fix rather than a working mechanism. If they differ, the
sign says whether Phase 1's knockout specialization helps or hurts on the
assay the submission actually faces.

**Cost:** one extra scored arm, no extra training — it is the same weights
through a different token.

### D73. The scratch arm is given the warm arm's *Phase 2* steps, not its total

**Decision.** `train.match_phase2_steps` (default true) sets the
`core_scratch` arm's budget to `phase2.steps x ablation_steps_fraction x
(1 - replay_fraction)`, so both arms train equally long on the objective they
are compared on. `budget_favours` now returns `None` when the two are within
`BUDGET_TOLERANCE` (2%), and `budgets_matched` says so.

**Reason.** The first server run measured the contribution at **−0.202** with
budgets of **442 against 600** — the arm that won had trained 36% longer on
the objective. That is not a small confound and it points the same way as the
result, so the number could not be read at all. D57 anticipated the
asymmetry and reported it; reporting a confound is not the same as removing
one, and a comparison that cannot be read is worse than no comparison because
it looks like evidence.

**Why default on.** The arm exists for one purpose: to say what Phase 1 is
worth. An unequal version of that measurement does not serve the purpose, and
nothing else uses the arm.

**What it does not fix.** The warm arm is still warm-started *and* held by
L2-SP toward Phase 1's weights, so "Phase 1's contribution" remains those two
together. Separating them takes a third setting (`train.l2sp_weight: 0`) and
another run; `pert_train_mse_ratio_to_no_change` is the number to watch,
because a warm start that hurts *training* fit is an optimization problem
rather than a transfer one.

**The tolerance exists because replay is sampled per step** rather than
scheduled, so two arms meant to train equally land a few steps apart.
Reporting that as a bias would be reporting noise.

### D74. An arm is judged at its best validation, not at its last step

**Decision.** `run_training` takes `select_on` and records `best_metrics`,
`best_step` and `divergence_final_over_best`. Phase 2 selects on
`pert_mse_ratio_to_no_change`, the ablations compare arms at their best, and
an arm whose final is more than `DIVERGENCE_RATIO` (1.25) worse than its own
best is logged and flagged as `diverged`.

**Reason.** The matched-budget server run made this unavoidable. The warm
`core_unfrozen` arm went 1.239 → **0.999** → 1.243 → **2.853** over 600
steps: it diverged in the last quarter. Every ablation read off
`final_metrics`, so `phase1_contribution` reported **−1.78** — a number that
describes a training blow-up, not what Phase 1 contributes. At each arm's
best the same comparison is **−0.019**, which is nearly nothing.

Two bad readings in a row from the same measurement, for two different
reasons (D73 was the budgets), and both times the number looked like
evidence.

**What it deliberately does not do.** The checkpoint saved is still the
final one, not the best. Selecting a checkpoint by validation is model
selection, which would change what Phase 3 inherits and is a bigger decision
than a reporting fix. So the report says plainly that for a diverged arm the
number and the model on disk are no longer the same thing, and names the arm.

**The divergence is the more useful finding.** A frozen core was stable
(final/best 1.03) and a from-scratch core was stable (1.09); only the
warm-started, unfrozen core blew up (2.86). That points at `train.lr`
against a warm start rather than at anything about transfer.

### D75. The config loader coerces numeric strings, because YAML will not

**Decision.** `_build_section` coerces each value to the type its field
declares. A numeric string becomes a number; a value that is not a number
raises `ConfigError` naming the section and key.

**Reason.** YAML 1.1 requires a decimal point **and** a signed exponent for
scientific notation, so `3.0e-4` is a float while `3e-4`, `1e-3` and `1E-3`
are all *strings*. The string then travels to the field's own `validate`,
where it fails as

    TypeError: '<=' not supported between instances of 'str' and 'int'

several frames from the config line that caused it. Almost every knob worth
tuning by hand is a small float — `train.lr`, `train.l2sp_weight`,
`model.delta_l2`, `phase2.ot_epsilon`, `predict.cpm_floor` — so this is a
trap the config is designed to walk into, and it cost a server run.

**Four things it must not break, each covered by a test.** `bool` subclasses
`int`, so a careless coercion turns `true` into `1`; tuple fields coerce
element by element; a `str` tuple like `priors.check_families` is left alone;
and the `None` in `rehearsal.mode_sweep_epsilons` — which *is* the pooled arm
— survives.

**A gap closed while here:** `steps: 1.5` used to pass validation and travel
into `range()`, which truncates silently. A fractional count is now rejected,
while `steps: 200.0` is accepted as 200.

### D76. Phase 1's warm start measurably hurts Phase 2, and `phase2.warm_start` can turn it off

**Measured**, server, 6,000 steps, matched Phase 2 budgets, each arm read at
its best:

| arm | map pearson | delta ÷ no-change | pert ÷ no-change | pert pearson |
|---|---|---|---|---|
| `core_unfrozen` (warm) | −0.021 | 1.001 | 1.080 | 0.095 |
| `core_frozen` (warm) | −0.022 | 1.006 | 1.010 | 0.089 |
| **`core_scratch`** (no Phase 1) | **+0.105** | **0.839** | **0.807** | **0.433** |

`phase1_contribution` = **−0.274**. Both warm arms sit at "no better than
predicting nothing" on every head, with a panel→rest mapping whose
correlation with the truth is *negative*. The scratch arm learns on all of
them, monotonically, and was still improving when its budget ran out
(0.918 → 0.921 → 0.857 → 0.807).

At 600 steps this looked like "nothing is learning and the budget is too
small". At 6,000 it is clear that the budget was only part of it: **the
scratch arm learns at this budget and the warm arms do not.**

**Decision.** `phase2.warm_start` (default true) can turn the warm start off,
making the shipped model a two-phase one. Default stays true because the
three-phase design is CLAUDE.md §4 and turning it off is a deviation that has
to be recorded as one — but the measurement now exists to justify it, and the
option is one line rather than a fork. A cold main arm makes the
`core_scratch` ablation a duplicate, so Phase 2 skips it and says why.

**What is still confounded.** The scratch arm differs from a warm arm in
three ways at once: no warm start, no L2-SP anchor, and no replay steps
interfering with the Phase 2 objective (the step *count* is matched, the
interference is not). Which of the three is doing the damage needs one run
with `train.l2sp_weight: 0` and `train.replay_fraction: 0` on a warm arm — if
that matches the scratch arm, the initialization is fine and the forgetting
guard is the problem.

### D77. Instability is measured as `worst / best`, not only `final / best`

**Decision.** `TrainResult` records `worst_value`/`worst_step` beside the
best, and an arm is flagged when **either** `final / best` or `worst / best`
exceeds `DIVERGENCE_RATIO`.

**Reason.** D74's check saw only how a run *ended*. The 6,000-step
`core_unfrozen` arm hit **2.4615** at step 4500 and ended at 1.2458 — 1.15x
its best, under the threshold — so it read as steady while swinging by 2.28x.
A run that swings that far and happens to land near its best is not under
control, and the ending cannot show it.

### D78. `phase2.steps` on the server is 6,000, and the server configs say `lr` out loud

**Decision.** `configs/server*.yaml` set `phase2.steps: 6000` and write
`train.lr` explicitly.

**Reason for the steps.** 600 was too small to say anything: at that budget
every arm sat at "no better than predicting no change" and it read as though
the model could not learn. At 6,000 the from-scratch arm reached 0.807
against no-change and was **still improving** — 0.918, 0.921, 0.857, 0.807 —
so 6,000 is a floor rather than a tuned value. Three arms at 6,000 steps took
about 96 minutes on the server, which is affordable inside §0's ~12-hour
budget.

**Reason for writing `lr` out.** It is unsettled and a default hides that.
The `core_unfrozen` arm was unstable at 1e-3 **and** at 3e-4 — it swung to
2.46x its own best mid-run — so lowering it did not fix the instability and
3e-4 is not established as better. Having the key present in the file is also
what stops a hand-edit being lost to a `git pull`, which is how the 6,000-step
setting disappeared between one run and the next.

**A run's own config is the record.** `runs/<name>/config.yaml` holds the
resolved config the run actually used, which is what settled what the
6,000-step run had been given after the file itself had changed underneath
it. Grepping the config *file* answers a different question.
