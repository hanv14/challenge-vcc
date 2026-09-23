# DECISIONS.md

One entry per non-obvious choice: the decision, the reason, and the
alternative that was not taken. Deviations from `CLAUDE.md` are marked
**DEVIATION** and are also listed in `PLAN.md` §9 and in the final message.

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
