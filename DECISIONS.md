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

### D79. The warm start itself is the damage, not the forgetting guard

**Measured**, `configs/server_isolate.yaml`: L2-SP and replay off for every
arm, 6,000 steps each, budgets equal by construction.

| arm | best `pert ÷ no-change` | pert pearson | map pearson | `delta ÷ no-change` |
|---|---|---|---|---|
| `core_unfrozen` (warm) | 0.9893 | 0.124 | 0.007 | 0.998 |
| `core_frozen` (warm) | 0.9878 | 0.139 | 0.029 | 0.994 |
| **`core_scratch`** (cold) | **0.7955** | **0.452** | **0.122** | **0.830** |

`phase1_contribution` = **−0.194**, against −0.274 with the guards on. So
removing L2-SP and replay recovered about **0.09** of a **0.27** gap. The
remaining **0.19 is the initialization**, and it is the larger part by far.

Whether the core is frozen barely matters once it is warm-started (0.9878
against 0.9893): the LINCS initialization dominates both.

**A prediction I made and got wrong.** I said that if L2-SP had been
*stabilising* `core_unfrozen`, removing the anchor would make its swing
worse. It made it much better — `worst / best` went from **2.28x to 1.03x**.
L2-SP and replay were *causing* the instability, not restraining it: an
anchor pulling toward Phase 1's weights while the data pulls away is a
tug-of-war, and replay alternating objectives every fourth step is the other
half of it.

**What follows.** `phase2.warm_start: false` is now justified by measurement
rather than by argument. It makes the shipped model two-phase, which is a
deviation from CLAUDE.md §4's design and is recorded as one — Phase 1 still
trains and is still scored, it simply stops feeding Phase 2.

**What is still unknown, and cheap to find out.** Whether Phase 1 learns
anything on LINCS at all. If its own held-out validation sits at no-change,
then its core is a badly conditioned random initialization and every result
above follows immediately. `runs/server/phase1/metrics.json` answers it.

### D80. Phase 2 can inherit Phase 1's network without its per-gene table

**The mechanism, from `runs/server/phase1/metrics.json`.** Phase 1 has
5,652,139 trainable parameters, and **4,744,448 of them — 84% — are the gene
vocabulary's `delta`**. Phase 1 only ever tokenizes the 955 panel genes, so
`delta[feature]` moves for those 955 and stays at zero for the other ~17,578;
`delta[target]` moves for the 4,968 targets it sees. `save_core` stores the
whole state dict, so a full warm start hands Phase 2 an embedding table whose
two halves sit on different footings — and Phase 2's entire job is mapping
the panel onto the rest genes. The from-scratch arm has no such asymmetry:
there every `delta` is zero, which is what §4.1 specifies it starts at.

**Decision.** `phase2.warm_start` becomes `full | without_delta | none`
rather than a bool. `without_delta` loads the checkpoint and then zeroes the
vocabulary's deltas, transferring the network Phase 1 learned without the
per-gene memorization it learned it on. `configs/server_nodelta.yaml`
differs from `server_isolate.yaml` in exactly that one setting.

**Why this is worth a run before abandoning Phase 1.** D79 established that
the warm start is the damage, but not which part of it. If `without_delta`
lands near the scratch arm's 0.80, the three-phase design survives and this
becomes the shipped setting; if it stays near 0.99, the core weights
themselves are the problem and `none` is the answer.

### D81. Phase 1 reports its no-change baselines, like Phase 2 always has

**Decision.** `phase1.evaluate` reports `*_mse_no_change` and
`*_mse_ratio_to_no_change` for the `sig`, `gmt` and `pert` heads. "Nothing"
is zero for `sig` and `gmt`, which are changes, and the control profile for
`pert`, which is a level the head predicts a change on top of — the same
convention as Phase 2's `pert_mse_no_change`.

**Reason.** Phase 1's numbers could only be read against each other. On the
server it reported `sig_mse 0.450`, `sig_pearson 0.120`, `gmt_pearson 0.006`
and `pert_pearson 0.353` — and nothing in that says whether any head beats
predicting nothing, which is the first question to ask of a phase whose
output is about to initialize another one.

**What the same file already shows without any new instrumentation.** Phase 1
trains for 400 steps at batch 32 — **12,800 row draws against 42,889 training
rows, under a third of one epoch.** Whatever it contributes, it contributes
without having seen its own data once.

### D82. `validation_per_context` is the saved model; the arm table is the best

**Decision.** `phase2/metrics.json` gains
`validation_per_context_describes` — which arm, which step, the arm's best
step, and whether they are the same — and the summary prints a line when they
are not.

**Reason.** D74 made `arms[...]["validation"]` each arm's *best*, but
`validation_per_context` still evaluates the main arm's *final* model,
because that is the checkpoint saved and the one Phase 3 inherits. On the
isolation run `core_unfrozen` peaked at step 4500 and was evaluated per
screen at 6000, so the file reported `pert_pearson` **0.124** in one table
and **0.007** in another, for what a reader would reasonably take to be the
same thing. Both numbers are correct and they describe the same arm at two
different moments.

Keeping the final model here is deliberate: it is what actually ships. The
fix is to say so, not to quietly switch which model is reported.

### D83. Per-screen noise floors differ enough that the aggregate hides them

**Observed**, isolation run: `map_delta_noise_floor_ratio` is **0.746** on
K562_essential against **0.484** aggregated over the three screens. On that
screen a *perfect* delta predictor could only reach 0.746, so the entire
measurable band is 0.746–1.0 — a quarter of the range the aggregate implies.

Nothing is wrong: the floor is measured per screen and reported per screen,
and averaging ratios across screens is sound. But a reader who compares an
aggregate ratio against the aggregate floor is averaging over screens whose
headroom differs by a factor of two, and will read a screen-specific
limitation as a model failure.

**Left as it is, deliberately.** The per-screen numbers are already in
`validation_per_context`, which is where this was found. Raising
`phase2.delta_eval_cells` above 128 would lower the floors — the noise in a
difference of sampled means falls as 1/n — at a cost paid only at evaluation
time, and it is worth doing if K562_essential has at least 512 control cells.
That is a fact about the server's data which this environment cannot check.

### D84. Every ablation arm is re-seeded immediately before it is built

**Decision.** `train_arm` calls `seed_everything(cfg.seed)` immediately before
`build_model`, and each arm records a digest of the weights it was built with
(`init_fingerprint`). `phase1_contribution` reports the digests and
`arms_share_initialization`, and Phase 2 warns when they disagree.

**Reason.** The run was seeded once, at `cli.py`'s start. Every arm then drew
its initial weights from a *global* torch stream that the preceding arms had
already advanced — so an arm's initialization depended on which arms ran
before it and how many steps they took. The arms in one run were therefore
not comparable to each other, and the same arm was not comparable across two
runs whose arm lists differed.

This was not hypothetical. `core_scratch` never warm-starts, so
`warm_start: full` and `warm_start: without_delta` cannot touch it; it is the
identical arm under both settings. It scored 0.7955 in the isolation run and
1.0057 in the `without_delta` run — a difference of 0.118 against a measured
"Phase 1 hurts" effect of 0.194. Half the effect we were about to write into
`DECISIONS.md` as a §4 deviation was the seed.

Treating the three server runs as repeats of the same measurement:

| arm | A (guards on) | B (isolate) | C (nodelta) | mean | sd |
|---|---|---|---|---|---|
| `core_unfrozen` | 1.0804 | 0.9893 | 0.9956 | 1.0218 | 0.0509 |
| `core_frozen` | 1.0100 | 0.9878 | 0.9908 | 0.9962 | 0.0120 |
| `core_scratch` | 0.8068 | 0.7955 | 0.9134 | 0.8386 | 0.0651 |

The gap (scratch − unfrozen) is −0.183 with a standard error of 0.048 over
three runs: about 3.8 standard errors, so the effect probably survives — but
it could not be read off any single run, and nothing in the artifacts said so.

**Alternative.** Seed per arm from `cfg.seed + hash(arm)` so arms differ
deliberately, and average over repeats. That measures the same thing with
more runs; re-seeding identically removes the variance instead of paying to
average it away. Repeats are still the right way to check a result, which is
why the table above is in this entry rather than a single run's number.

**Consequence.** Numbers from before this change are not directly comparable
with numbers after it, and `phase1_contribution` from any earlier run should
be read as one draw from a distribution with sd ≈ 0.05, not as a measurement.

### D85. A phase keeps the weights it measured as best, not the last ones

**Decision.** `train.checkpoint_selection` (default `best`). A phase that
selects on a validation metric keeps a CPU copy of the weights at its best
step and loads them back before saving. The arm record says `weights_kept`,
and the checkpoint's `extra` carries the validation of the weights it
actually holds. `final` stays available.

**Reason.** Three seeded repeats of Phase 2 on the server (`--seed 0/1/2`,
6000 steps, guards off) showed arms that learn and then lose it:

| arm | best | at step | last step | `final / best` | still there at the end |
|---|---|---|---|---|---|
| seed 2 `core_scratch` | 0.8171 | 3000 | 1.0013 | 1.225 | **−1%** |
| seed 1 `core_scratch` | 0.9171 | 1500 | 1.0642 | 1.160 | **−77%** |
| seed 0 `core_scratch` | 0.9627 | 3000 | 0.9816 | 1.020 | 49% |

Seed 2's arm reached `pert_pearson` 0.42 at step 3000 and ended at 0.02.
The checkpoint saved was the one that had forgotten. Phase 3 inherits
`core_phase2.pt`, so this is not bookkeeping: it is what gets submitted.

The same runs showed the instability detector missing all of it. On a metric
that is a **ratio against predicting no change**, 1.0 is the no-change line
and `final / best` is the wrong scale: an arm that gives back everything it
learned reads as 1.225, under the 1.25 threshold, and `diverged` said false.
So `TrainResult` gained an `anchor` and `signal_retained = (anchor − final) /
(anchor − best)` — 1.0 ended at its best, 0.0 ended knowing nothing more than
the baseline, negative ended worse than that. Phase 2 passes `anchor=1.0`,
flags an arm below 50%, and the summary tables those arms out.

`signal_retained` is `None` when the arm's *best* is already worse than the
anchor: there is no learning to have kept, and the ratio would otherwise
invert — a mini arm whose best was 1.0844 against a baseline of 1.0 reported
keeping 116% of what it learned before this was guarded.

**Alternative.** Early stopping — end the run at the best step. Rejected: the
budget is a config number the user tunes, and stopping early would hide that
the run does not converge. Keeping the best weights *and* reporting the swing
shows both.

**Consequence.** Keeping the best is selection on a validation set, so the
selected number is optimistic by the usual amount; it is held out by gene,
and the alternative was shipping a model we had measured and knew to be
worse. It also means `validation_per_context` and the arm table now describe
the same weights, which they did not before (D82 said so; this removes the
mismatch rather than annotating it).

### D86. Phase 1's contribution is not measurable at this budget

**Decision.** Keep CLAUDE.md §4's three phases. No deviation.

**Reason.** With the seeding fixed (D84), three repeats that all report
`arms_share_initialization: true`, gap = scratch − warm on
`pert_mse_ratio_to_no_change`, negative means Phase 1 hurt:

| read at | seed 0 | seed 1 | seed 2 | mean | se | t |
|---|---|---|---|---|---|---|
| each arm's best | −0.0199 | −0.0798 | −0.1734 | −0.0910 | 0.0447 | −2.04 |
| the last step | −0.0009 | +0.0671 | +0.0036 | +0.0233 | 0.0220 | +1.06 |

The sign depends on which step you read. Worse, "best of four evaluations"
favours the arm with the larger variance, and `core_scratch` is exactly that
arm — it swings between 0.82 and 1.06 while the warm arms sit between 0.98
and 1.00. So the −2.0 SE is an upper bound on the evidence, not a measurement.

The earlier three runs gave −0.183 at 3.8 SE; that was the seeding defect.
The honest statement is that Phase 1 neither clearly helps nor clearly hurts
Phase 2 at 6000 steps, and nothing here justifies dropping a phase the
specification asks for.

**What the repeats did show**, consistently across all three seeds: the warm
arms barely move. `pert_pearson` for `core_unfrozen` reaches 0.11–0.18 and
`map_delta_ratio_to_no_change` never leaves 0.99–1.00 against a noise floor
of 0.46–0.56. The scratch arm reaches 0.25–0.42 before collapsing. The
question worth the next server cycle is therefore not "is Phase 1 worth it"
but "why does Phase 2 end at the no-change line in every arm" — which is
D85's subject, not this one's.

**Alternative.** More repeats. At sd ≈ 0.077, separating a 0.05 effect at 2 SE
needs about ten runs of 4.5 hours each. That is a cycle and a half of server
time spent on a question that does not change what is submitted, while the
convergence problem does.

### D87. `train.evals_per_run`, and `--set` for one-knob sweeps

**Decision.** How often a phase evaluates is a config key (default 4, as
before), and any single config value can be overridden from the command line:
`--set train.lr=0.0001`, repeatable.

**Reason.** The evaluation count is not a logging detail. It is the resolution
at which a run's shape exists at all: `checkpoint_selection` can only keep a
step it evaluated, and `signal_retained` can only be measured across the
evaluations there are. Four over 6,000 steps showed an arm at 0.8171 and then
at 1.0013 with nothing between them — enough to know it collapsed, not enough
to see when it started. It was `steps // 4`, hardcoded in the training loop.

`--set` exists because the alternative was a fourth server config file
differing from the third by one line. Near-identical configs drift apart, and
then a comparison is between two things nobody can diff. Values are parsed as
YAML and coerced like file values, so `--set train.lr=3e-4` does not walk into
D75's trap, and an unknown section or key fails before any work starts. The
override is written into the run directory's `config.yaml`, so the run still
describes itself.

**Alternative.** Environment variables, or a `sweep:` block in the config.
Both put the setting somewhere other than where the run records what it did.

**Also.** `configs/server_isolate.yaml` now turns both ablation arms off by
default (D86 settled what they measured; one arm is 45 minutes instead of
2.2 hours), evaluates eight times, and its header carries the three-run
result rather than the question it was written to ask. And a test now loads
every config in `configs/`, which is what would have caught D75's `3e-4`
before it reached the server.

### D88. The selection metric is scored over 256 targets, not over `batch_size`

**Decision.** `phase2.eval_targets` (default 256, `0` = every held-out target
with a pseudobulk row). The perturbation module's validation used to score
`usable[: cfg.phase2.batch_size]` — the *training* batch size, borrowed by
accident — so every reading rested on **32 of the 1,907** held-out targets
K562_gwps offers, in list order. The forward pass now runs in blocks of
`batch_size`, which changes how many rows go through at once and nothing else.

**Reason.** `pert_mse_ratio_to_no_change` chooses the checkpoint (D85),
decides the core-freeze and Phase 1 ablations (D86), and is what
`signal_retained` is measured on. A sample of 32 is small enough that its
noise is the same size as the effects being read off it: three seeded repeats
disagreed by an sd of 0.077 where the gap being measured was 0.091, and the
"best of N evaluations" rule that D85 introduced selects partly on that noise
— the more evaluations, the more of it is selected on.

The seeds also change *which* 32, because the target split is seeded, so the
run-to-run spread is partly a different validation set each time rather than a
different model. Within a run the arms share the split, which is why the
paired gap was the only reading worth quoting at all.

**Cost.** Forward passes at evaluation time, nothing else: 256 targets is one
extra matrix of (256 x n_panel) per screen per evaluation, against the cell
sampling and the mapping pass that dominate. The delta metrics keep their own
`delta_eval_targets` (8) because those draw 128 cells per target.

**Alternative.** Resample the scored targets each evaluation. Rejected: that
adds noise *between* evaluations, which is exactly what the checkpoint rule
must not select on. A fixed, larger, deterministic set is comparable across
evaluations, arms and runs.

### D89. Lowering the learning rate does not fix Phase 2; it trains less

**Decision.** `train.lr` stays at 3e-4. The convergence problem is not the
step size.

**Reason.** D85 read the collapse (0.8171 at step 3000, 1.0013 at 6000) as a
rate that was too large, and predicted a lower one would hold the peak. It
does not. One arm, seed 2, 6000 steps, eight evaluations:

| `train.lr` | best | at step | last | `pert_pearson` last |
|---|---|---|---|---|
| 3e-4 | 0.9905 | 3000 | 0.9977 | 0.1103 |
| 1e-4 | 1.0052 | 6000 | 1.0052 | 0.1329 |
| 3e-5 | 1.0213 | 750 | 1.0450 | 0.0773 |

Monotonically worse, and neither lower rate ever beat the no-change baseline
at all — `signal_retained` reports `None` for both, which is the guard added
in D85 saying exactly that. The warm arm sits at ~1.0 at every rate tested.

What this rules out matters as much as what it shows: the warm arm's failure
to learn is not an optimization-step-size problem, so the next thing to vary
is not another training knob. It is the measurement (D88) and then the warm
start itself — at 3e-4 the only configuration that has ever reached a real
signal is `core_scratch` (0.8171, `pert_pearson` 0.42), and since D85 that
peak is kept rather than thrown away, which it was when D86 called the choice
immaterial.

**Alternative.** A schedule (warmup, cosine decay) rather than a flat rate.
Worth trying, but only after D88: choosing between schedules on a metric
computed over 32 targets is how the last three cycles were spent.

**Superseded — do not cite this entry as evidence** (see D98). Both runs above
were measured on the 32-target metric, before D88 fixed it, *and* on
`warm_start: full`, which D90 then showed does not learn at any setting.
Lowering the learning rate of a flat line gives a flatter line. Nothing here
says anything about the arm that does learn.

### D90. The warm start prevents Phase 2 from learning. D86 is superseded

**What the measurement now says.** One run, seed 2, three arms sharing an
initialization (`arms_share_initialization: true`), no replay, no L2-SP, the
same 6000 steps, and `pert_mse_ratio_to_no_change` scored over **908**
held-out targets instead of 32 (D88):

| step | `core_scratch` | its pearson | `core_unfrozen` | its pearson |
|---|---|---|---|---|
| 750 | 0.9195 | 0.275 | 1.0545 | 0.125 |
| 1500 | 0.8976 | 0.278 | 1.0487 | 0.174 |
| 2250 | 0.8348 | 0.372 | 1.0360 | 0.154 |
| 3000 | 0.8209 | 0.400 | 1.4475 | −0.021 |
| 3750 | 0.8068 | 0.432 | 1.0093 | 0.150 |
| 4500 | 0.7809 | 0.450 | 1.0133 | 0.179 |
| 5250 | 0.7734 | 0.463 | 1.0575 | 0.165 |
| 6000 | **0.7585** | **0.472** | **0.9911** | 0.161 |

`core_scratch` falls monotonically at all eight evaluations and is still
falling at the last one. `core_unfrozen` sits at the no-change line
throughout. The same holds on the training targets (0.9176 → 0.7192 against
0.9911), so this is not regularization or overfitting: the warm arm cannot
fit the targets it is trained on. And on the mapping's delta, which is what
the official metrics score, `core_scratch` reaches 0.7950 against a noise
floor of 0.4625 — **38% of the way from predicting nothing to the best a
predictor could do at these cell counts** — while the warm arm reaches 0.1%.

**This is not the reading D86 refused.** D86 declined to call a gap of
−0.091 ± 0.045 measurable, and was right to: it was a best-of-N difference
between two noisy flat lines, scored over 32 targets. What is here is a
monotone curve against a flat one, on a 28x larger sample, with
`signal_retained` 1.0 for both arms — neither one passed through its best,
so no selection rule is doing the work. The gap is −0.2326. D86 is
superseded; the question it could not answer was a measurement problem
(D88), not a small effect.

**What is not yet decided: whether this costs a deviation from §4.** Two
things make it smaller than it sounds.

First, `core_scratch` is not "Phase 2 without LINCS". Three of the gene
vocabulary's prior blocks are built from `phase1_lincs.h5ad` — signature
co-variation, GMT co-membership, and the target-role phenotype from
`layers['sig']` — in the `priors` stage, from the data file rather than from
Phase 1's trained weights. Every arm carries them. What `warm_start: none`
removes is the transfer of *weights*, not of LINCS.

Second, `without_delta` has not actually been tested. D80 and the run behind
D84 measured it, but with unseeded arms and the 32-target metric, so that
verdict is void — it was taken on the instrument this entry's numbers
replace. The per-gene table is 84% of the trainable parameters and Phase 1
trains it on 955 panel genes alone, which is the obvious suspect. One 45
minute arm settles whether §4's warm start can be kept in a form that helps.

**The default is therefore unchanged** pending that arm. Shipping a
deviation from the specification when a cheap test might avoid it is the
wrong order.

**Also visible in the same run, and not yet explained.** The per-cell
objective is the one thing that does *not* improve:
`pert_percell_ratio_to_no_change` is 1.0437 for `core_scratch` at step 6000,
slightly worse than predicting no change, while the pooled question it is
scored beside goes to 0.7585. The OT pairing is not yet delivering what
PLAN_PERCELL §3 asks of it, and `scripts/coupling_check.py` has still never
been run on the server. That is the next question after the warm start.

**And a budget note.** `core_scratch` was still improving at step 6000, so
`phase2.steps` is now the binding constraint rather than the learning rate
(D89). At roughly 5.5 minutes per 750 steps, 12,000 steps is about 1.5 hours
for one arm.

### D91. `phase2.warm_start: none` is the default. Deviation from §4, measured

**Decision.** Phase 2 starts from the gene vocabulary priors, not from Phase
1's checkpoint. This is a **deviation from CLAUDE.md §4.2's "one core used by
all three phases"**, reported by checklist item 5 on every run that uses it,
by a warning at the start of Phase 2, and here.

**Reason.** All three ways of handing Phase 1's weights over were measured on
the same seed, the same split, the same 6000 steps, arms sharing an
initialization, no replay, no L2-SP, and `pert_mse_ratio_to_no_change` scored
over 908 held-out targets (D88):

| step | `full` | `without_delta` | `none` |
|---|---|---|---|
| 750 | 1.0545 | 1.0942 | 0.9195 |
| 1500 | 1.0487 | 1.2946 | 0.8976 |
| 2250 | 1.0360 | **0.9984** | 0.8348 |
| 3000 | 1.4475 | 1.0066 | 0.8209 |
| 3750 | 1.0093 | 1.0064 | 0.8068 |
| 4500 | 1.0133 | 1.0111 | 0.7809 |
| 5250 | 1.0575 | 1.0187 | 0.7734 |
| 6000 | **0.9911** | 1.1554 | **0.7585** |

`none` falls monotonically at all eight evaluations and is still falling at
the last. Neither warm form leaves the no-change line, and the same holds on
the *training* targets (0.7192 against 0.9911 and 1.1444), so this is not
regularization: the warm arms cannot fit what they are trained on. On the
mapping delta, `none` reaches 38% of the way from no-change to the noise
floor where the warm arms reach 0.1%.

`without_delta` was the hypothesis that could have saved the warm start —
`delta` is 84% of the trainable parameters and Phase 1 trains it on the 955
panel genes alone. It is refuted: resetting the per-gene table changes
nothing. The damage is in the core weights themselves.

**What the deviation does and does not remove.** Phase 1 is still trained,
still scored on held-out targets, still re-scored after Phases 2 and 3 for
the forgetting report, and still reaches Phase 2 — three of the gene
vocabulary's prior blocks are built from `phase1_lincs.h5ad` (signature
co-variation, GMT co-membership, and the target-role phenotype from
`layers['sig']`), in the `priors` stage, from the data file rather than from
Phase 1's trained weights, and every arm carries them. What is removed is the
transfer of *weights* between phase 1 and phase 2.

**Alternative, not taken yet.** Transfer only the perturbation module —
Phase 1's type embeddings and target-role parameters — and initialize the
rest from the priors. That is the form of transfer §4's design is really
after (LINCS knows what a knockdown does; Replogle knows what cells look
like), and it is one 45-minute arm to test. It is not built because it is a
new mode the specification does not ask for, and because the pipeline needed
a configuration that learns before it needed a better one.

**Also.** The instability warning printed "-9922% of what it learned was
still there" for the `without_delta` arm, whose best beat the baseline by
0.0016. The ratio is correct; as a percentage it is arithmetic, not
information. `signal_headroom` is now recorded beside `signal_retained`, and
below 0.01 of headroom the message reports the headroom instead of a share.

### D92. `warm_start: perturbation_only` — the transfer §4 is actually after

**Decision.** A fourth `phase2.warm_start` mode. Phase 2 builds from the
priors as `none` does, then keeps Phase 1's values for four tensors only: the
target-role projection and per-gene table (`vocabulary.projections.target.*`,
`vocabulary.deltas.target`) and the assay-type vocabulary (`pert_type_base`,
`pert_type_delta`). Everything else is put back to the fresh initialization.
Not the default — it is an experiment until it is measured.

**Reason.** D91 established that handing Phase 1's weights to Phase 2 stops
Phase 2 learning, in both forms the code could hand them over. But "Phase 1's
weights" is not one thing. Phase 1 sees 688 LINCS knockouts against 955 panel
genes, on bulk Level-3 profiles. What it can plausibly know that Replogle
cannot is **what perturbing a given gene does** — which is exactly the
target-role embedding and the type vocabulary that `models/core.py: tokenize`
builds the perturbation token from. What it cannot plausibly know is what a
single cell looks like, and that is everything else: the value encoder, the
latent blocks, the decoder, the feature-role embedding. Those are the weights
measured to do the damage.

So the two settings that failed both transferred the second along with the
first. This transfers the first alone, which is the claim §4's design is
really making — and it is the only version of that claim still standing.

**How it is implemented.** The fresh weights are snapshotted before the
checkpoint is loaded, and restored afterwards for every tensor that is not
in `PERTURBATION_PARAMETERS`. The alternative — loading the checkpoint into a
filtered state dict — would depend on what `load_core(strict=False)` happens
to match; snapshot-and-restore is exact either way, and a test asserts every
non-perturbation tensor comes back bit-identical.

L2-SP anchors to the model's state *after* the restore, so it pulls toward
where this arm started rather than toward Phase 1, which is what the guard
means here. Replay stays on, because there is now something of Phase 1 to
forget.

**Not measured yet.** It is one 45-minute arm against `core_scratch`'s curve
(`configs/server_isolate.yaml` carries both). If it beats 0.7585, CLAUDE.md
§4's three phases come back and the D91 deviation goes away; if it matches
`none`, Phase 1's weights have nothing to give Phase 2 in any form and D91
stands as the final answer.

### D93. A NaN is not a score, and the checklist does not call it one

**Decision.** The forgetting stage marks each core `measurable` and says why
when it is not; the core-freeze ablation refuses to compare non-finite
scores; and checklist item 7 reports a `deviation` when Phase 1 validation
could not be re-scored, rather than `done` because the file exists.

**Reason.** The first full server run under `warm_start: none` logged:

```
  phase2                   sig_pearson +nan
  phase3                   sig_pearson +nan
  core-freeze ablation: core_frozen is better on sig_pearson by +nan
      ->  core_frozen retains more of Phase 1 than the configured default
          (core_unfrozen); consider flipping phase2.unfreeze_core
```

Three things wrong. The NaN is not a bad score: the Phase 1 head is
zero-initialized, so a core that never carried Phase 1's weights predicts a
constant, and the Pearson of a constant has no value. `main >= other` with a
NaN on either side is always false, so the ablation silently declared the
*other* arm better and recommended flipping a config key on the strength of
it. And item 7 of §4.8 — "Phase 1 validation re-scored after Phase 2 and
after Phase 3" — read `done`, because `forgetting.json` existed.

**What it means for the run, not just the report.** Under D91's
`warm_start: none` there is nothing of Phase 1 in Phase 2's core to forget,
so §4.3's guard has nothing to guard and item 7 cannot be satisfied as
written. That is the D91 deviation surfacing a second time, and the honest
place to say so is the checklist, which now does.

**Alternative.** Score the phase 2 and 3 cores with Phase 1's *trained* head
grafted on. Rejected: that measures a model the run never had, and the
question item 7 asks — what does the saved core still remember — has the
answer "nothing, by construction" under this setting. Saying that is better
than manufacturing a number for it.

### D94. Phase 3 adapts only if the rehearsal measured that adapting helps

**Superseded by D109/D112.** The arm this policy was read from started from
Phase 1's core, not from a trained Phase 2. Measured on the shipped recipe,
adapting is neutral (0.985 against 0.986 unadapted), and the policy lets
Phase 3 adapt again. The mechanism, a policy read off the rehearsal, stands.

**Decision.** `phase3_policy.json`'s `adapt` block is read off rehearsal
variant 1 instead of being written in. When the controls-only mapping scores
worse than predicting no change (median above `CONTROLS_ONLY_TOLERANCE`,
1.02), `context_adapters` and `delta` are both false, and Phase 3 measures
the mapping without touching it — `adapt_<context>.json` is still written,
with `steps: 0` and equal loss before and after. `POLICY_VERSION` is 2, so a
policy from before this is refused rather than obeyed.

**Reason.** §4.6 says the policy "states which modules Phase 3 may adapt …
chosen from these results". It was a literal transcription of §4.7 instead:
`context_adapters: True, delta: True`, hardcoded, with a comment explaining
why that is reasonable. Variant 1 exists precisely to test it — it fits the
mapping on a held-out context's controls alone, which *is* Phase 3's
procedure, and scores the rest genes it then predicts. The first full server
run measured, per context, against a floor of 1.0:

| context | method | upper bound (perturbed cells allowed) |
|---|---|---|
| K562_essential | 1.484 | 0.989 |
| K562_gwps | 1.509 | 1.002 |
| rpe1 | 1.468 | 0.890 |

Adapting the mapping on controls alone makes the rest genes about 50% worse
than leaving them at no change, where the same mapping given perturbed cells
sits at 1.0 or better. The seed-2 run measured the same thing worse still:
2.030, 2.167 and 2.505.

**Correction.** This entry originally also cited Phase 3's own before/after
numbers — A 1.0091 → 0.8786, B 0.7208 → 0.8422, C 0.8699 → 0.9568 — as two
of three contexts being made worse by their own adaptation. That evidence
was worthless: the two measurements drew different control cells, so their
difference was sampling noise (D96). The rehearsal variant, which scores 100
targets against a floor, is what this decision rests on, and it is
unaffected.

**This is a departure from §4.7's described procedure**, taken on §4.6's
explicit instruction that the policy be chosen from the rehearsal. It is
recorded in the policy file itself, with the numbers, so a reader can see
which of the two sections the run followed and why. When variant 1 does not
run, the policy falls back to §4.7 as written and says `measured: false`.

**Alternative.** Adapt anyway and let the generator's confidence threshold
suppress the damage. Rejected: the threshold works on predicted fold changes
per gene, and a mapping that is uniformly worse does not announce itself
gene by gene. The cheaper and more honest fix is not to do the thing that
was measured to hurt.

**What it does not change.** The core, the perturbation module and the heads
are never adaptable whatever the rehearsal says: §4.7 is explicit, and
nothing here measures them.

### D95. `perturbation_only` helps at first and then holds Phase 2 back

**Decision.** `warm_start` stays at `none`. `perturbation_only` remains
available and is worth another look if the instability is fixed, but it is
not the default.

**Reason.** Measured on seed 2, one arm, the same config as everything else
in D90–D91, against `none` on the same seed:

| step | `none` | `perturbation_only` |
|---|---|---|
| 750 | 0.9195 | **0.8815** |
| 1500 | 0.8976 | 0.9301 |
| 2250 | 0.8348 | 0.9232 |
| 3000 | 0.8209 | 0.9683 |
| 3750 | 0.8068 | 0.8832 |
| 4500 | 0.7809 | 0.9308 |
| 5250 | 0.7734 | 0.9468 |
| 6000 | **0.7585** | 1.0275 |

Phase 1's perturbation module is worth something: at step 750 the warm arm is
ahead, 0.8815 against 0.9195, with `pert_pearson` 0.299 against 0.275. It is
the first evidence in this project that anything of Phase 1 transfers. But it
then stops improving and drifts back to the no-change line, where `none`
falls monotonically past it and keeps going. Best against best, 0.8815
against 0.7585.

So the reading is not "Phase 1's perturbation module is useless" but "it is a
better starting point than the priors and a worse place to stay". That is the
signature of a constraint, not of bad information.

**The likely constraint, not yet tested.** Phase 2 runs with
`projections=False`: `vocabulary.projections.target.weight` — the `W` of
§4.1's `embedding = W·prior + δ` — is **frozen for the whole of Phase 2**,
in every warm-start mode. Phase 1 trains it (`projections=True`), so a warm
arm inherits a `W` fitted to LINCS bulk and can never move it, while a cold
arm gets the prior-basis initialization and also cannot move it. That would
explain both halves of the table: the transferred target-role information
helps immediately, and the projection it arrives through is locked to the
wrong assay. `phase2.train_projections` (default off, so this changes
nothing until it is asked for) makes that one 45-minute arm.

**Alternative.** Lower the learning rate for the transferred tensors alone,
so they are nudged rather than overwritten. Rejected for now: the arm is
unstable in the same way every arm here is (best at step 750, ending 1.17x
worse), so the first thing to try is the constraint, not a schedule.

### D96. Loss before and after are measured on the same cells

**Decision.** `adapt_on_controls` measures the mapping through one helper
that builds a fresh generator from `cfg.seed` each time, so "before" and
"after" draw the identical control cells and their difference is what
adapting did.

**Reason.** The two calls shared the training `rng`, which the first call had
already advanced, so they drew different cells. The seed-2 run made that
visible by accident: the policy set `steps: 0` (D94), the model provably did
not change, and the report still showed

| context | before | after | "improvement" |
|---|---|---|---|
| A | 0.7818 | 0.8635 | −0.0817 |
| B | **1.1076** | **0.6941** | +0.4135 |
| C | 0.8903 | 0.9364 | −0.0460 |

B moved by 0.41 between two measurements of the same unchanged model. The
draw is 32 control cells, so this is sampling noise reported as an effect —
and checklist item 11's artifact is exactly "loss before and after", which
means the artifact was not measuring anything.

**What it invalidates.** The seed-0 run's "two of three contexts made worse
by adapting" (cited in D94) was this noise, and D94 now says so. The
decision itself stands on rehearsal variant 1, which scores 100 targets
against a floor and is unaffected.

**Alternative.** Average several draws instead. Unnecessary once the
comparison is paired: the cell-to-cell variation cancels, which is the whole
reason to hold the draw fixed rather than to average it away.

### D97. A `.vcc` from an earlier run is not this run's submission

**Decision.** The `package` stage deletes an existing `prediction.vcc` before
calling `vcc prep`, and checklist item 13 reports a deviation when the `.vcc`
on disk is older than the `prediction.h5ad` beside it.

**Reason.** The seed-2 server run ended:

```
16:17:47 ERROR   vcc exited 1: ['vcc prep: Output already exists:
                 .../submission/prediction.vcc. Re-run with --force ...']
16:17:47 ERROR   `vcc prep` did not write a .vcc; prediction.h5ad remains the deliverable
...
  13. [  done   ] Submission and validation, `vcc prep` where available
```

The tool refuses to overwrite, so the `.vcc` from the *previous* run stayed
on disk — packaged from a different model's cells — and item 13 read `done`
because a file with the right name existed. The run's own log said the
packaging had failed two lines earlier. Uploading that file would have
submitted the wrong predictions with no warning anywhere.

Deleting first rather than passing `--force` is deliberate: after a failure
for any other reason there is then no `.vcc` at all, so the checklist's
existence test is honest by construction rather than by a second check. The
mtime test covers the other path, where `package` is not rerun at all.

**Alternative.** Keep the old file and name the new one after the run. That
multiplies files whose only difference is which model made them, in a
directory the user uploads from. One current file is safer.

### D98. Freeing the projection makes Phase 2 learn faster and then collapse

**Decision.** `phase2.train_projections` stays off. D95's explanation of the
`perturbation_only` arm is refuted.

**Reason.** Four arms, seed 2, guards off, 6000 steps, 908 held-out targets —
every one of them the same except for two switches:

| step | `none`, W frozen | `none`, W trained | `pert_only`, W frozen | `pert_only`, W trained |
|---|---|---|---|---|
| 750 | 0.9195 | 0.8454 | 0.8815 | **0.8295** |
| 1500 | 0.8976 | **0.8061** | 0.9301 | 0.9716 |
| 3000 | 0.8209 | 1.0376 | 0.9683 | 0.9640 |
| 4500 | 0.7809 | 1.1040 | 0.9308 | 1.0799 |
| 6000 | **0.7585** | 0.9843 | 1.0275 | 1.0000 |

Training `W` did not rescue the warm start. It made *both* arms learn faster
at the start — 0.8454 and 0.8295 at step 750, against 0.9195 frozen — and
then destabilised both. Best against best, the fully constrained arm still
wins: 0.7585, and it is the only one of the four that is monotone, and the
only one still improving at the last step it was given.

**The pattern across all four, which is the real finding.** Every degree of
freedom added to Phase 2 — a transferred perturbation module, a trainable
projection, or both — buys faster early progress and costs convergence. The
arm that reaches the best value is the most constrained one, by a clear
margin, and it has not finished. So the question is no longer "what should
Phase 2 inherit" but "why does this optimisation diverge the moment it has
room to", and the cheapest thing to try is not another transfer: it is more
steps for the arm that is still going down.

**Correction to D89.** D89 concluded that lowering `train.lr` makes things
monotonically worse and that the step size is therefore not the problem.
Those two runs were measured on the 32-target metric (before D88) *and* on
`warm_start: full` — the arm that has since been shown not to learn at any
setting. Lowering the learning rate of a flat line produces a flatter line,
which is what was observed. D89's numbers are not evidence about the arm that
learns, and the lr question is open for it, behind the budget question.

**Alternative considered.** Training `W` with a smaller learning rate than
the rest. That is a third knob on an optimisation already failing on two, and
the arm that works needs neither.

### D99. The runs were not reproducible, and the comparisons rest on that

**Decision.** When `train.deterministic` is on, the fused attention backends
(cuDNN, flash, memory-efficient) are disabled and only the math kernel is
left, so `F.scaled_dot_product_attention` has a deterministic backward. The
run header now reports which backends are enabled and says plainly when the
run is **not** reproducible, instead of printing "deterministic kernels"
either way.

**Reason.** Every server log carried this line and none of us read it:

```
UserWarning: cuDNN Attention defaults to a non-deterministic algorithm. To
explicitly enable determinism call torch.use_deterministic_algorithms(True,
warn_only=False).
```

`set_determinism` called `use_deterministic_algorithms(True, warn_only=True)`,
which lets a non-deterministic op through with a warning rather than failing.
That was a deliberate choice for ops with no deterministic implementation —
but attention is in the model's inner loop, so its backward was
non-deterministic on every step of every run.

The 12,000-step run is what made it undeniable. Same seed, same config, same
code as the 6,000-step run, `warm_start: none`, W frozen:

| step | 6,000-step run | 12,000-step run |
|---|---|---|
| ~1,000 | 0.9195 (750) | 0.9178 |
| ~2,000 | 0.8976 (1500) | 0.8904 |
| ~3,000 | 0.8209 | **0.8820** |
| ~4,500 | 0.7809 | 0.8935 (4000) |
| ~6,000 | **0.7585** | 0.9015 |
| 12,000 | — | 1.0139 |

At step 6,000 one realisation reads 0.7585 and the other 0.9015. Nothing
differed but the arithmetic.

**What this costs.** The monotone curve that D98 built its argument on is one
realisation of a stochastic process, not a property of the configuration. So:

* **D98's four-arm pattern is weakened.** "Every degree of freedom costs
  convergence" was read off single runs whose run-to-run spread is now known
  to be at least 0.14 on this metric — larger than most of the gaps it
  compared. The arms may still differ; the evidence no longer shows it.
* **D95's `perturbation_only` reading is weakened** for the same reason.
* **D90 and D91 survive.** Those gaps are 0.2 and larger, they reproduced on
  the training targets as well as the held-out ones, and the flat arms were
  flat at every evaluation of every run. A 0.14 spread does not turn a flat
  line into a monotone descent.
* **D84's seeding fix is untouched** — it was about initialization, which is
  seeded, not about the arithmetic afterwards.

**Cost of the fix.** The math kernel materialises the attention matrix. Here
that is 64 latents against at most 956 tokens, and the model's expense is the
decoder over 17,578 genes, so the cost should be small — but it is unmeasured
in the cloud and worth watching on the first server run.

**Alternative.** `warn_only=False`, which raises on the first
non-deterministic op. Rejected: it would fail a twelve-hour run at hour
three over an op we could have simply avoided, and the avoidance is one call.

### D100. The coupling check computed a null arm and then ignored it

**Decision.** `verdict` compares the perturbed arm against the
control-against-control arm and returns `matches-the-null` when the perturbed
arm's per-cell target spread is under `NULL_MARGIN` (1.25×) of the null's.

**Reason.** The check ran on the server and returned `informative`: "the
barycentric targets vary 0.544 as much as the perturbed cells at epsilon
0.005 … The premise holds." It reached that verdict from the perturbed arm's
numbers alone. The null arm sat in the same file:

| epsilon | target spread, null | perturbed | ratio | residual removed, null | perturbed |
|---|---|---|---|---|---|
| 0.005 | 0.5131 | 0.5438 | 1.06 | −0.294 | −0.333 |
| 0.01 | 0.3113 | 0.3253 | 1.05 | −0.103 | −0.124 |
| 0.02 | 0.0987 | 0.1056 | 1.07 | +0.062 | +0.053 |
| 0.05 | 0.0118 | 0.0111 | 0.95 | +0.064 | +0.063 |

Pairing control cells against **other control cells** produces 95–107% of the
per-cell variation that pairing them against perturbed cells does, and
removes the same share of the residual — slightly *more* of it at the epsilon
the pipeline actually uses. Whatever the coupling is matching on, it matches
on it when there is no perturbation at all: sequencing depth and cell state,
which is real structure and is not the perturbation.

So the premise of PLAN_PERCELL §3 does not hold as implemented. This is the
reading that explains `pert_percell_ratio_to_no_change` sitting at 1.00 in
every run since the per-cell mode was built: the per-cell target is the
pooled target plus noise that is unrelated to what was perturbed.

**What it does not say.** The costs are not concentrated — `cost_spread`
0.198 against a null of 0.053 — so there is real structure between these
cells. The failure is not distance concentration, which is what the check was
originally written to catch. It is that the structure OT finds is not
perturbation-specific.

**Alternative.** Compare the residuals rather than the spreads. Both are in
the reading, and the spread is what decides, because the spread is what the
training step sees: it is the quantity that makes the per-cell loss differ
from the pooled loss at all.

### D101. The reliability is not the ceiling; its square root is

**Decision.** `split_half_reliability` reports `reliability` (the share of the
target's variance that is signal) and derives the two bounds it implies:
`max_achievable_pearson` = `sqrt(reliability)` and `best_possible_mse_ratio`
= `1 - reliability`. The verdict compares against the first, with
`MIN_ACHIEVABLE_PEARSON` (0.10) as the threshold below which per-cell
prediction is not worth pursuing.

**Reason.** D100's sibling diagnostic said "no predictor correlates with a
target better than the target correlates with itself" and reported the
Spearman–Brown reliability as the ceiling. That is wrong. Two fallible
measurements of one quantity correlate `r` with each other; a *noiseless*
predictor of the underlying signal correlates `sqrt(r)` with either of them —
the classical attenuation bound. At `r = 0.086`, the ceiling is 0.29, not
0.086.

The data caught it on the first server run. The ridge regression came back at
0.140, 0.146 and 0.188 pearson, **above** the claimed ceilings of 0.086,
0.086 and 0.101. A predictor cannot exceed a real ceiling, so either the
ridge or the bound was wrong, and it was the bound.

Had the error stood, the verdict rule — reliability under 0.05 means "not
predictable" — would have been one bad run away from declaring a task
impossible that a linear regression solves to half its ceiling.

**Alternative.** Report the reliability alone and leave the reader to square-
root it. Rejected: the number is only useful as a bound, the bound is what the
verdict tests, and the first person to use it got it wrong.

### D102. Phase 2's mapping reaches a fifth of what is available

**What the measurement says.** `scripts/mapping_ceiling.py` on the server,
8.6k–20k control cells fitted per screen, 2,048 held out, ridge optimum
interior to the grid in all three:

| screen | model | ridge | depth only | ceiling | model's share | ridge's share |
|---|---|---|---|---|---|---|
| K562_essential | 0.033 | 0.140 | 0.054 | 0.294 | **11%** | 48% |
| K562_gwps | 0.016 | 0.146 | 0.077 | 0.294 | **5%** | 50% |
| rpe1 | 0.061 | 0.188 | 0.065 | 0.317 | **19%** | 59% |

Pearson on held-out control cells; the ceiling is `sqrt(reliability)` (D101).
In error terms, against a no-change baseline of 1.0 and a best possible of
0.914, 0.914 and 0.899:

*(D105: the model column is `map_control_mse`, not a ratio. The 32 cells it
is scored on have their own no-change error, 0.97–1.07, so it cannot be read
against 1.0.)*

| screen | model mse ratio | ridge | best possible |
|---|---|---|---|
| K562_essential | 0.964 | 0.986 | 0.914 |
| K562_gwps | **1.144** | 0.985 | 0.914 |
| rpe1 | **1.109** | 0.971 | 0.899 |

**Three readings, in order of how much they matter.**

1. **The task is real and the model is not doing it.** A single cell's rest
   genes carry 8.6–10% reproducible variance, enough for a correlation of
   0.29–0.32. A ridge regression from the panel reaches about half of that. The
   gene-token model reaches 5–19%, so between 2.5× and 9× less than a linear
   map on the same inputs. PLAN_MAPPING §B2's reframe is **not** forced: the
   per-cell quantity exists.
2. **On two screens the model is worse than predicting library size alone.**
   0.016 against 0.077 on K562_gwps, 0.033 against 0.054 on K562_essential.
   One predictor beats 716.
3. **On two screens it is worse than predicting nothing.** *(Refuted by D105: the model column below was a raw error read against 1.0, not a ratio. On the same cells no change scores about the same.)* An mse ratio of
   1.144 and 1.109 against a no-change baseline of 1.0, on control cells,
   which are data it trains on. A zero-initialized model scores exactly 1.0,
   so training moved it backwards.

Taken together these say the mapping is not capacity-limited at the margin —
it is failing at something a ridge regression does not fail at. §B3 is the
branch, and reading 3 says the first thing to look at is not the latent width
but why the objective moves the mapping away from where it started.
`phase2/curves.csv` has `map_control` per step and answers that without a GPU.

**What it does not license.** The ridge is fitted on control cells and scored
on control cells. It says nothing about predicting *perturbed* cells, which is
the job, and nothing about whether a better mapping would move the six
official metrics — the rehearsal's controls-only upper bound sat at 1.00–1.03
on rest genes even when it was allowed the perturbed cells (D94). This is a
diagnosis of one step, not a route to a score.

### D103. The runs are deterministic, and the error bar is sd 0.0146

**Measured, not built.** Recorded a session late: the previous session ran
both measurements and could not commit them.

**A0 — determinism.** Two 1,000-step Phase 2 runs of one configuration
(`configs/server_isolate.yaml`, seed 2) agreed on **26 of 26** validation
metrics to the last bit, and the run header read "fused attention off". D99's
fix works: with only the math attention kernel enabled, one configuration and
one seed now give one answer.

**A1 — the standing error bar.** Three seeds of `configs/server_isolate.yaml`
(`warm_start: none`, `W` frozen, 6,000 steps), selection metric
`pert_mse_ratio_to_no_change` on the held-out targets:

| seed | best | at step | last | retained |
|---|---|---|---|---|
| 0 | 0.7802 | 6000 | 0.7802 | 1.00 |
| 1 | 0.8092 | 5250 | 0.8117 | 0.99 |
| 2 | 0.7915 | 6000 | 0.7915 | 1.00 |
| **mean** | **0.7936** | | | |

sd **0.0146**, range 0.0290. By PLAN_MAPPING §A2's table this is the first
row: **one run per configuration is enough**, and a difference under **0.03**
(2 sd) between two configurations is not a difference. Every later claim
states the gap against this figure. The seed moves the held-out target set as
well as the initialization and the cells, so 0.0146 is the spread of a
comparison between two *different* seeds; comparing two configurations at the
same seed can only be tighter, which makes 0.03 conservative.

**What it says about the instability.** All three deterministic runs end at
or beside their best (retained 0.99–1.00). The same configuration run
non-deterministically for 12,000 steps (D99) peaked at step 3,000 and had
given everything back by 12,000 (retained −0.12). That weakens D98 further:
its four-arm "every degree of freedom costs convergence" pattern was read off
runs of that non-deterministic kind.

**What it does not say.** Reordering floating-point sums does not make
training worse *in expectation*; it makes one run unrepresentative of the
configuration. So the runs above show that 6,000 steps of this configuration
do not collapse, three times out of three. They do not show what happens
between 6,000 and 12,000, which is where most of D99's decline happened
(0.9015 at 6,000, 1.0139 at 12,000). The direct test is one deterministic
12,000-step arm; with sd 0.0146 one run answers it.

**Alternative.** Five seeds instead of three. Not needed: at this spread the
effects the project is chasing (0.05 and up) are over three sd, and the
seed-to-seed range is already smaller than every gap D90–D92 rest on.

### D104. The panel→rest mapping does not learn, and the logged curve cannot show it

**Superseded in part by D105.** The paired curve was read against 1.0, but
on its own 32 cells no change scores about 1.07. The mapping never moved
backwards, and the two explanations below explain something that did not
happen. What stands: the logged value is 16 cells, the training curve cannot
show a ridge-sized gain, and `loss_mapping=0` alone leaves the delta term
training the decoder. D106 ran the separating arm anyway, and it answered a
different question.

**Measured.** `map_control` from `runs/server/phase2/curves.csv`, 6,000 steps
— the mapping's training loss on control cells, in control-SD units, where a
zero-initialized model (predict each gene's control mean) scores 1.0:

| step | 25 | 525 | 1025 | 1525 | 2025 | 2525 | 3025 | 3525 | 4025 | 4525 | 5025 | 5525 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| map | 0.974 | 0.922 | 0.940 | 1.027 | 0.957 | 1.041 | 0.933 | 1.050 | 1.114 | 0.962 | 1.090 | 0.948 |

No trend: it bounces around 1.0 and ends where it started, while the
perturbation module in the same network trains to 0.79 (D103). It is not
losing a weighting argument: `loss_mapping` and `loss_perturbation` are both
1.0 and both terms sit on a ~1.0 scale.

**The ratio that explains why nothing is visible.** From D102's ceiling, on
this loss a ridge regression gains 0.014 over no change and a perfect
predictor 0.086; the half-range of `map_control` between logged steps is
0.096. What a good linear map would achieve is about **a seventh** of the
step-to-step noise, and even a perfect predictor's gain is smaller than it.

**Two corrections to how that was first read.**

1. **The logged value is on 16 cells, not 32.** `make_step` splits
   `batch_size` 32 into 16 control and 16 perturbed cells, so `map_control` is
   16 cells × 2,048 sampled rest genes (`train.output_genes_per_step`).
2. **The bounce is not, by itself, evidence of not learning.** Each logged
   value is a *different* 16 cells, and most of its movement is common to
   every predictor — a batch of deep cells has a larger variance around the
   control mean whatever predicts it. A zero model would bounce the same way,
   and so would a mapping that had learned the ridge's 0.014. The curve cannot
   tell those apart. What does tell them apart is a *paired* measurement, and
   there are two: D102's held-out 1.144 and 1.109 on K562_gwps and rpe1, and
   the `val_map_control_mse` columns in the same `curves.csv`, which the loop
   writes at each of the eight evaluations on **the same** control cells every
   time (`evaluate` reseeds from `cfg.seed + 1`). Those, not the training
   curve, are the evidence that the mapping moves *backwards*.

**What it leaves open.** Why the mapping ends above 1.0 on data it trains on.
Two explanations fit, and they predict different arms:

* **It fits per-step noise.** The achievable signal is a seventh of the noise
  in each gradient estimate. Predicts that `phase2.batch_size=128` and
  `train.output_genes_per_step=0` (all rest genes) move it and
  `loss_mapping=0` does not help it.
* **The other objectives drag it.** The mapping's decoder, value head and
  latents are shared with the perturbation module, and the **delta term**
  (`loss_delta` 0.5) trains the same decoder on group-mean changes. Predicts
  that the mapping drifts above 1.0 even with `loss_mapping=0`.

`--set phase2.loss_mapping=0` alone does not separate these: the delta term
still trains the decoder. `loss_mapping=0` *and* `loss_delta=0` does — the
decoder then receives no gradient from any mapping objective, and wherever
`val_map_control_mse` ends is where the shared weights put it.

**Alternative.** Read the smoothed training curve instead of the paired one.
Smoothing twenty logged points removes about 4.5× of the noise, which still
leaves a 0.014 effect at the edge of visibility; the paired evaluation column
has no such problem and is already on disk.

### D105. The mapping was never worse than no change; the reference was wrong

**Measured.** `val_map_control_mse` at every evaluation, from the runs'
own `curves.csv`, 6,000 steps:

| run | step 750 | step 6000 | range across the eight evaluations |
|---|---|---|---|
| `server` (seed 2, non-deterministic) | 1.0751 | 1.0708 | 1.0694–1.0751 |
| `noise_seed0` | 0.9720 | 0.9694 | 0.9687–0.9813 |
| `noise_seed1` | 1.0008 | 0.9924 | 0.9913–1.0008 |
| `noise_seed2` | 1.0708 | 1.0659 | 1.0635–1.0708 |
| `b3_nomap` (seed 2, no mapping gradient, D106) | 1.0735 | 1.0719 | 1.0719–1.0735 |

**What it says.** The level is set by the seed, not by training. Each run
moves by under 0.01 across its whole curve, while seed 0's cells score 0.97
and seed 2's 1.07. And the arm whose mapping received **no gradient at all**
scores 1.072 on seed 2's cells, the same as the arms that trained it. So
1.07 is roughly what predicting no change scores on those 32 cells, not a
mapping that has moved backwards from 1.0.

The mechanism: `evaluate` draws `batch_size` (32) control cells per screen,
and in control-SD units no change is 0, so its error is those cells' own
mean square. That is 1.0 over the whole control population, which is what
`ctrl_std` is computed from. It is not 1.0 over 32 cells, because cells
share a depth factor across all 7,000 genes, and a deep or shallow draw
moves every gene together.

**What it refutes.**
* **D102 reading 3** ("on two screens it is worse than predicting nothing …
  training moved it backwards"). The 1.144 and 1.109 were raw errors read
  against a population value, and D102's "model mse ratio" column was not a
  ratio. The ridge's column was a ratio on its own 2,048 cells and stands.
  So do D102's pearson readings (1 and 2), since correlation does not
  depend on the baseline.
* **D104's "those, not the training curve, are the evidence that the
  mapping moves backwards."** The paired curve was the right instrument, but
  it was read against the wrong reference. Neither the paired nor the
  unpaired reading shows a mapping moving backwards.
* **D104's "fits noise" and "dragged by the shared objectives"** were both
  explanations of a drift above 1.0 that did not happen. Neither is needed.

**What the mapping actually does.** It learns, slowly. In the deterministic
12,000-step run (D107), `val_map_control_pearson` rises monotonically from
0.058 at step 1,500 to 0.082 at 6,000 and 0.1105 at 12,000, and
`val_map_control_mse` falls from 1.0673 to 1.0573 on cells where the
no-gradient arm scores 1.0719. That is 1.4% below the no-gradient arm,
which is about what the ridge gains (D102: 0.986). It is an approximate
reference, not an exact one: the no-gradient arm's decoder is still shared
with the perturbation module, though its pearson (−0.005) says it predicts
nothing. On pearson, 0.11 is about 37% of the ~0.30 ceiling, up from the
5–19% D102 measured on the non-deterministic run. Those two numbers come
from 32 and 2,048 cells respectively, so the share is indicative only.

**Decision.** Phase 2 now reports `map_control_no_change` and
`map_control_ratio_to_no_change`, and the same pair for perturbed cells,
computed on the cells each evaluation scores. The results summary's Phase 2
table shows the ratios instead of the raw errors. The ceiling diagnostic
passes the ratio through.

**Alternative.** Evaluate on more cells, so that the draw's no-change error
tends to 1.0. That helps, and it costs forward passes on every evaluation,
but it never removes the offset. The ratio on the same cells removes it at
any size, which is what `pert_mse_ratio_to_no_change` already does for the
perturbation module.

### D106. The mapping objectives cost the perturbation module 0.052

**Measured.** `configs/server_isolate.yaml`, seed 2, 6,000 steps, with
`phase2.loss_mapping=0 phase2.loss_delta=0`, against the same seed with
both on (`noise_seed2`, D103):

| at step 6000 | both on | both off | difference |
|---|---|---|---|
| `pert_mse_ratio_to_no_change` (held out, selects) | 0.7915 | **0.7397** | −0.052 |
| `pert_pearson` (held out) | 0.4526 | 0.4869 | +0.034 |
| `pert_train_mse_ratio_to_no_change` | 0.7447 | 0.7033 | −0.041 |
| `map_delta_ratio_to_no_change` (floor 0.4625) | 0.8300 | 1.0008 | +0.171 |
| `map_control_mse` (no-change ≈ 1.07 on these cells, D105) | 1.0659 | 1.0719 | — |

Both arms ended at their best (retained 1.00).

**Readings.**

1. **The mapping objectives cost the perturbation module 0.052.** That is
   3.5 sd of the seed-to-seed spread (D103), at the same seed, so it is a
   real difference by the project's standing rule. It shows on the training
   targets as well (−0.041), so the cost is not only in generalization: the
   shared network fits the perturbation objective less well when it also
   has to fit the mapping. Without them, the module reaches in 6,000 steps
   what it reaches with them in 12,000 (0.7435, D107).
2. **The delta term is the only thing teaching the group-level rest
   change.** With it off, `map_delta_ratio_to_no_change` sits at 1.00 for
   the whole run. With it on, it falls to 0.83 by 6,000 steps and 0.80 by
   12,000, against a floor of 0.46. That is the quantity §B2 would submit and
   the one the official metrics consume. Only 8 targets are scored, so any
   one value is noisy, but it falls in every run on every seed (seeds 0, 1
   and 2 end at 0.849, 0.879 and 0.830).
3. **Which of the two terms costs the perturbation module is not yet
   known.** This arm turned off both. The arm that separates them is
   `loss_mapping=0` alone, which keeps the delta term. If it keeps most of
   the 0.052 **and** keeps `map_delta` near 0.83, the per-cell level loss is
   paying for nothing the metrics score. Dropping it would be a §4.5
   deviation (the Siamese mapping would then be trained on differences
   only), recorded as one with these numbers.

Nothing changes in the shipped config on this arm alone. Turning off both
terms would give up the only rest-gene signal the network learns.

**A false alarm this arm exposed, and the fix.** The run ended with

```
core_unfrozen is unstable on pert_mse_ratio_to_no_change: best 0.7397 at step
6000, worst 1.0225 at step 750 ... swing 1.38x; 100% of what it learned was
still there at the end ... lower train.lr.
```

on a run that learned almost monotonically from its first evaluation to its
last. `instability` is `worst / best`, and a from-scratch arm's worst is its
start, at the no-change line. So every arm that learns more than 25% of the
way from 1.0 would be flagged, and told to lower a learning rate that D89
and D98 leave unsettled.

`TrainResult` now records `relapse`: the largest ratio of an evaluation to
the best seen **before** it. That is what "swings" means. On this arm it is
1.016 (0.8164 after 0.8034). On the arm D77 was written for, 2.4615 after a
best of 1.0804, it is 2.28 and still flagged. Phase 2's warning uses
`relapse`. `instability_worst_over_best` is still reported beside it, so
earlier readings stay comparable.

### D107. 12,000 deterministic steps: no collapse, and 0.048 better

**Measured.** `configs/server_isolate.yaml`, seed 2,
`--set phase2.steps=12000`:

| step | 1500 | 3000 | 4500 | 6000 | 7500 | 9000 | 10500 | 12000 |
|---|---|---|---|---|---|---|---|---|
| `pert_mse_ratio_to_no_change` | 0.8987 | 0.8597 | 0.8356 | 0.7915 | 0.7474 | 0.7445 | **0.7435** | 0.7439 |
| `pert_train_mse_ratio_to_no_change` | 0.8814 | 0.8326 | 0.7965 | 0.7447 | 0.7071 | 0.6954 | 0.6942 | 0.6900 |
| `map_control_pearson` | 0.058 | 0.079 | 0.081 | 0.082 | 0.086 | 0.095 | 0.104 | 0.1105 |
| `map_delta_ratio_to_no_change` | 0.894 | 0.859 | 0.826 | 0.830 | 0.820 | 0.801 | 0.798 | 0.798 |

Best 0.7435 at 10,500, last 0.7439, retained 1.00. Phase 2 took 72 minutes.

**Readings.**

1. **Determinism, a third time.** At every evaluation the two runs share
   (1,500, 3,000, 4,500 and 6,000), this run reproduces `noise_seed2`
   exactly, on both the perturbation and the mapping metrics. That holds
   although the two were given different `steps`, so nothing in training
   depends on the total budget.
2. **More steps help, then plateau.** 0.7915 → 0.7435 is 0.048, 3.3 sd
   (D103). Almost all of it arrives by 7,500 (0.7474). After that the
   held-out metric moves by 0.004 while the training-target metric keeps
   falling (0.7071 → 0.6900), which is where generalization stops paying.
3. **No collapse.** The same configuration and seed run non-
   deterministically for 12,000 steps peaked at 3,000 and ended at 1.0139
   (D99). Run deterministically it is flat from 7,500 to the end. The
   configuration does not collapse, and the non-deterministic run was one
   realization among many. That is all D103 said can be concluded from
   this: arithmetic reordering cannot make training worse on average, only
   make one run unrepresentative. D98's pattern, read off runs of that
   kind, is weaker again.
4. **The mapping is still improving at 12,000.** Pearson rises at every
   evaluation, and `map_delta` falls to 0.80 and holds there (D105, D106).

**Decision.** `phase2.steps` is 12,000 in `configs/server.yaml`. That is
72 minutes per arm, and about 2.4 hours of Phase 2 with the core-freeze
ablation, inside §0's budget. `configs/server_isolate.yaml` stays at 6,000,
so the next arms remain comparable with D103's error bar and with D106.

**Alternative.** 7,500 steps, which captures nearly all of the gain at 62%
of the cost. Rejected: the mapping and the delta term are still improving
between 7,500 and 12,000. The 2.4 hours fit the budget, and
`checkpoint_selection: best` keeps the best step whatever comes after it.

### D108. The per-cell level loss costs 0.019, not significant; it stays, because it anchors the level

**Measured.** `configs/server_isolate.yaml`, seed 2, 6,000 steps, three arms
that differ only in which mapping objectives are on:

| at step 6000 | both on (`noise_seed2`) | delta only (`loss_mapping=0`) | neither (`b3_nomap`) |
|---|---|---|---|
| `pert_mse_ratio_to_no_change` (held out) | 0.7915 | 0.7721 | 0.7397 |
| `pert_train_mse_ratio_to_no_change` | 0.7447 | 0.7302 | 0.7033 |
| `map_delta_ratio_to_no_change` (floor 0.46, 8 targets) | 0.830 | 0.846 | 1.001 |
| `map_control_pearson` | 0.082 | 0.063 | −0.005 |
| `map_control_mse` (no change on these cells: **1.0718**) | 1.0659 | 1.1004 | 1.0719 |

All three ended at their best step.

**Readings.**

1. **D105 is confirmed directly.** The new column measures no change on
   seed 2's 32 control cells at 1.0718. The arm whose mapping received no
   gradient scored 1.0719, so it was exactly no change. As ratios, the
   6,000-step mapping is at 0.9945 and the 12,000-step one (D107) at
   0.9865. That matches what the ridge gains on its own cells (0.986,
   D102).
2. **Neither term's cost is established alone.** Dropping the per-cell level
   loss gains 0.019 (1.3 sd, D103), which is not a difference. The remaining
   gap to "neither" is 0.032, just at 2 sd. Only the total cost of the two
   together, 0.052, is established. As an indicative split, about a third of
   the cost is the level loss and two thirds the delta term.
3. **The delta term alone teaches the group-level rest change.** Its
   `map_delta` (0.846) is indistinguishable from both-on (0.830) on 8
   targets. The level loss adds nothing measurable to the quantity the
   metrics score.
4. **Without the level loss, the level is unanchored.** The delta loss
   compares `mean(map(perturbed)) − mean(map(control))` with the observed
   change, so adding the same constant per gene to the mapping's output
   leaves it unchanged. With nothing else pinning the level, the last
   evaluation moved in one jump: `map_control` ratio 0.9959 → **1.0270**,
   `map_perturbed` 0.9886 → 1.0170, `pert_percell` 1.0031 → 1.0132. Over
   the same interval the delta and the pooled perturbation metrics improved.

**Why reading 4 decides it.** Phase 3's pooled prediction
(`predict/run.py`, `predicted_log2fc`) takes the mapping's **absolute**
output for the predicted perturbed panel and converts it to a fold change
against `ctrl_mean`. It does not subtract the mapping's output for the
control profile. A drifting offset would therefore become a predicted
change on every rest gene of every target: the same for all targets, so
worse on discrimination, and exactly what the DE metrics punish when too
many genes are called.

**Decision.** `phase2.loss_mapping` stays 1.0. The gain is inside the error
bar, and dropping the loss would buy it with an offset the prediction path
passes straight through. A §4.5 deviation is not justified.

**What it opens.** Training teaches the rest genes as a *difference*: the
delta term is what learns them, and it is offset-invariant. Prediction
reads them as an *absolute level*. Predicting the rest change as
`map(predicted panel) − map(control panel)` would read them the way they
were trained. The level loss would then only need to keep the per-cell
mapping sane, not carry the submission's offset. Not built. It is one
switch in `predicted_log2fc`, and the rehearsal's cross-context variant
is what measures it.

### D109. The rehearsal never scored the shipped model

**Measured.** The first full `all` at 12,000 Phase 2 steps (`server_12k`,
seed 2). Phase 2 improved as D107 predicted: held-out
`pert_mse_ratio_to_no_change` went 0.9173 → 0.8280 → 0.7938 → **0.7577**
at steps 3,000, 6,000, 9,000 and 12,000 (256 targets). The rehearsal's
end-to-end variant did not move:

| cross-context, leaderboard scale | method | floor | upper bound |
|---|---|---|---|
| K562_essential | −0.0470 | −0.0405 | +0.0511 |
| K562_gwps | −0.0134 | −0.0587 | +0.0586 |
| rpe1 | −0.2581 | −0.2628 | +0.0003 |
| **mean** | **−0.1062** | −0.1207 | +0.0367 |

The seed-2 run of PLAN_MAPPING §1 fact 5, at 6,000 non-deterministic steps,
had the method at −0.038, −0.030 and −0.253, a mean of −0.107. A Phase 2 gain
of 0.05 to 0.07 on its own metric moved the rehearsal by 0.001.

**Why: the rehearsal's method arms were not the shipped recipe.**

1. **Warm start.** Every method arm loaded Phase 1's core unconditionally.
   That is `warm_start: full`, which D90 measured not to learn, while the
   shipped Phase 2 has started from the priors since D91.
2. **Budget.** They trained Phase 2 for `rehearsal.phase2_steps`, a fixed
   300, while the shipped Phase 2 trains for 12,000. At 600 steps every arm
   D78 measured sat at no change. The log shows it: cross-context rebuilt its
   priors by 18:15:49 and had trained *and* adapted by 18:17:45.
3. **Variant 1 was not Phase 3's procedure.** Its method was Phase 1's core,
   which never saw Replogle, plus 150 adapter steps on one screen's
   controls. Its mapping started at 1.35–1.42 on control cells. Phase 3
   adapts a *trained* Phase 2. D94's policy ("adapting on controls makes the
   rest genes 1.5–3× worse, so do not adapt") was read off this arm.

So the method column has measured a model with no perturbation knowledge in
every run since D91. Four things rest on it and are **not evidence about
the shipped model**:
* D94's no-adapt policy, and PLAN_MAPPING §1 fact 3;
* the generator calibration (threshold 0, scale 1 in this run);
* sanity checks 6 and 7, which read the rehearsal;
* "the method sits at the no-change floor" (fact 5), as a statement about
  the model rather than the rehearsal.

The two leaderboard uploads (−0.1313) *were* the shipped model, so the
leaderboard stands as the one end-to-end reading. It is also affected by
D110.

**Decision.**
* Method arms start as `phase2.warm_start` says (`start_like_phase2`,
  sharing Phase 2's `reset_delta` and `keep_only_perturbation`) and train
  with `phase2.train_projections`.
* `rehearsal.phase2_steps` defaults to 0, meaning `phase2.steps`.
  `configs/server.yaml` sets 7,500: a 12,000-step run was at 0.7474 there
  against its best of 0.7435, within the error bar (D107), at 62% of the
  cost. That is four arms of about 45 minutes (three held-out screens, plus
  unseen genes). The rehearsal arms do not evaluate while training, so they
  keep their last step. At 7,500 the D107 curve was flat, so last ≈ best.
* Variant 1's method **is** variant 2's held-out model: Phase 2 trained
  without the screen, then adapted on its controls. It is trained once per
  screen and shared (`HeldOutModel`). Variant 1 also scores the same model
  **before** adapting (`method_unadapted`, rest genes only). The policy now
  compares adapted with unadapted, because that is Phase 3's actual choice.
  The floor comparison stays only as a fallback for a report without the
  unadapted arm. `POLICY_VERSION` is 3, so every policy written from the old
  rehearsal is refused.

**Alternative.** Score the shipped Phase 2 checkpoint directly. Rejected:
it trained on the held-out screen, which is what the upper bound is for.
The method has to be the shipped *recipe* applied without that screen, and
§4.1's leakage rule requires the priors rebuilt without it too.

### D110. Prediction switched off Phase 2's adapter

**Measured since (D111): it cost nothing.** On `server_12k`'s checkpoint the
adapter moves `pert_mse_ratio_to_no_change` by 0.0003 (0.7577 on, 0.7581
off). The fix stays, because it makes the predicting network the validated
one, but it explains none of the score.

**Found, not yet measured on the server.** Phase 2 trains its core and its
LoRA adapter (`phase2`) together, and validates them together: every Phase 2
number in this file is with the adapter on. Phase 3's adaptation
(`adapt_on_controls`), the prediction (`predict/run.py`) and every rehearsal
arm then called `use_adapters([context_adapter(name)])`. That activates the
context adapter *alone*. The Phase 2 adapter was loaded from the checkpoint
and never used. So every submission, including both uploads, and every
rehearsal upper bound ran a network that nothing had validated: the core
without the low-rank weights trained alongside it. No decision chose this.
The adapters were designed per phase and per context (§4.3), and nothing
stacked them.

**Size: unknown until measured.** `scripts/adapter_check.py` scores the
saved Phase 2 checkpoint on Phase 2's own validation with the adapter on
and off, on the same cells and targets. On mini the gap is 0.0001 (0.9717
against 0.9716), because a 150-step adapter has barely moved, and the "on"
reading reproduces the run's recorded best exactly. On the server the
adapter has trained for 12,000 steps next to an unfrozen core. The gap
could be anything from nothing to the whole of what Phase 2 learned, and
it is a few GPU-minutes to find out.

**Decision.** `adapt.inference_adapters(context, cfg)` returns
`[phase2, context:<name>]`, and all four callers use it. The context adapter
trains on top of a frozen, active Phase 2 adapter.
`phase3.stack_phase2_adapter: false` restores the old behaviour for the A/B.

**Alternative.** Merge the Phase 2 adapter into the base weights at the end
of Phase 2. Equivalent in function, but it changes what the checkpoint
holds, and the frozen-parameter check would need to know about it. Stacking
is one list.

### D111. Switching the Phase 2 adapter off cost 0.0003

**Measured.** `scripts/adapter_check.py` on `server_12k`'s Phase 2
checkpoint, same cells and targets, adapter on / off:

| | on | off |
|---|---|---|
| `pert_mse_ratio_to_no_change` | 0.7577 | 0.7581 |
| `pert_train_mse_ratio_to_no_change` | 0.7622 | 0.7634 |
| `map_control_ratio_to_no_change` | 0.9861 | 0.9863 |
| `map_perturbed_ratio_to_no_change` | 0.9742 | 0.9748 |
| `map_delta_ratio_to_no_change` | 0.7893 | 0.7944 |

The "on" column reproduces the run's recorded validation exactly. With the
core unfrozen, the core carries everything Phase 2 learns, and the rank-limited
adapter trained beside it carries almost nothing. D110's bug was real and is
fixed. It cost the earlier submissions nothing measurable, and it is not why
they scored where they did.

### D112. The rehearsal of the shipped recipe: +0.056 on the leaderboard scale, and all of it within K562

**Measured.** `server_fix`, seed 2, the first `all` with D109 and D110
(10.6 hours end to end; rehearsal 5.9 h). The cross-context variant, now a
Phase 2 trained for 7,500 steps without the held-out screen, default
generator settings:

| leaderboard scale | method | floor | upper bound | before D109 |
|---|---|---|---|---|
| K562_essential | **+0.0352** | −0.0405 | +0.0548 | −0.0470 |
| K562_gwps | **+0.0727** | −0.0587 | +0.0597 | −0.0134 |
| rpe1 | −0.2583 | −0.2628 | −0.0203 | −0.2581 |
| **mean** | **−0.0501** | −0.1207 | +0.0314 | −0.1062 |

**Readings.**

1. **The model does something once the rehearsal measures it.** On both K562
   screens the method is above the organizers' mean-response baseline (0 on
   this scale). On K562_essential it captures 79% of the distance from the
   floor to the upper bound. On K562_gwps it exceeds the upper bound. Before
   D109 the same column was at the floor.
2. **But that is transfer within one cell line.** A held-out K562 screen is
   predicted from a model that trained on the *other* K562 screen. rpe1 is
   predicted from K562 alone, which is a new cell type, and there the method
   is at the floor (+0.005 above it) while the upper bound is 0.24 above it.
   The challenge contexts are new cell types too, so **rpe1 is the variant
   that predicts the leaderboard.** It says perturbation knowledge learned in
   K562 does not yet reach a different cell line through the control-only
   adaptation.
3. **The failure mode at default settings is too much change.** On the K562
   screens DE fidelity rises from 0 to 0.46–0.57, and nmae falls below the
   floor. But the expression metric gets much worse (1.85 against 1.13 on
   K562_essential, 8.56 against 1.22 on K562_gwps), Jaccard drops (0.016
   against 0.150), and sanity check 7 counts 8.1× as many significant genes
   as the reference. Calibration answered with the most conservative corner
   of its grid: threshold 0.25, scale 0.25, objective +0.080 against −0.016
   at the defaults. That is fitted and scored on K562_essential alone, so it
   is in-sample. The corner also means the grid may be too narrow.
4. **Adapting on controls is neutral, not harmful.** Variant 1, same model
   before and after: 0.986 → 0.985 (K562_essential), 1.322 → 1.323
   (K562_gwps), 0.926 → 0.919 (rpe1). The policy now permits adapting, and
   Phase 3 adapted. The challenge contexts moved by about 0.001 (A 0.7701 →
   0.7691). D94's "adapting makes it 1.5–3× worse" was the untrained arm.
5. **The predicted targets are no longer one profile.** Sanity's median
   between-target correlation of predicted changes fell from 0.895, 0.880
   and 0.916 (`server_12k`) to 0.032, 0.036 and 0.029. The adapter made no
   difference (D111) and adapting moved almost nothing. What changed is the
   calibration: a 0.25 threshold zeroes small changes. D108 suspected a
   shared offset in the mapping's absolute output, and this is consistent
   with the threshold removing it. It is consistent, not a proof.

**Open, and flagged rather than resolved.** The "upper bound" is not always
above the method: 1.50 against 0.985 in variant 1 on K562_essential, and
below the method in cross-context K562_gwps. It is the main Phase 2
checkpoint, which saw the held-out screen. Why a model that saw the screen
maps its rest genes worse than one that did not is unexplained. Until it is,
the upper bound is a reference point, not a ceiling.

**What it does not change.** `server_fix/submission/prediction.vcc` passed
our validator and `vcc prep --dry-run`, and it is the first submission whose
rehearsal measured the model that made it. Only the leaderboard can say
whether reading 2 holds for the challenge's cell lines.

### D113. The calibration chose "predict almost nothing", and the leaderboard punished it

**Measured.** Three uploads, leaderboard scale (0 = the organizers'
mean-response baseline):

| upload | run | generator | overall | pds | mse | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|---|---|
| v4 | `server` | as that run calibrated | **−0.014** | −0.006 | 0 | −0.037 | −0.015 | −0.002 | −0.027 |
| v5 | `server_12k` | threshold 0, scale 1 | −0.083 | −0.004 | 0 | **−0.504** | +0.009 | +0.004 | −0.001 |
| v6 | `server_fix` | threshold 0.25, scale 0.25 | −0.242 | −0.016 | 0 | +0.002 | **−1.340** | −0.019 | −0.078 |

The rehearsal ranked them in the opposite order. v6's loss is almost all DE
direction fidelity: −1.34 on a scale where the baseline is 0.

**Why.** Two faults in how the calibration was fitted, both older than this
cycle:

1. **It maximized the wrong objective.** The objective maps each metric so
   that *predicting no change* scores 0 (D5). The leaderboard maps it so that
   the *mean-response baseline* scores 0, and that baseline calls genes, in
   mostly the right direction. So a prediction that moves almost nothing
   costs nothing on the calibration's objective and a great deal on the
   leaderboard. The grid shows it: the chosen point had fidelity **0.0** raw
   and an objective of +0.080, the best in the grid. Threshold 0 at scale 1.0
   had fidelity 0.30 and reach 0.22, and scored lower (+0.055). The mode
   sweep's docstring had already recorded that the two scales disagree. The
   calibration was never moved onto the one the leaderboard uses, although
   the rehearsal builds that scale for every cross-context screen.
2. **It was fitted on five targets of one screen.** It re-scored the stored
   per-target predictions (`n_saved_predictions: 5`), on K562_essential only
   (the screen with the most targets). The pds values in the grid (0.75,
   0.70, 0.65, 0.55) are multiples of 0.05 because they are means of five
   ranks. Sixteen settings compared on five targets is noise.

**Decision.** The cross-context variant runs the grid itself, on **every**
target it scored (100 per screen), with the screen's leaderboard scale.
`calibrate.combine` picks the one setting with the best **mean over all
cross-context screens** on the leaderboard scale. It falls back to the
no-change objective only where no screen could be placed on that scale. A
screen whose scorer failed everywhere is left out rather than vetoing every
point. The per-screen optimum and the per-screen value of every point are
kept in the report (`per_dataset.others`, `combined`), so the spread is
visible. `rehearsal.calibrate_per_dataset` is no longer read.

**Cost.** Sixteen scorings of 100 targets on each of three screens, about
2.3 minutes each: roughly 1.8 hours added to the rehearsal.

**What it does not fix.** Reading 2 of D112 stands. The model transfers
within K562 and not yet to rpe1, and the challenge contexts are new cell
lines. Averaging rpe1 into the calibration means the setting chosen is one
that does not lose on the screen most like the challenge. It does not make
rpe1's predictions better.

**Unexplained, recorded.** v4 (`runs/server`) scored −0.014 on this upload.
The earlier uploads recorded at −0.1313 came from the same directory. Either
that run was redone in between, or the leaderboard's anchors changed (the
upload reports anchor set `...-r4`). The two cannot be compared until we know
which.

### D114. The leaderboard-scale grid: effect scale is what matters, and one rehearsal run is noisy

**Measured.** `server_cal`: `server_fix`'s Phase 2 with the rehearsal and
everything after it rerun under D113. The grid, as the mean leaderboard
score over the three cross-context screens (100 targets each):

| threshold \ scale | 0.25 | 0.5 | 1.0 | 1.5 |
|---|---|---|---|---|
| 0.0 | −0.087 | −0.047 | −0.011 | −0.014 |
| 0.05 | −0.091 | −0.047 | −0.011 | **−0.010** |
| 0.1 | −0.090 | −0.050 | −0.013 | −0.014 |
| 0.25 | −0.091 | −0.053 | −0.020 | −0.026 |

Chosen: threshold 0.05, scale 1.5.

**Readings.**

1. **Scale decides, threshold barely matters.** Every row climbs from about
   −0.09 at scale 0.25 to a plateau of −0.010 to −0.014 at 1.0–1.5. Across
   thresholds 0 to 0.1 the spread within a column is under 0.004. The
   chosen point and threshold 0 / scale 1.0 differ by 0.0015, which is no
   difference. D113's no-change objective picked 0.25 / 0.25, which is
   −0.091 here, in the worst row of the worst column. That agrees with what
   the leaderboard did to v6.
2. **The screens want different scales.** K562_essential rises to 1.5
   (+0.053). K562_gwps peaks at 0.5 (+0.044) and is negative by 1.5. rpe1
   improves all the way to 1.5 (−0.241 → −0.080). The mean hides a real
   disagreement, and the top of the grid is at its edge again for two of
   three screens.
3. **rpe1 at threshold 0 / scale 1.0 is −0.093. Upload v5, with the same
   settings on a different model, scored −0.083.** One point, but it is the
   first place where a rehearsal number and a leaderboard number line up. It
   is consistent with D112's reading that the screen from a new cell type is
   the one that predicts the challenge, and it says the K562 screens (+0.024,
   +0.036 at those settings) are optimistic by about 0.1.
4. **The same recipe and seed gave very different rehearsals.** Method at
   default settings, `server_fix` → `server_cal`: K562_essential +0.035 →
   +0.023, K562_gwps +0.073 → +0.033, **rpe1 −0.258 → −0.090**. On rpe1
   the direction-fidelity score went from 0.016 to 0.492. The Phase 2
   checkpoint was the same file. The held-out models were retrained, and
   their priors rebuilt.

**Why reading 4: the priors were not deterministic.** `sparse_svd`, which
builds the HGNC gene-group block, called `scipy.sparse.linalg.svds` with no
starting vector. ARPACK then draws one at random, and a multi-hot membership
matrix has many tied singular values. The vectors it returns depend on the
random state, and not only by sign. On a synthetic membership matrix, three
global seeds gave outputs differing by up to 2.75. So every fresh `priors`
stage, and every rehearsal variant that rebuilds its scoped priors, could
start from a different W. It also explains why Phase 1 differed at step 100
between `server_12k` and `server_fix` (`sig_mse` 0.4592 against 0.4594), where
D103's bit-exact runs had shared one copied `priors/` directory.

**Consequences.**
* D103's error bar holds for Phase 2 runs sharing one prior. It is **not** the
  spread of a full `all` run or of a rehearsal, which also carry the prior
  draw. For the rehearsal that spread has now been seen at 0.17 on rpe1.
* D112's "rpe1 is at the floor" was one draw. This draw is 0.17 above the
  floor. Neither is the answer by itself.
* Comparisons *within* one rehearsal, like this grid (same predictions, only
  the generator setting changed), are paired and unaffected.

**Decision.** `sparse_svd` passes a fixed `v0` (seed 0) and fixes each
column's sign so its largest entry is positive. A test checks that three
global random states give identical output. This changes the HGNC block from
one arbitrary draw to one fixed one. Every run from here builds the same
priors from the same data.

**What it leaves.** The generator setting is now chosen on the right scale,
but the mean it maximizes sits at −0.010 on the rehearsal and, by reading 3,
nearer −0.07 on the leaderboard. Upload v4 (−0.014) is still the best, and
why is not known. That run's generator settings are the first thing to look
up.

### D115. Three uploads, three models, three settings: the next upload separates them

**Measured.** The generator settings each uploaded run used, read from its
`phase3_policy.json`:

| upload | run | Phase 2 | threshold | scale | leaderboard |
|---|---|---|---|---|---|
| v4 | `server` | 6,000 steps, non-deterministic | 0.1 | **0.5** | **−0.014** |
| v5 | `server_12k` | 12,000 steps | 0.0 | 1.0 | −0.083 |
| v6 | `server_fix` | 12,000 steps | 0.25 | 0.25 | −0.242 |
| — | `server_cal` | as `server_fix` | 0.05 | 1.5 | not uploaded |

**Readings.**

1. **Every pair differs in both model and setting**, so the leaderboard alone
   cannot say whether v4 leads because of its scale or its model.
2. **The leaderboard's per-metric losses fit a middle scale.** v5 at scale
   1.0 lost 0.50 on DE log-FC accuracy (errors on the reference's significant
   genes grow with the size of the predicted change). v6 at 0.25, with a 0.25
   threshold, lost 1.34 on direction fidelity (too few calls). v4 at 0.5 lost
   little on either.
3. **The rehearsal's rpe1 column matched both 12,000-step uploads.** At v5's
   settings rpe1 is −0.093 (leaderboard −0.083), and at v6's it is −0.243
   (leaderboard −0.242). At v4's settings, on the current model, it is −0.163,
   far below v4's −0.014, but v4 is a different model. Either rpe1 stops
   predicting the leaderboard at scale 0.5, or the 6,000-step model is the
   better one for a new cell line. D107 is consistent with the second: after
   7,500 steps, the training-target metric kept falling while the held-out
   one stayed flat.

**Decision.** `predict.override_threshold` and `predict.override_scale`
(negative = use the policy) let the `predict` stage use settings other than
the policy's. The run logs a warning and records the settings actually used,
so the current model can be uploaded at v4's settings without rerunning the
rehearsal. That upload, next to v4, separates model from setting. `server_cal`
at its own 1.5, next to it, gives the same model at two scales.

**Alternative.** Rerun the 6,000-step model through the new rehearsal. That
answers the same question on the rehearsal rather than the leaderboard, at
about ten hours instead of thirty minutes, and the leaderboard is the
instrument in question.
