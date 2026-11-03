# Final round: checklist and commands

Rewritten 2026-11-03 for the recipe the validation leaderboard selected
(DECISIONS.md D125–D136). The earlier version described the pre-redesign
pipeline (a full `all`, rehearsal-calibrated shared 2 / specific 1); it no
longer applies.

## What ships: the L8 recipe, unchanged

| part | setting | evidence |
|---|---|---|
| model | `runs/server_shared`'s priors, Phase 2 checkpoint and policy, **not retrained** | the model every probe used (D118, D125) |
| Phase 3 | rerun on the round's contexts, per that policy | `predict` needs an adapter for every context of the round |
| model's own target-specific part | off (effect scale 0.01, threshold 0.05) | P1 +0.0094 (D125) |
| model's shared part | scale **0.5** | the bracket 2.0 / 1.5 / 1.0 / 0.5 / 0.25 / 0: +0.0098 / +0.0189 / +0.0286 / **+0.0369** / +0.0296 / +0.0282 (D134, D136) |
| K562 lookup | raw, scale 1.0, rebuilt from the round's processed data | L4 +0.0322 over no lookup; denoisers lost (D126, D135) |
| knockdown prior | as in every run | §4.7 |

Validation score of exactly this recipe: **+0.0369** (L8). Targets the K562
screen does not measure get the model's shared part alone, as the 28
uncovered validation targets did. Seed 2, as every probe.

The whole path after the data rebuild was drilled on `mini_data` relabelled
as a new round (contexts D, E, F): `phase3` adapted the new contexts,
`predict` rebuilt the lookup from the round's data, `validate` passed with
exactly D, E and F (D136).

---

## 1. The release: one clean directory

Both `data_prep` scripts process every `context_*.h5ad` in the release
directory. Leftover validation files beside the final ones would become
contexts of this round, so unpack the final release into a directory of its
own:

```bash
cd /data/share/han/VCC
unzip -l vcc_2026_controls_final.zip            # what the release contains
export VCC_FINAL=/data/share/han/VCC_final
mkdir -p $VCC_FINAL && unzip -n -q vcc_2026_controls_final.zip -d $VCC_FINAL
ls -la $VCC_FINAL                               # files at the top level?
```

If the zip unpacked into a subfolder, point `VCC_FINAL` at that subfolder. If
it holds only the `context_*.h5ad` files, copy `manifest.json`,
`gene_names.csv` and `pert_counts.csv` from the final release's download into
`$VCC_FINAL` (not the validation ones). Then:

```bash
cd /data/han/projects/VCC
python scripts/final_release_check.py $VCC_FINAL --validation-data /data/han/projects/VCC/data
```

- [ ] It ends `OK one clean release; the fast path applies`.
- [ ] `partition` is the test partition, the contexts are the round's, and
      `targets` / `cells_per_pert` match the challenge's announcement.
- [ ] Note the targets line: how many were validation-round targets.
- [ ] If it says the gene axis is **DIFFERENT**, the validation model cannot
      be reused: steps 2–4 as written, then the fallback in step 5.

## 2. Rebuild the processed data for the round

Every processed file speaks the round's gene axis and flags the round's
targets; the harmonization maps Replogle's target labels with the round's
target list, which the lookup reads. Rebuild into a **new** `data_root`,
leaving the validation data intact.

```bash
source scripts/server_env.sh
cd /data/han/projects/VCC
git pull origin claude/blissful-volta-lmijw9
export DATA_FINAL=/data/han/projects/VCC/data_final
du -sh /data/han/projects/VCC/data/processed; df -h /data/han/projects/VCC   # room for a second copy?

mkdir -p $DATA_FINAL/ref
cp /data/han/projects/VCC/data/ref/hgnc_complete_set.txt $DATA_FINAL/ref/
REF="--ref $DATA_FINAL/ref/challenge_genes.tsv --perts $DATA_FINAL/ref/challenge_perts.txt --hgnc $DATA_FINAL/ref/hgnc_complete_set.txt"

# a. reference files from the release: gene order, the perturbation list
nice -n 10 python data_prep/challenge_prep.py --vcc $VCC_FINAL --out $DATA_FINAL/ref \
  --hgnc $DATA_FINAL/ref/hgnc_complete_set.txt 2>&1 | tee $DATA_FINAL/prep_challenge.log

# b. Replogle, LINCS and GMT harmonized to that vocabulary (the long step)
nice -n 10 ionice -c3 python data_prep/harmonize_data.py all $REF \
  --out $DATA_FINAL/processed 2>&1 | tee $DATA_FINAL/prep_harmonize.log

# c. the three phases' inputs, including phase3/controls_<ctx>.h5ad and targets.csv
nice -n 10 ionice -c3 python data_prep/phase_data.py all $REF --vcc $VCC_FINAL \
  --processed $DATA_FINAL/processed --out $DATA_FINAL/processed/phases \
  2>&1 | tee $DATA_FINAL/prep_phases.log
```

- [ ] Each log ends without a traceback.
- [ ] `prep_harmonize.log` prints "challenge perturbations present: n/N" per
      source. Note K562 genome-wide's: it is the lookup's coverage of the
      round (validation: 272 of 300).

## 3. Point the config at the round

`configs/server_final.yaml` already reads `vcc_root: /data/share/han/VCC_final`
and `data_root: /data/han/projects/VCC/data_final`. If you used other paths,
edit those two lines only (a test fails if anything else differs from
`configs/server.yaml`) and commit the edit.

## 4. Check before running

```bash
python scripts/check_env.py --config configs/server_final.yaml
python -m vccp check-data --config configs/server_final.yaml --seed 2 --run-name final_check
```

- [ ] `check_env`: `vcc` and `cell-eval2` found, GPU visible, thread
      variables set.
- [ ] `check-data`: **0 fail**, and the contexts, target count and gene count
      it prints are the release's.

## 5. Run

**Fast path** (the gene axis is the validation model's). About 1.5–3 h:
`phase3` minutes, `predict` up to an hour (it reads the K562 cells of the
round's covered targets), `validate` and `package` each up to an hour
(`vcc prep` reads the whole file; D133).

```bash
CONFIG=configs/server_final.yaml STAGES="phase3 predict sanity validate package" \
LOOKUP_CACHE=runs/lookup_cache_final \
  scripts/probe.sh final_l8 2 0.05 0.01 0.5 \
    --set predict.lookup_scale=1.0 --set predict.lookup_shrinkage=none
```

It copies `runs/server_shared` into `runs/final_l8` (without its predictions,
submission, sanity reports or per-context Phase 3 records), adapts the
round's contexts, and stops at the first failing stage. Its own lookup cache
leaves the validation round's `runs/lookup_cache` untouched.

**Fallback** (the gene axis changed): train on the round's data, then apply
the same settings.

```bash
nice -n 10 ionice -c3 python -m vccp all --config configs/server_final.yaml \
  --seed 2 --allow-warnings 2>&1 | tee runs/final.log          # about 12 h
for stage in predict sanity validate package; do
  python -m vccp $stage --config configs/server_final.yaml --seed 2 --force --allow-warnings \
    --set predict.override_threshold=0.05 --set predict.override_scale=0.01 \
    --set predict.override_shared_scale=0.5 \
    --set predict.lookup_scale=1.0 --set predict.lookup_shrinkage=none || break
done
```

## 6. Read before submitting

All under `runs/final_l8/` (fast path) or `runs/final/` (fallback):

| where | what | reference | stop if |
|---|---|---|---|
| `probe.log` | every stage ran; last lines list `submission/` | — | a traceback |
| `phase3/adapt_<ctx>.json` | one per round context; `loss_after` ≤ `loss_before` | the validation run's | a context missing |
| `predictions/model/index.json` → `generator` | threshold 0.05, effect 0.01, shared 0.5 | L8 | anything else |
| same → `lookup` | scale 1.0, shrinkage none, `n_covered` / `n_uncovered` | 272 / 28 of 300 | `null` (lookup off) |
| `sanity/summary.txt` | checks 1–5 | PASS | FAIL in 1, 3, 4 or 5 |
| same | checks 6, 7 | WARN in every run (they read the K562 rehearsal) | — |
| `reports/validate.json` | `ok`, and `requires.contexts` = the manifest's | true | false, or the wrong contexts |
| `submission/package.json` | `ok`, `seconds` | true | false (package by hand, §8) |

The predictions should look like L8's:

```bash
python scripts/compare_predictions.py runs/probe_l8 runs/final_l8
```

New contexts move the numbers; a `target_corr` near 1.0 would mean every
target got the same prediction.

## 7. Submit

```bash
ls -l runs/final_l8/submission/      # the .vcc must be newer than prediction.h5ad
vcc submit runs/final_l8/submission/prediction.vcc \
  -m "final T1: L8 recipe (shared 0.5, raw K562 lookup 1.0), seed 2" --wait
```

- [ ] The result reads the **test** partition and the round's panel, not
      `val`.

## 8. If something goes wrong

* **A stage fails.** Fix the cause and rerun the same probe command: it
  copies the parent again and reruns every stage, deterministically.
* **`vcc prep` stops at its limit** (`submission.vcc_timeout_minutes`, 240):
  the log prints the exact command; run it by hand after deleting any
  `prediction.vcc` beside the submission.
* **Out of GPU memory in `phase3`.** The error names the key to lower.

## 9. After the first upload

The first test upload is the validation-best recipe. After it, only changes
the validation round supported, one at a time, each paired against T1 with
the same seed. The nearest alternatives were all worse on validation: shared
1.0 (L7, −0.0083), shared 0.25 (L9, −0.0073), lookup 1.5 (L5, at shared 2.0).
Seed noise is about 0.002 overall (P0); the leaderboard of a new round may
rescale that.

## 10. Record it

A `DECISIONS.md` entry: the release check's output, the coverage line of
`prep_harmonize.log`, check-data's sizes, the lookup's coverage, the sanity
summary, `validate.json`, and the test leaderboard's result.
