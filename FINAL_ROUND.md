# Final round: checklist and commands

The final test data arrives **2026-10-22**: new contexts (D, E, F) and a
**different** perturbation list. The method doesn't change, only the data.
This file is the whole procedure, in order, with what to check at each step
and what each check should read. Tick the boxes as you go.

**What ships** is the pipeline as measured through the validation round
(DECISIONS.md D99–D118):

* Phase 2 from the priors (`warm_start: none`, D91): 12,000 deterministic
  steps (D103, D107).
* The rehearsal trains the shipped recipe without each screen (D109). It
  calibrates the generator on **rpe1**, the held-out screen from a new cell
  line, on the leaderboard's scale. Its score tracked six uploads within
  about 0.02 (D113, D116, D118).
* The generator scales the part every target shares separately from each
  target's own part (D117). This is a recorded §4.7 deviation, worth 0.032
  on the validation leaderboard (D118).

Validation-round reference values are quoted below. A final-round number far
from its reference is a reason to stop and look before uploading.

Budget: data prep a few hours (mostly copying the harmonized Replogle files),
then about 12 hours for `all`. **Start the morning the data lands.** Check the
challenge site for the final-round deadline and allow two days.

---

## 0. Before 2026-10-22

- [ ] **Validation round:** keep v4 (−0.014) as the entry that counts, if the
      leaderboard lets you choose (D118).
- [ ] **Disk.** The rebuild writes a second `processed/` tree. The new copy
      needs at least as much free space as the old one uses:
  ```bash
  du -sh /data/han/projects/VCC/data/processed
  df -h /data/han/projects/VCC
  ```
- [ ] **Optional fire drill (a few hours, no GPU):** run step 2 against the
      **validation** release into a scratch root and confirm it reproduces
      today's data. Then the commands are proven before they matter:
  ```bash
  export VCC_FINAL=/data/share/han/VCC                 # the validation release, for the drill
  export DATA_FINAL=/data/han/projects/VCC/data_drill
  # ... run step 2 exactly as written below, then:
  for f in processed/phases/gene_split.csv processed/phases/phase3/targets.csv \
           processed/phases/phase3/genes.csv ref/challenge_perts.txt; do
    cmp -s /data/han/projects/VCC/data/$f $DATA_FINAL/$f && echo "same  $f" || echo "DIFF  $f"
  done
  python -m vccp check-data --config configs/server.yaml \
    --set data_root=$DATA_FINAL --run-name drill
  rm -rf $DATA_FINAL runs/drill                        # once it reads "same" and passes
  ```

## 1. The release

- [ ] Unpack it, then check it has what the pipeline reads:
  ```bash
  export VCC_FINAL=/data/share/han/VCC_final            # wherever you unpacked it
  export DATA_FINAL=/data/han/projects/VCC/data_final
  ls $VCC_FINAL     # manifest.json  gene_names.csv  pert_counts.csv  context_*.h5ad
  python -c "import json; m=json.load(open('$VCC_FINAL/manifest.json')); print({k: m[k] for k in m if k != 'per_context'}); print(sorted(m['per_context']))"
  ```
  Expect contexts D, E, F (whatever they are named, the code reads them) and
  `cells_per_pert` from the manifest.
- [ ] **Is the gene axis the same as the validation round's?**
  ```bash
  cmp -s /data/share/han/VCC/gene_names.csv $VCC_FINAL/gene_names.csv \
    && echo "same 18,533 genes" || echo "GENE AXIS CHANGED"
  ```
  If it changed, nothing in the code needs editing (the axis is read from
  the file), but the rebuild in step 2 is then required, not just prudent.

## 2. Rebuild the processed data for the round

Every processed file speaks the round's gene axis and flags the round's
targets (`is_challenge_pert`, `targets.csv`, the coverage tables). The
harmonization also uses the target list to map Replogle's target labels.
So rebuild everything into a **new** `data_root`, which leaves the
validation-round data intact. Run from the repository checkout:

```bash
source scripts/server_env.sh
cd /data/han/projects/VCC
git checkout main && git pull origin main

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
      source. Note the numbers: this round's coverage of the new target
      list by Replogle and LINCS. In the validation round, K562 genome-wide
      held 272 of 300.

## 3. Point the config at the round

- [ ] Edit the two paths at the top of `configs/server_final.yaml` to
      `$VCC_FINAL` and `$DATA_FINAL`. Change nothing else. A test fails if
      that file differs from `configs/server.yaml` in anything but
      `data_root`, `vcc_root` and `run_name`.
- [ ] Commit that edit, so the run's `config.yaml` and the repository agree.

## 4. Check before training

```bash
python scripts/check_env.py --config configs/server_final.yaml
python -m vccp check-data --config configs/server_final.yaml --seed 2
```

- [ ] `check_env`: `vcc` and `cell-eval2` found, GPU visible, thread
      variables set.
- [ ] `check-data`: **0 fail**. It verifies `ref/challenge_perts.txt` matches
      `pert_counts.csv` and every gene order agrees.
- [ ] The sizes it prints are what the release says: the new contexts, the new
      target count × `cells_per_pert`, and 18,533 genes (or the new axis).

## 5. Run

```bash
nice -n 10 ionice -c3 python -m vccp all --config configs/server_final.yaml \
  --seed 2 --allow-warnings 2>&1 | tee runs/final.log
```

Seed 2 is the seed every validation-round measurement used. The run is
deterministic (D99, D114), so the same data gives the same submission.
About 12 hours: Phase 2 ≈ 2.4 h, rehearsal ≈ 7 h, the rest ≈ 1.5 h.

**If time is short**, these cut it without changing the method's choices:

| saves | how |
|---|---|
| ~1 h | `--set train.core_freeze_ablation=false` (an ablation arm only) |
| ~1.5 h | `--set rehearsal.phase2_steps=4500` (the rehearsal's own arms) |

## 6. Read before submitting

| where | what | validation-round reference | stop if |
|---|---|---|---|
| `runs/final/phase2/metrics.json` | held-out `pert_mse_ratio_to_no_change` | 0.742–0.758 | above 0.85: the data rebuild is suspect |
| `runs/final/rehearsal/summary.txt` | `objective : leaderboard (over rpe1)` | the same line | it says `no_change`: the scale could not be built |
| same | chosen `effect_scale` / `shared_scale` | 1.0 / 2.0 | — (the grid chooses; record it) |
| `rehearsal/report.json` → `calibration.combined` | rpe1 score at the chosen setting | −0.040 | below −0.10 |
| `runs/final/sanity/summary.txt` | check 1 | PASS | anything but PASS (the run stops itself) |
| same | checks 3, 4, 5 | PASS | FAIL |
| same | checks 2, 6, 7 | WARN in every validation run | — (they read the rehearsal's K562 screen) |
| `reports/validate.json` | our validator and `vcc prep --dry-run` | PASS | anything else |
| final checklist | items 5 and 7 `deviation` (D91), item 12 `deviation` if a shared scale was chosen (D117), **0 missing** | the same | a `missing`, or item 13 not `done` |

Print the rehearsal numbers in one go:

```bash
python - <<'PY'
import json
c = json.load(open("runs/final/rehearsal/report.json"))["calibration"]
print("objective:", c["objective_used"], "| fitted on:", c["fitted_on"])
print("chosen:", c["settings"])
for r in sorted(c["combined"], key=lambda r: -(r["mean"] if r["mean"] is not None else -9))[:6]:
    print(f"  effect {r['effect_scale']}  shared {r['shared_scale']}  rpe1 {r['mean']:+.4f}")
PY
```

Optional, and informative for the write-up: how the predictions are built.

```bash
python scripts/compare_predictions.py runs/server_shared runs/final
```

The `shared_share` and `target_corr` columns should look like
`server_shared`'s, the validation-round run that scored −0.032. Different
contexts will move them somewhat. A `target_corr` near 1.0 means every
target got nearly the same prediction.

## 7. Package and submit

`all` already wrote `runs/final/submission/prediction.vcc` when `vcc` is on
the PATH (checklist item 13). Otherwise:

```bash
python -m vccp package --config configs/server_final.yaml --seed 2 --force
ls -l runs/final/submission/          # the .vcc must be newer than prediction.h5ad
vcc submit runs/final/submission/prediction.vcc -m "final: shared 2 / specific 1, rpe1-calibrated" --wait
```

## 8. If something goes wrong

* **A stage fails.** `all` is resumable: fix the cause and rerun the same
  command. Finished stages are skipped. Rerun one stage with
  `python -m vccp <stage> --config configs/server_final.yaml --seed 2 --force`.
* **The rehearsal cannot build the leaderboard scale** (the objective reads
  `no_change`). Use the setting the validation leaderboard confirmed instead
  of the grid's choice, and redo the last stages:
  ```bash
  for stage in predict sanity validate package; do
    python -m vccp $stage --config configs/server_final.yaml --seed 2 --force --allow-warnings \
      --set predict.override_threshold=0.05 --set predict.override_scale=1.0 \
      --set predict.override_shared_scale=2.0 || break
  done
  ```
  The log says "generator settings overridden", and the prediction index
  records what was used.
* **Out of GPU memory.** The error names the key to lower
  (`train.output_genes_per_step` first).

## 9. Record it

- [ ] A `DECISIONS.md` entry for the final run: the data-prep coverage
      numbers, check-data sizes, Phase 2's held-out ratio, the chosen
      setting and its rpe1 score, the sanity summary, the checklist, and the
      leaderboard result when it comes.
