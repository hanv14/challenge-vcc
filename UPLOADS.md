# UPLOADS.md: every leaderboard upload, one row each

The validation leaderboard is the instrument for choosing settings
(REDESIGN.md §5). Each upload is a paired experiment against a named parent:
**one** change, the same seed unless the seed is the change. Scores are on
the leaderboard's scale (0 = the organizers' oracle mean-response baseline,
1 = a split-half replicate); `expr` is clamped to [0, 1] and has read 0 on
every upload so far (D120).

Anchor set for every validation upload so far:
`vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4`, panel `vcc2026-val-1` (D123).

## Validation round

| upload | run dir | parent | the one change | thr / effect / shared | lookup | overall | pds | expr | nmae | fid | reach | jac | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4 | `runs/server` | — | (old model, 98% shared) | 0.1 / 0.5 / — | — | −0.014 | −0.006 | 0 | −0.037 | −0.015 | −0.002 | −0.027 | D113, D117 |
| v5 | `runs/server_12k` | — | — | 0.0 / 1.0 / — | — | −0.083 | −0.004 | 0 | −0.504 | +0.009 | +0.004 | −0.001 | D113 |
| v6 | `runs/server_fix` | — | — | 0.25 / 0.25 / — | — | −0.242 | −0.016 | 0 | +0.002 | −1.340 | −0.019 | −0.078 | D113 |
| v7 | `runs/server_cal` | — | — | 0.05 / 1.5 / — | — | −0.0673 | −0.0059 | 0 | −0.0860 | −0.2537 | −0.0145 | −0.0435 | D116 |
| v8 | `runs/server_cal_s05` | v7 | generator setting | 0.1 / 0.5 / — | — | −0.1534 | −0.0079 | 0 | −0.0011 | −0.8257 | −0.0191 | −0.0668 | D116 |
| shared 2 / specific 1 | `runs/server_shared` | — | — | 0.05 / 1.0 / 2.0 | — | −0.0318 | −0.0063 | 0 | −0.0907 | −0.0589 | −0.0090 | −0.0256 | D118 |
| single scale 2 | `runs/server_single2` | shared 2 / specific 1 | shared scale = effect scale | 0.05 / 2.0 / — | — | −0.0641 | −0.0037 | 0 | −0.1599 | −0.1673 | −0.0158 | −0.0378 | D118 |
| P0 | `runs/probe_p0` | shared 2 / specific 1 | generator seed 2 → 3 (the noise floor) | 0.05 / 1.0 / 2.0 | — | −0.0298 | +0.0018 | 0 | −0.0912 | −0.0545 | −0.0092 | −0.0256 | D125 |
| P1 | `runs/probe_p1` | shared 2 / specific 1 | model's specific part off (effect 1.0 → 0.01) | 0.05 / 0.01 / 2.0 | — | −0.0224 | −0.0062 | 0 | −0.0677 | −0.0291 | −0.0113 | −0.0202 | D125 |
| P2 | `runs/probe_p2` | P1 | shared 2.0 → 1.5 | 0.05 / 0.01 / 1.5 | — | −0.0255 | −0.0091 | 0 | −0.0291 | −0.0690 | −0.0145 | −0.0311 | D125 |
| P3 | `runs/probe_p3` | P1 | shared 2.0 → 3.0 | 0.05 / 0.01 / 3.0 | — | −0.0320 | −0.0040 | 0 | −0.1730 | +0.0018 | −0.0095 | −0.0074 | D125 |
| P4 | `runs/probe_p4` | shared 2 / specific 1 | model's specific part halved (effect 1.0 → 0.5) | 0.05 / 0.5 / 2.0 | — | −0.0244 | −0.0056 | 0 | −0.0741 | −0.0342 | −0.0114 | −0.0213 | D125 |
| L1 | `runs/probe_l1` | P1 | lookup on at 0.5, gene-shrunk | 0.05 / 0.01 / 2.0 | 0.5, gene | −0.0151 | +0.0071 | 0 | −0.0523 | −0.0244 | −0.0019 | −0.0189 | D126 |
| L2 | `runs/probe_l2` | L1 | lookup scale 0.5 → 1.0 | 0.05 / 0.01 / 2.0 | 1.0, gene | −0.0097 | +0.0197 | 0 | −0.0444 | −0.0206 | +0.0049 | −0.0177 | D126 |
| L3 | `runs/probe_l3` | L2 | reliability weighting | 0.05 / 0.01 / 2.0 | 1.0, gene, reliability | −0.0193 | +0.0011 | 0 | −0.0591 | −0.0275 | −0.0108 | −0.0195 | D127 |
| L4 | `runs/probe_l4` | L2 | shrinkage gene → none (raw) | 0.05 / 0.01 / 2.0 | 1.0, none | +0.0098 | +0.0897 | 0 | −0.0615 | +0.0047 | +0.0352 | −0.0093 | D127 |
| L5 | `runs/probe_l5` | L4 | lookup scale 1.0 → 1.5 | 0.05 / 0.01 / 2.0 | 1.5, none | +0.0050 | +0.1268 | 0 | −0.1333 | +0.0055 | +0.0391 | −0.0079 | D129 |
| L6 | `runs/probe_l6` | L4 | shared 2.0 → 1.5 | 0.05 / 0.01 / 1.5 | 1.0, none | +0.0189 | +0.1146 | 0 | −0.0195 | −0.0073 | +0.0393 | −0.0137 | D129 |
| D1 | `runs/probe_d1` | L6 | lookup soft-SNR τ = 4, energy matched to raw | 0.05 / 0.01 / 1.5 | 1.0, snr τ=4, matched | +0.0096 | +0.1418 | 0 | −0.0916 | −0.0128 | +0.0373 | −0.0173 | D132 |
| L7 | `runs/probe_l7` | L6 | shared 1.5 → 1.0 | 0.05 / 0.01 / 1.0 | 1.0, none | **+0.0286** | +0.1572 | 0 | +0.0120 | −0.0231 | +0.0444 | −0.0187 | D131 |

The probes P0–L4 were built by `scripts/probe.sh` (commit `aca4f08`; `fc03171`
changed documents only) from the parent run `runs/server_shared`, at generator
seed 2 unless the row says otherwise, so each differs from its parent in the
one change named. The lookup reads K562 genome-wide only (272 of 300 targets;
table `runs/lookup_cache`, D126). "Effect 0.01" stands for "the model's
specific part off": `shared_scale` needs a positive effect scale, and at 0.01
the 0.05 threshold no longer removes anything either (D125). **L4 is the
parent of L5 and L6; L6 of D1 and L7; L7 (rank 836) of everything after.**
L5 and L6 ran on code at `c105982` or later; the raw lookup is bit-identical
there (D128).

Seed noise, from P0 against its parent: 0.0020 overall, 0.0081 on pds, 0.0044
on fid, under 0.001 on the rest. A change is adopted when it beats its parent
by more than twice that, about 0.004 overall (REDESIGN.md §5.1 rule 2).

### Planned, hypotheses written before the upload

| upload | command | parent | the one change | hypothesis, expected sign | adopt if |
|---|---|---|---|---|---|
| L8 | `scripts/probe.sh probe_l8 2 0.05 0.01 0.5 --set predict.lookup_scale=1.0 --set predict.lookup_shrinkage=none` | L7 | shared 1.0 → 0.5 | Continues the bracket (two steps of +0.009 and +0.010). pds up again (+0.02 to +0.05); fid down further (−0.02 to −0.05), most of the truth's calls being the shared shift's; nmae, now above the oracle, may turn down as the shared part gets undersized. Overall −0.01 to +0.01: where the trend turns is the question. (D131) | > +0.004 |
| D2 | `scripts/probe.sh probe_d2 2 0.05 0.01 1.0 --set predict.lookup_scale=1.0 --set predict.lookup_shrinkage=snr --set predict.lookup_snr_tau=4` | L7 | the lookup soft-shrunk by SNR (τ = 4), **same scale** | The agreed test D1 was not: strong entries stay at their K562 size, noise shrinks. nmae up (+0.01 to +0.05: less noise on the truth's significant genes, no amplification); pds down (−0.03 to 0: dilution, partly offset by direction); jac up (fewer noise calls); reach ±0.005. Overall −0.005 to +0.01. (D132) | > +0.004; under +0.01, a context-isolated pair before the final round |

L5–L7 and D1 were planned here before their uploads; D129, D131 and D132
check them against the results.

Per-context scores are not available: neither the `vcc` tool nor the website
reports them (2026-10-06). Each row is the average over the round's contexts;
REDESIGN.md §5.1 rule 3 says when a context-isolated pair is worth two uploads.

Rows before this file existed were not designed as paired experiments;
v4, v5, v6 and v7 differ in model and setting at once (D115).
