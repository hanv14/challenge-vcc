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

| upload | run dir | parent | the one change | thr / effect / shared | overall | pds | expr | nmae | fid | reach | jac | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4 | `runs/server` | — | (old model, 98% shared) | 0.1 / 0.5 / — | −0.014 | −0.006 | 0 | −0.037 | −0.015 | −0.002 | −0.027 | D113, D117 |
| v5 | `runs/server_12k` | — | — | 0.0 / 1.0 / — | −0.083 | −0.004 | 0 | −0.504 | +0.009 | +0.004 | −0.001 | D113 |
| v6 | `runs/server_fix` | — | — | 0.25 / 0.25 / — | −0.242 | −0.016 | 0 | +0.002 | −1.340 | −0.019 | −0.078 | D113 |
| v7 | `runs/server_cal` | — | — | 0.05 / 1.5 / — | −0.0673 | −0.0059 | 0 | −0.0860 | −0.2537 | −0.0145 | −0.0435 | D116 |
| v8 | `runs/server_cal_s05` | v7 | generator setting | 0.1 / 0.5 / — | −0.1534 | −0.0079 | 0 | −0.0011 | −0.8257 | −0.0191 | −0.0668 | D116 |
| shared 2 / specific 1 | `runs/server_shared` | — | — | 0.05 / 1.0 / 2.0 | **−0.0318** | −0.0063 | 0 | −0.0907 | −0.0589 | −0.0090 | −0.0256 | D118 |
| single scale 2 | `runs/server_single2` | shared 2 / specific 1 | shared scale = effect scale | 0.05 / 2.0 / — | −0.0641 | −0.0037 | 0 | −0.1599 | −0.1673 | −0.0158 | −0.0378 | D118 |

Rows before this file existed were not designed as paired experiments;
v4, v5, v6 and v7 differ in model and setting at once (D115).
