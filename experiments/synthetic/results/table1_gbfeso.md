# Table 1 with the GB-FESO competitor

TV(P(Y | x̂), G), mean ± std over 5 runs (10 trajectories each); lower is better; **bold**: the best in the row.
The Table 1 methods are `table1.json` (this folder); GB-FESO (Zick & Mendels, arXiv:2607.06421) runs on the
same networks and data at the same sample budget: 250 conditional samples per step, 1,000 per refinement step
(GB-FESO: phase 1), scored on 1,000. GB-FESO (MMD) uses the Table 1 loss, so MLGD vs GB-FESO (MMD) differs only
in the search; GB-FESO (KL) is their method with their KDE-KL loss. Setup and command: `../README.md`,
section "GB-FESO competitor".

| dim X × dim Y | MLGD | MLGD-F | PMLGD | PMLGD-F | GB-FESO (KL) | GB-FESO (MMD) |
|---|---|---|---|---|---|---|
| 1×1 | 0.034 ± 0.021 | 0.072 ± 0.030 | 0.030 ± 0.013 | 0.077 ± 0.017 | **0.027 ± 0.016** | 0.037 ± 0.020 |
| 1×5 | **0.026 ± 0.009** | 0.056 ± 0.016 | 0.033 ± 0.023 | 0.070 ± 0.028 | 0.029 ± 0.019 | 0.028 ± 0.008 |
| 1×10 | 0.171 ± 0.053 | 0.366 ± 0.043 | **0.169 ± 0.117** | 0.374 ± 0.071 | 0.237 ± 0.163 | 0.172 ± 0.138 |
| 5×1 | 0.172 ± 0.118 | 0.203 ± 0.091 | 0.115 ± 0.027 | 0.178 ± 0.058 | **0.095 ± 0.057** | 0.274 ± 0.357 |
| 5×5 | 0.108 ± 0.031 | 0.484 ± 0.054 | 0.085 ± 0.043 | 0.487 ± 0.060 | **0.060 ± 0.040** | 0.113 ± 0.066 |
| 5×10 | **0.168 ± 0.032** | 0.401 ± 0.031 | 0.210 ± 0.044 | 0.447 ± 0.034 | 0.284 ± 0.105 | 0.194 ± 0.052 |
| 10×1 | 0.569 ± 0.086 | **0.537 ± 0.118** | 0.997 ± 0.004 | 0.819 ± 0.226 | 0.670 ± 0.191 | 0.681 ± 0.138 |
| 10×5 | 0.142 ± 0.023 | 0.295 ± 0.008 | 0.135 ± 0.108 | 0.308 ± 0.030 | **0.101 ± 0.033** | 0.158 ± 0.026 |
| 10×10 | **0.327 ± 0.127** | 0.943 ± 0.004 | 0.381 ± 0.169 | 0.855 ± 0.112 | 0.618 ± 0.070 | 0.454 ± 0.174 |
| time / run (s), mean | 509 | 46 | 506 | 47 | 486 | 489 |

Same loss (MMD), different search: MLGD's mean TV is at or below GB-FESO (MMD)'s in all 9 scenarios, by
the widest margins in the hardest ones (10×10, 10×1, 5×1). GB-FESO (KL) has the lowest TV at 1×1, 5×1, 5×5 and
10×5 (small dim Y), and degrades at dim Y = 10. With n = 5 runs, many differences are within one std.

## GB-FESO, all columns

`make_table1.py --collect --tag _main --methods GBFESO-KL GBFESO-MMD` (the columns of `table1.md`).

| dim X | dim Y | method | TV(P(Y|x_hat), G) mean +- std | ||x_hat-x*|| (sd units) | relative L2 | time / attempt (s) | trajectories with TV<0.5 | in x* basin |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | GBFESO-KL | 0.027 +- 0.016 (n=5) | 0.007 +- 0.004 | 0.006 +- 0.006 | 492 | 46/50 | 46/50 |
| 1 | 1 | GBFESO-MMD | 0.037 +- 0.020 (n=5) | 0.009 +- 0.005 | 0.012 +- 0.009 | 503 | 46/50 | 46/50 |
| 1 | 5 | GBFESO-KL | 0.029 +- 0.019 (n=5) | 0.028 +- 0.019 | 0.007 +- 0.007 | 474 | 43/50 | 43/50 |
| 1 | 5 | GBFESO-MMD | 0.028 +- 0.008 (n=5) | 0.027 +- 0.008 | 0.004 +- 0.002 | 476 | 44/50 | 44/50 |
| 1 | 10 | GBFESO-KL | 0.237 +- 0.163 (n=5) | 0.050 +- 0.036 | 0.360 +- 0.345 | 470 | 47/50 | 50/50 |
| 1 | 10 | GBFESO-MMD | 0.172 +- 0.138 (n=5) | 0.036 +- 0.030 | 0.221 +- 0.308 | 471 | 49/50 | 50/50 |
| 5 | 1 | GBFESO-KL | 0.095 +- 0.057 (n=5) | 0.871 +- 0.318 | 0.057 +- 0.044 | 488 | 24/50 | 8/50 |
| 5 | 1 | GBFESO-MMD | 0.274 +- 0.357 (n=5) | 1.714 +- 1.637 | 0.875 +- 1.648 | 498 | 20/50 | 9/50 |
| 5 | 5 | GBFESO-KL | 0.060 +- 0.040 (n=5) | 0.037 +- 0.019 | 0.019 +- 0.027 | 519 | 29/50 | 29/50 |
| 5 | 5 | GBFESO-MMD | 0.113 +- 0.066 (n=5) | 0.072 +- 0.039 | 0.061 +- 0.074 | 533 | 20/50 | 20/50 |
| 5 | 10 | GBFESO-KL | 0.284 +- 0.105 (n=5) | 0.168 +- 0.075 | 0.487 +- 0.321 | 486 | 47/50 | 50/50 |
| 5 | 10 | GBFESO-MMD | 0.194 +- 0.052 (n=5) | 0.120 +- 0.047 | 0.171 +- 0.076 | 481 | 48/50 | 50/50 |
| 10 | 1 | GBFESO-KL | 0.670 +- 0.191 (n=5) | 3.182 +- 0.727 | 1.504 +- 0.555 | 485 | 4/50 | 0/50 |
| 10 | 1 | GBFESO-MMD | 0.681 +- 0.138 (n=5) | 3.560 +- 0.446 | 1.474 +- 0.248 | 491 | 3/50 | 0/50 |
| 10 | 5 | GBFESO-KL | 0.101 +- 0.033 (n=5) | 0.380 +- 0.138 | 0.118 +- 0.100 | 492 | 28/50 | 0/50 |
| 10 | 5 | GBFESO-MMD | 0.158 +- 0.026 (n=5) | 0.871 +- 0.563 | 0.325 +- 0.136 | 483 | 26/50 | 0/50 |
| 10 | 10 | GBFESO-KL | 0.618 +- 0.070 (n=5) | 0.213 +- 0.094 | 1.870 +- 0.161 | 466 | 13/50 | 0/50 |
| 10 | 10 | GBFESO-MMD | 0.454 +- 0.174 (n=5) | 0.236 +- 0.026 | 1.046 +- 0.664 | 466 | 22/50 | 0/50 |
