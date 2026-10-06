# KDE-KL vs MMD as the MLGD loss (2026-10-06)

Track 2 (algorithm experiments; nothing here changes the paper). Question: does GB-FESO's KDE estimate of the
symmetric KL (`--loss kde-kl`, bandwidth 0.5 unless noted) improve MLGD over the MMD, beyond the single N=5
peptide result?

## Peptide (MLGD, 2 trajectories, on GB-FESO's surrogate; scored by re-simulation, 40 targets)

| N | loss | mean dF err (k_BT) | median | dF err <= 0.3 | success (mean <= 1 k_BT) | time / run |
|---|---|---|---|---|---|---|
| 5 | MMD | 0.458 | 0.149 | 69% | 87.5% | 103 s |
| 5 | KDE-KL, h = 0.5 | **0.315** | **0.118** | **86%** | **95.0%** | 103 s |
| 5 | KDE-KL, h = 0.25 | 0.338 | 0.182 | 72% | 92.5% | 103 s |
| 5 | GB-FESO, 1 start (KL) | 0.182 | 0.090 | 79% | 98.8% | 119 s |
| 6 | MMD | **0.334** | **0.179** | 60% | 82.5% | 113 s |
| 6 | KDE-KL, h = 0.5 | 0.419 | 0.211 | **65%** | 85.0% | 116 s |
| 6 | GB-FESO, 1 start (KL) | 0.274 | 0.149 | 68% | 95.6% | 278 s |
| 4 | MMD | 0.118 | 0.102 | 100% | 100% | 90 s |
| 4 | KDE-KL, h = 0.5 | (evaluation rerun pending: first attempt hit the faulty node salmon-02) | | | | |

- N = 5: KL helps clearly (the earlier result); the smaller bandwidth 0.25 is worse than 0.5.
- N = 6: KL does **not** help: mean error 0.33 -> 0.42, median 0.18 -> 0.21; only the share within 0.3 rises (60 -> 65%).
- Same cost per run (the KDE is cheap next to the sampler).

## Synthetic grid (MLGD* recipe: d256 samplers, FiniteDiff guidance, 30 restarts, keep-best refinement, re-estimated selection; exact TV, 3 seeds; lower is better)

The MMD reference runs are the MLGD* runs behind Table 1 (`runs/night/grid/<cell>/G_zo_s42-44`; (10,1): `runs/zodirs/x10_y1/zodirs_coord_s42-44`, the same recipe).

| (p, q) | MMD: TV per seed | mean | KDE-KL (h = 0.5): TV per seed | mean | better |
|---|---|---|---|---|---|
| (1, 1) | .008 .035 .006 | **.016** | .079 .073 .005 | .052 | MMD |
| (1, 5) | .027 .017 .002 | **.015** | .045 .039 .031 | .038 | MMD |
| (1, 10) | .074 .095 .003 | .057 | .037 .056 .079 | .057 | tie |
| (5, 1) | .025 .027 .014 | **.022** | .036 .034 .034 | .035 | MMD |
| (5, 5) | .043 .092 .073 | .069 | .044 .059 .045 | **.049** | KL |
| (5, 10) | .045 .059 .116 | **.073** | .204 .184 .175 | .188 | MMD |
| (10, 1) | .138 .113 .041 | **.097** | .319 .097 .026 | .147 | MMD (KL: one failure at .32) |
| (10, 5) | .050 .048 .057 | .052 | .048 .045 .072 | .055 | tie |
| (10, 10) | .530 .066 .082 | .226 | .569 .282 .137 | .329 | MMD |

- MMD is better or tied in 8 of 9 cells; KL is better only at (5, 5) (as for GB-FESO's KL in Table 1).
- KL is clearly worse at q = 10 with p = 5 (.19 vs .07), consistent with a fixed-bandwidth KDE degrading with dim Y.
- Infrastructure: 15 of 27 tasks first died in 2 s without a log on salmon-02 / salmon-09 (both idle afterwards); resubmitted with `--exclude=salmon-02,salmon-09`. The N=4 peptide evaluation hit the same node.

## Recommendation

Do not switch the default loss. KL helped on one benchmark (peptide N = 5) and hurt or tied elsewhere (peptide N = 6, 8 of 9 synthetic cells). The likely
reason it helped at N = 5 — weighting small basins by log ratio, which is what the dF score measures — is specific to log-population metrics.
If pursued, an adaptive choice (KL only when the target is a few well-separated, small-variance modes and dim Y is small) or a combined
loss (MMD + λ KL) are the natural next tests; neither is supported by these data yet.
