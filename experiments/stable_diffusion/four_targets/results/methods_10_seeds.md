# Methods over seeds (Stable Diffusion, base_zeta 7.5)

Each seed is a different source sketch. Per scenario and method: the final MMD and SWD to the target (CLIP
embeddings, N = 2,000 evaluation) and their improvement over the source sketch, as mean ± SD over the seeds;
and the method's seed with the lowest MMD alone, with its value (the SWD table uses the same seed). Only
deterministic runs are used. Written by `experiments/stable_diffusion/seed_deltas.py`.

## MMD: delta = mmd(source) - mmd(method), mean ± SD over seeds (positive = improvement)

| scenario | method | n seeds | MMD (method) | Δ | Δ % | MMD (source) | seed selected by MMD | its MMD | its Δ % |
|---|---|---|---|---|---|---|---|---|---|
| BalancedTarget | mlgd_f | 10 | 0.4208 ± 0.0425 | 0.0987 ± 0.0337 | 19.0 ± 6.2 | 0.5195 ± 0.0377 | 4 | 0.3477 | +31.3 |
| BalancedTarget | avg_prompt | 10 | 0.4912 ± 0.0270 | 0.0283 ± 0.0420 | 5.0 ± 8.4 | 0.5195 ± 0.0377 | 4 | 0.4526 | +10.6 |
| BalancedTarget | sdedit_best | 10 | 0.4567 ± 0.0183 | 0.0628 ± 0.0263 | 11.8 ± 4.6 | 0.5195 ± 0.0377 | 6 | 0.4280 | +14.7 |
| SkewedTarget | mlgd_f | 10 | 0.4092 ± 0.0529 | 0.1840 ± 0.0660 | 30.7 ± 10.1 | 0.5931 ± 0.0470 | 4 | 0.3237 | +45.7 |
| SkewedTarget | avg_prompt | 10 | 0.4969 ± 0.0413 | 0.0962 ± 0.0637 | 15.7 ± 10.4 | 0.5931 ± 0.0470 | 10 | 0.4098 | +31.7 |
| SkewedTarget | sdedit_best | 10 | 0.4922 ± 0.0369 | 0.1009 ± 0.0316 | 16.8 ± 5.0 | 0.5931 ± 0.0470 | 3 | 0.4418 | +7.8 |
| GenderInterpolation | mlgd_f | 10 | 0.4050 ± 0.0615 | 0.0981 ± 0.0577 | 19.3 ± 10.7 | 0.5031 ± 0.0497 | 9 | 0.3411 | +26.9 |
| GenderInterpolation | avg_prompt | 10 | 0.4704 ± 0.0279 | 0.0327 ± 0.0543 | 5.6 ± 11.5 | 0.5031 ± 0.0497 | 8 | 0.4371 | +18.3 |
| GenderInterpolation | sdedit_best | 10 | 0.4490 ± 0.0248 | 0.0541 ± 0.0290 | 10.3 ± 5.2 | 0.5031 ± 0.0497 | 3 | 0.4012 | +0.1 |
| AgeInterpolation | mlgd_f | 10 | 0.4860 ± 0.1375 | -0.0062 ± 0.1022 | -0.6 ± 19.1 | 0.4798 ± 0.0638 | 2 | 0.3451 | +33.6 |
| AgeInterpolation | avg_prompt | 10 | 0.4443 ± 0.0364 | 0.0355 ± 0.0402 | 6.7 ± 7.5 | 0.4798 ± 0.0638 | 4 | 0.3776 | -3.5 |
| AgeInterpolation | sdedit_best | 10 | 0.4826 ± 0.0671 | -0.0029 ± 0.0519 | -0.9 ± 10.0 | 0.4798 ± 0.0638 | 4 | 0.3549 | +2.7 |

## SWD: delta = swd(source) - swd(method), mean ± SD over seeds (positive = improvement)

| scenario | method | n seeds | SWD (method) | Δ | Δ % | SWD (source) | seed selected by MMD | its SWD | its Δ % |
|---|---|---|---|---|---|---|---|---|---|
| BalancedTarget | mlgd_f | 10 | 0.0104 ± 0.0016 | 0.0013 ± 0.0016 | 10.3 ± 12.9 | 0.0117 ± 0.0017 | 4 | 0.0080 | +28.1 |
| BalancedTarget | avg_prompt | 10 | 0.0114 ± 0.0012 | 0.0002 ± 0.0024 | -0.2 ± 20.7 | 0.0117 ± 0.0017 | 4 | 0.0098 | +11.3 |
| BalancedTarget | sdedit_best | 10 | 0.0112 ± 0.0008 | 0.0004 ± 0.0018 | 2.1 ± 13.8 | 0.0117 ± 0.0017 | 6 | 0.0103 | +13.4 |
| SkewedTarget | mlgd_f | 10 | 0.0104 ± 0.0016 | 0.0040 ± 0.0018 | 27.2 ± 9.8 | 0.0144 ± 0.0026 | 4 | 0.0082 | +33.6 |
| SkewedTarget | avg_prompt | 10 | 0.0115 ± 0.0010 | 0.0028 ± 0.0029 | 17.3 ± 16.7 | 0.0144 ± 0.0026 | 10 | 0.0098 | +39.0 |
| SkewedTarget | sdedit_best | 10 | 0.0127 ± 0.0023 | 0.0017 ± 0.0019 | 10.7 ± 13.2 | 0.0144 ± 0.0026 | 3 | 0.0101 | +0.7 |
| GenderInterpolation | mlgd_f | 10 | 0.0099 ± 0.0025 | 0.0015 ± 0.0019 | 13.1 ± 16.5 | 0.0115 ± 0.0020 | 9 | 0.0062 | +32.4 |
| GenderInterpolation | avg_prompt | 10 | 0.0102 ± 0.0008 | 0.0013 ± 0.0020 | 8.6 ± 16.5 | 0.0115 ± 0.0020 | 8 | 0.0100 | +24.7 |
| GenderInterpolation | sdedit_best | 10 | 0.0106 ± 0.0013 | 0.0009 ± 0.0016 | 6.0 ± 14.1 | 0.0115 ± 0.0020 | 3 | 0.0088 | -5.7 |
| AgeInterpolation | mlgd_f | 10 | 0.0136 ± 0.0056 | -0.0021 ± 0.0047 | -17.2 ± 34.5 | 0.0114 ± 0.0018 | 2 | 0.0077 | +41.5 |
| AgeInterpolation | avg_prompt | 10 | 0.0103 ± 0.0012 | 0.0012 ± 0.0016 | 8.7 ± 13.9 | 0.0114 ± 0.0018 | 4 | 0.0090 | +0.3 |
| AgeInterpolation | sdedit_best | 10 | 0.0133 ± 0.0032 | -0.0018 ± 0.0031 | -17.0 ± 27.9 | 0.0114 ± 0.0018 | 4 | 0.0082 | +8.4 |

## MLGD-F: mean ± std over all seeds, and the seed selected by MMD alone (lowest MLGD-F MMD)

| scenario | n seeds | MMD | Δ MMD % | SWD | Δ SWD % | p(male) | selected seed | its MMD | its Δ MMD % | its SWD | its p(male) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BalancedTarget | 10 | 0.4208 ± 0.0425 | 19.0 ± 6.2 | 0.0104 ± 0.0016 | 10.3 ± 12.9 | 0.609 ± 0.158 | 4 | 0.3477 | +31.3 | 0.0080 | 0.493 |
| SkewedTarget | 10 | 0.4092 ± 0.0529 | 30.7 ± 10.1 | 0.0104 ± 0.0016 | 27.2 ± 9.8 | 0.355 ± 0.121 | 4 | 0.3237 | +45.7 | 0.0082 | 0.175 |
| GenderInterpolation | 10 | 0.4050 ± 0.0615 | 19.3 ± 10.7 | 0.0099 ± 0.0025 | 13.1 ± 16.5 | 0.586 ± 0.130 | 9 | 0.3411 | +26.9 | 0.0062 | 0.631 |
| AgeInterpolation | 10 | 0.4860 ± 0.1375 | -0.6 ± 19.1 | 0.0136 ± 0.0056 | -17.2 ± 34.5 | - | 2 | 0.3451 | +33.6 | 0.0077 | - |

runs used:
  AgeInterpolation seed 1: full_AgeInterpolation_seed1_zeta7.5_46276539
  AgeInterpolation seed 2: full_AgeInterpolation_seed2_zeta7.5_46276541
  AgeInterpolation seed 3: full_AgeInterpolation_seed3_zeta7.5_46276543
  AgeInterpolation seed 4: full_AgeInterpolation_seed4_zeta7.5_46276545
  AgeInterpolation seed 5: full_AgeInterpolation_seed5_zeta7.5_46276547
  AgeInterpolation seed 6: full_AgeInterpolation_seed6_zeta7.5_46284561
  AgeInterpolation seed 7: full_AgeInterpolation_seed7_zeta7.5_46284564
  AgeInterpolation seed 8: full_AgeInterpolation_seed8_zeta7.5_46284567
  AgeInterpolation seed 9: full_AgeInterpolation_seed9_zeta7.5_46284570
  AgeInterpolation seed 10: full_AgeInterpolation_seed10_zeta7.5_46284573
  BalancedTarget seed 1: full_BalancedTarget_seed1_zeta7.5_46275288
  BalancedTarget seed 2: full_BalancedTarget_seed2_zeta7.5_46788115
  BalancedTarget seed 3: full_BalancedTarget_seed3_zeta7.5_46275290
  BalancedTarget seed 4: full_BalancedTarget_seed4_zeta7.5_46275291
  BalancedTarget seed 5: full_BalancedTarget_seed5_zeta7.5_46275292
  BalancedTarget seed 6: full_BalancedTarget_seed6_zeta7.5_46284597
  BalancedTarget seed 7: full_BalancedTarget_seed7_zeta7.5_46284600
  BalancedTarget seed 8: full_BalancedTarget_seed8_zeta7.5_46284603
  BalancedTarget seed 9: full_BalancedTarget_seed9_zeta7.5_46284606
  BalancedTarget seed 10: full_BalancedTarget_seed10_zeta7.5_46284609
  GenderInterpolation seed 1: full_GenderInterpolation_seed1_zeta7.5_46275298
  GenderInterpolation seed 2: full_GenderInterpolation_seed2_zeta7.5_46275300
  GenderInterpolation seed 3: full_GenderInterpolation_seed3_zeta7.5_46275301
  GenderInterpolation seed 4: full_GenderInterpolation_seed4_zeta7.5_46788117
  GenderInterpolation seed 5: full_GenderInterpolation_seed5_zeta7.5_46275303
  GenderInterpolation seed 6: full_GenderInterpolation_seed6_zeta7.5_46284627
  GenderInterpolation seed 7: full_GenderInterpolation_seed7_zeta7.5_46284630
  GenderInterpolation seed 8: full_GenderInterpolation_seed8_zeta7.5_46284633
  GenderInterpolation seed 9: full_GenderInterpolation_seed9_zeta7.5_46284636
  GenderInterpolation seed 10: full_GenderInterpolation_seed10_zeta7.5_46284639
  SkewedTarget seed 1: full_SkewedTarget_seed1_zeta7.5_46275293
  SkewedTarget seed 2: full_SkewedTarget_seed2_zeta7.5_46275294
  SkewedTarget seed 3: full_SkewedTarget_seed3_zeta7.5_46275295
  SkewedTarget seed 4: full_SkewedTarget_seed4_zeta7.5_46275296
  SkewedTarget seed 5: full_SkewedTarget_seed5_zeta7.5_46275297
  SkewedTarget seed 6: full_SkewedTarget_seed6_zeta7.5_46284612
  SkewedTarget seed 7: full_SkewedTarget_seed7_zeta7.5_46284615
  SkewedTarget seed 8: full_SkewedTarget_seed8_zeta7.5_46284618
  SkewedTarget seed 9: full_SkewedTarget_seed9_zeta7.5_46284621
  SkewedTarget seed 10: full_SkewedTarget_seed10_zeta7.5_46284624
