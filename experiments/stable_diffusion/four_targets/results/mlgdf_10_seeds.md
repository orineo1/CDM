# MLGD-F over seeds (Stable Diffusion, base_zeta 7.5)

Each seed is a different source sketch. Per scenario: MLGD-F's final MMD and SWD to the target (CLIP
embeddings, N = 2,000 evaluation), their improvement over the source sketch (Δ %, positive = better),
and p(male), as mean ± std over the seeds; and the same for the seed selected by MMD alone (lowest
MLGD-F MMD). Only deterministic runs are used. Written by `experiments/stable_diffusion/seed_deltas.py --mlgdf-report`.

## MLGD-F: mean ± std over all seeds, and the seed selected by MMD alone (lowest MLGD-F MMD)

| scenario | n seeds | MMD | Δ MMD % | SWD | Δ SWD % | p(male) | selected seed | its MMD | its Δ MMD % | its SWD | its p(male) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BalancedTarget | 10 | 0.4207 ± 0.0425 | 19.0 ± 6.2 | 0.0104 ± 0.0016 | 10.3 ± 12.9 | 0.609 ± 0.158 | 4 | 0.3477 | +31.3 | 0.0080 | 0.493 |
| SkewedTarget | 10 | 0.4091 ± 0.0529 | 30.7 ± 10.1 | 0.0104 ± 0.0016 | 27.2 ± 9.8 | 0.355 ± 0.121 | 4 | 0.3235 | +45.7 | 0.0082 | 0.175 |
| GenderInterpolation | 10 | 0.4050 ± 0.0615 | 19.3 ± 10.7 | 0.0099 ± 0.0025 | 13.1 ± 16.5 | 0.586 ± 0.130 | 9 | 0.3411 | +26.9 | 0.0062 | 0.631 |
| AgeInterpolation | 10 | 0.4859 ± 0.1375 | -0.6 ± 19.1 | 0.0136 ± 0.0056 | -17.2 ± 34.5 | - | 2 | 0.3451 | +33.6 | 0.0077 | - |

## The same without AgeInterpolation seed 6, AgeInterpolation seed 9

| scenario | n seeds | MMD | Δ MMD % | SWD | Δ SWD % | p(male) | selected seed | its MMD | its Δ MMD % | its SWD | its p(male) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BalancedTarget | 10 | 0.4207 ± 0.0425 | 19.0 ± 6.2 | 0.0104 ± 0.0016 | 10.3 ± 12.9 | 0.609 ± 0.158 | 4 | 0.3477 | +31.3 | 0.0080 | 0.493 |
| SkewedTarget | 10 | 0.4091 ± 0.0529 | 30.7 ± 10.1 | 0.0104 ± 0.0016 | 27.2 ± 9.8 | 0.355 ± 0.121 | 4 | 0.3235 | +45.7 | 0.0082 | 0.175 |
| GenderInterpolation | 10 | 0.4050 ± 0.0615 | 19.3 ± 10.7 | 0.0099 ± 0.0025 | 13.1 ± 16.5 | 0.586 ± 0.130 | 9 | 0.3411 | +26.9 | 0.0062 | 0.631 |
| AgeInterpolation | 8 | 0.4242 ± 0.0477 | 6.8 ± 12.3 | 0.0111 ± 0.0023 | -3.9 ± 22.5 | - | 2 | 0.3451 | +33.6 | 0.0077 | - |

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
