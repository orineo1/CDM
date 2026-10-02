# MLGD-F runs (full), exported from `outputs/full_*`

Written by `experiments/stable_diffusion/export_results.py`, one row per run. The N = 2,000
columns are the paper's protocol (`run.sh eval`, Table 3; `--` until it has run); "run MMD"
is the run's own end-of-run MMD on n images against its target pool, comparable in scale
but not the same protocol.

| Run | Scenario | seed | job | MLGD-F (N=2000) ↓ | source | sdedit_best | avg_prompt | paper MLGD-F | run MMD (n) | guided steps | variations / k | ζ | time (min) | commit | wandb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_BalancedTarget_seed4 | BalancedTarget | 4 | -- | 0.348 | 0.506 | 0.439 | 0.453 | 0.386 | 0.361 (250) | 125 | 100 / 50 | 7.5 | 288.4 | 197ca7e | [full_BalancedTarget_seed4_zeta7.5_46275291](https://wandb.ai/conditional-matching/sdxl-mlgdf-balanced/runs/xwsjwdy3) |
| full_SkewedTarget_seed6 | SkewedTarget | 6 | -- | 0.368 | 0.582 | 0.443 | 0.476 | 0.373 | 0.399 (250) | 125 | 100 / 50 | 7.5 | 284.6 | a2b3124 | [full_SkewedTarget_seed6_zeta7.5_46284612](https://wandb.ai/conditional-matching/sdxl-mlgdf-skewed/runs/m1x5are4) |
| full_GenderInterpolation_seed8 | GenderInterpolation | 8 | -- | 0.349 | 0.535 | 0.455 | 0.437 | 0.328 | 0.350 (250) | 125 | 100 / 50 | 7.5 | 283.9 | a2b3124 | [full_GenderInterpolation_seed8_zeta7.5_46284633](https://wandb.ai/conditional-matching/sdxl-mlgdf-gender_interpolation/runs/iwqf89qh) |
| full_AgeInterpolation_seed2 | AgeInterpolation | 2 | -- | 0.345 | 0.520 | 0.481 | 0.447 | 0.381 | 0.361 (250) | 125 | 120 / 60 | 7.5 | 342.1 | 197ca7e | [full_AgeInterpolation_seed2_zeta7.5_46276541](https://wandb.ai/conditional-matching/sdxl-mlgdf-age_interpolation/runs/z80c1vrw) |
