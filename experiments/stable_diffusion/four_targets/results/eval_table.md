# Final evaluation (N = 2,000), every run

Written by `experiments/stable_diffusion/export_results.py` from each run's `eval/`; the
protocol of the paper's Table 3 / Table 15 (`cdm/models/sd/eval_all_experiments.py`).
Delta% is the improvement over the source scribble.

## full_BalancedTarget_seed4: N = 2000

| Method | MMD ↓ | ΔMMD (%) | SWD ↓ | ΔSWD (%) | p(male) | 95% CI | n_male / n_female |
|---|---|---|---|---|---|---|---|
| source | 0.506 | +0.0% | 0.0111 | +0.0% | 0.882 | [0.868, 0.897] | 1765 / 235 |
| avg_prompt | 0.453 | +10.6% | 0.0098 | +11.3% | 0.412 | [0.390, 0.434] | 824 / 1176 |
| sdedit_best | 0.439 | +13.2% | 0.0111 | +0.0% | 0.651 | [0.631, 0.672] | 1303 / 697 |
| mlgd_f | 0.348 | +31.3% | 0.0080 | +28.1% | 0.493 | [0.472, 0.515] | 987 / 1013 |

## full_SkewedTarget_seed6: N = 2000

| Method | MMD ↓ | ΔMMD (%) | SWD ↓ | ΔSWD (%) | p(male) | 95% CI | n_male / n_female |
|---|---|---|---|---|---|---|---|
| source | 0.582 | +0.0% | 0.0130 | +0.0% | 0.896 | [0.883, 0.909] | 1792 / 208 |
| avg_prompt | 0.476 | +18.1% | 0.0118 | +9.0% | 0.207 | [0.190, 0.225] | 415 / 1585 |
| sdedit_best | 0.443 | +23.8% | 0.0103 | +20.3% | 0.430 | [0.409, 0.452] | 861 / 1139 |
| mlgd_f | 0.368 | +36.8% | 0.0093 | +28.0% | 0.246 | [0.227, 0.265] | 492 / 1508 |

## full_GenderInterpolation_seed8: N = 2000

| Method | MMD ↓ | ΔMMD (%) | SWD ↓ | ΔSWD (%) | p(male) | 95% CI | n_male / n_female |
|---|---|---|---|---|---|---|---|
| source | 0.535 | +0.0% | 0.0133 | +0.0% | 0.986 | [0.980, 0.991] | 1971 / 29 |
| avg_prompt | 0.437 | +18.3% | 0.0100 | +24.7% | 0.778 | [0.760, 0.796] | 1556 / 444 |
| sdedit_best | 0.455 | +15.0% | 0.0108 | +18.4% | 0.745 | [0.725, 0.764] | 1489 / 511 |
| mlgd_f | 0.349 | +34.8% | 0.0082 | +38.1% | 0.537 | [0.515, 0.559] | 1074 / 926 |

## full_AgeInterpolation_seed2: N = 2000

| Method | MMD ↓ | ΔMMD (%) | SWD ↓ | ΔSWD (%) |
|---|---|---|---|---|
| source | 0.520 | +0.0% | 0.0131 | +0.0% |
| avg_prompt | 0.447 | +14.0% | 0.0093 | +29.0% |
| sdedit_best | 0.481 | +7.5% | 0.0118 | +10.4% |
| mlgd_f | 0.345 | +33.6% | 0.0077 | +41.5% |
