# Table 15 — `tab:sd_swd`

Published numbers, transcribed from the paper. **Imported, not reproduced.**
Nothing in this repository re-ran them.

MMD and SWD across all four target scenarios (N = 2,000 images). SWD is computed on the
same images as Table 3 and is not part of the optimisation objective. ΔSWD (%) is
defined analogously to ΔMMD (%).

| Scenario | Method | MMD ↓ | ΔMMD (%) | SWD ↓ | ΔSWD (%) |
|---|---|---|---|---|---|
| Balanced (50% male) | Source      | 0.565 | +0.0%  | 0.0136 | +0.0%  |
| Balanced (50% male) | Avg prompt  | 0.533 | +5.7%  | 0.0122 | +10.1% |
| Balanced (50% male) | SDEdit Best | 0.472 | +16.5% | 0.0122 | +10.8% |
| Balanced (50% male) | **MLGD-F**  | **0.386** | **+31.7%** | **0.0106** | **+21.9%** |
| Skewed (25% male)   | Source      | 0.629 | +0.0%  | 0.0146 | +0.0%  |
| Skewed (25% male)   | Avg prompt  | 0.551 | +12.4% | 0.0130 | +10.8% |
| Skewed (25% male)   | SDEdit Best | 0.527 | +16.2% | 0.0146 | −0.4%  |
| Skewed (25% male)   | **MLGD-F**  | **0.373** | **+40.8%** | **0.0093** | **+36.3%** |
| Gender interpolation | Source      | 0.497 | +0.0%  | 0.0120 | +0.0%  |
| Gender interpolation | Avg prompt  | 0.451 | +9.1%  | 0.0095 | +21.1% |
| Gender interpolation | SDEdit Best | 0.437 | +12.0% | 0.0116 | +3.1%  |
| Gender interpolation | **MLGD-F**  | **0.328** | **+33.9%** | **0.0072** | **+40.0%** |
| Age interpolation   | Source      | 0.464 | +0.0%  | 0.0108 | +0.0%  |
| Age interpolation   | Avg prompt  | 0.440 | +5.3%  | 0.0103 | +4.7%  |
| Age interpolation   | SDEdit Best | 0.445 | +4.1%  | 0.0119 | −10.4% |
| Age interpolation   | **MLGD-F**  | **0.381** | **+18.1%** | **0.0091** | **+15.4%** |
