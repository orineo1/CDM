# Table 3 — `tab:sd_results`

Published numbers, transcribed from the paper. **Imported, not reproduced.**
Nothing in this repository re-ran them.

Distributional targets, N = 2,000 images; best per scenario in bold in the paper.
MMD = (MMD²)^(1/2); ΔMMD is the relative improvement over the source scribble.

## Discrete targets

| Scenario | Method | MMD ↓ | ΔMMD | % male (95% CI) |
|---|---|---|---|---|
| Balanced (50% male) | Avg prompt   | 0.533 | +5.7%  | 98.4 [97.8, 98.9] |
| Balanced (50% male) | SDEdit Best  | 0.472 | +16.5% | 88.8 [87.4, 90.2] |
| Balanced (50% male) | **MLGD-F**   | **0.386** | **+31.7%** | **51.1 [49.0, 53.3]** |
| Skewed (25% male)   | Avg prompt   | 0.551 | +12.4% | 3.8 [3.0, 4.7] |
| Skewed (25% male)   | SDEdit Best  | 0.527 | +16.2% | 76.6 [74.7, 78.5] |
| Skewed (25% male)   | **MLGD-F**   | **0.373** | **+40.8%** | **26.6 [24.7, 28.5]** |

## Continuum targets

| Scenario | Method | MMD ↓ | ΔMMD |
|---|---|---|---|
| Gender interpolation | Avg prompt  | 0.451 | +9.1% |
| Gender interpolation | SDEdit Best | 0.437 | +12.0% |
| Gender interpolation | **MLGD-F**  | **0.328** | **+33.9%** |
| Age interpolation    | Avg prompt  | 0.440 | +5.3% |
| Age interpolation    | SDEdit Best | 0.445 | +4.1% |
| Age interpolation    | **MLGD-F**  | **0.381** | **+18.1%** |
