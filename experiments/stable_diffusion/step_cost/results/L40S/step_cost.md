# One MLGD-F guidance step: NVIDIA L40S, deterministic kernels: False

## Memory and run time, All selection (NVIDIA L40S)

Turbo (K = 2) measured; Base (K* = 30) projected: per-sample part x 30/2, models and fixed part unchanged. Run = 125 guided steps.

| n_cond | models (GB) | fixed (GB) | samples (GB) | Turbo peak (GB) | Base peak (GB, proj.) | Base / GPU memory | Turbo run (h) | Base run (h, proj.) |
|---|---|---|---|---|---|---|---|---|
| 100 | 18.4 | 1.3 | 23.5 | 43.2 | 372 | 7.8x | 4.4 | 66 |
| 120 | 18.4 | 1.3 | 27.0 | 46.7 | 425 | 8.9x | 5.2 | 79 |

## Per-step time, All vs Uniform (NVIDIA L40S)

One guidance step in isolation, mean ± std over 5 repeats.

| n_cond | All (s/step) | Uniform, n_sel/n_cond = 0.5 (s/step) | saving |
|---|---|---|---|
| 100 | 126.24 ± 0.87 | 72.99 ± 0.81 | 42% |
| 120 | 151.08 ± 0.47 | 87.39 ± 0.71 | 42% |
