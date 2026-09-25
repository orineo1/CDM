> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# Table 1 `tab:sim_main` (paper.txt:282-290)

TV(P(Y | X = x_hat), G), mean +- std over five runs.

| (dim X, dim Y) | MLGD | MLGD-F | PMLGD | PMLGD-F |
|---|---|---|---|---|
| (1, 1)   | 0.02 +- 0.01 | 0.08 +- 0.01 | 0.02 +- 0.01 | 0.08 +- 0.02 |
| (1, 5)   | 0.02 +- 0.01 | 0.08 +- 0.02 | 0.01 +- 0.01 | 0.07 +- 0.03 |
| (1, 10)  | 0.07 +- 0.07 | 0.38 +- 0.05 | 0.06 +- 0.05 | 0.37 +- 0.07 |
| (5, 1)   | 0.08 +- 0.01 | 0.25 +- 0.09 | 0.17 +- 0.11 | 0.27 +- 0.09 |
| (5, 5)   | 0.06 +- 0.02 | 0.50 +- 0.05 | 0.17 +- 0.19 | 0.48 +- 0.03 |
| (5, 10)  | 0.14 +- 0.03 | 0.41 +- 0.04 | 0.16 +- 0.06 | 0.45 +- 0.03 |
| (10, 1)  | 0.49 +- 0.08 | 0.56 +- 0.03 | 0.99 +- 0.02 | 0.78 +- 0.22 |
| (10, 5)  | 0.13 +- 0.04 | 0.30 +- 0.01 | 0.18 +- 0.12 | 0.30 +- 0.03 |
| (10, 10) | 0.20 +- 0.05 | 0.92 +- 0.02 | 0.43 +- 0.13 | 0.81 +- 0.12 |

Caption timing (paper.txt:275-276): ~8 min/run with the conditional-diffusion sampler,
~50 s/run with the consistency sampler, NVIDIA L40S.

Table 7 `tab:sim_full` (paper.txt:918) is the same runs with three more columns
(||x_hat - x*|| in sd units, relative L2, seconds per run).

Transcribed from `paper.txt`.
