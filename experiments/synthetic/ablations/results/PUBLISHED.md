> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# Table 8 `tab:sim_ablation` (paper.txt:977 ff.)

TV of the returned x_hat, mean +- std over five runs. First five scenarios transcribed;
the table continues through all nine in the PDF.

| Setting | Method | no refinement | refined | annealed resampling | oracle f_phi |
|---|---|---|---|---|---|
| (1, 1)  | MLGD    | 0.01 +- 0.01 | 0.02 +- 0.01 | -- | 0.02 +- 0.01 |
| (1, 1)  | MLGD-F  | 0.08 +- 0.05 | 0.08 +- 0.01 | -- | -- |
| (1, 1)  | PMLGD   | 0.02 +- 0.02 | 0.02 +- 0.01 | 0.02 +- 0.01 | 0.02 +- 0.01 |
| (1, 1)  | PMLGD-F | 0.07 +- 0.01 | 0.08 +- 0.02 | 0.10 +- 0.03 | -- |
| (1, 5)  | MLGD    | 0.02 +- 0.01 | 0.02 +- 0.01 | -- | 0.02 +- 0.01 |
| (1, 5)  | MLGD-F  | 0.08 +- 0.05 | 0.08 +- 0.02 | -- | -- |
| (1, 5)  | PMLGD   | 0.03 +- 0.03 | 0.01 +- 0.01 | 0.03 +- 0.03 | 0.02 +- 0.01 |
| (1, 5)  | PMLGD-F | 0.07 +- 0.04 | 0.07 +- 0.03 | 0.07 +- 0.04 | -- |
| (1, 10) | MLGD    | 0.12 +- 0.08 | 0.07 +- 0.07 | -- | 0.07 +- 0.04 |
| (1, 10) | MLGD-F  | 0.42 +- 0.08 | 0.38 +- 0.05 | -- | -- |
| (1, 10) | PMLGD   | 0.12 +- 0.13 | 0.06 +- 0.05 | 0.12 +- 0.13 | 0.06 +- 0.05 |
| (1, 10) | PMLGD-F | 0.41 +- 0.06 | 0.37 +- 0.07 | 0.41 +- 0.06 | -- |
| (5, 1)  | MLGD    | 0.10 +- 0.06 | 0.08 +- 0.01 | -- | 0.17 +- 0.17 |
| (5, 1)  | MLGD-F  | 0.22 +- 0.11 | 0.25 +- 0.09 | -- | -- |
| (5, 1)  | PMLGD   | 0.19 +- 0.09 | 0.17 +- 0.11 | 0.12 +- 0.03 | 0.51 +- 0.02 |
| (5, 1)  | PMLGD-F | 0.30 +- 0.10 | 0.27 +- 0.09 | 0.27 +- 0.06 | -- |
| (5, 5)  | MLGD    | 0.28 +- 0.09 | 0.06 +- 0.02 | -- | 0.24 +- 0.15 |
| (5, 5)  | MLGD-F  | 0.49 +- 0.03 | 0.50 +- 0.05 | -- | -- |
| (5, 5)  | PMLGD   | 0.43 +- 0.28 | 0.17 +- 0.19 | 0.23 +- 0.07 | 0.63 +- 0.25 |

The "refined" column is Table 1 reprinted.

Transcribed from `paper.txt`.
