# ablations

**Produces:** the appendix table `tab:sim_full` (formerly Table 8 `tab:sim_ablation`): per
setting and method, ‖x̂ − x*‖ and the relative L² after refinement (from `../main_grid`), and
the TV without refinement of three arms: the plain search, annealed resampling (particle
methods only) and the oracle `f_phi` (the exact conditional as sampler). `config/arms.txt`
lists the flag that distinguishes each arm.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/make_table1.py` for the three extra arms, then `make_table8.py` joins them with `../main_grid/outputs/table1.json` |
| Full configuration | about three grids: 9 scenarios x (4 + 2 + 2 methods) x 5 runs |
| GPU | yes; the networks are those trained by `../main_grid` (`--runs-dir` shared), so run `main_grid` first |
| Reduced configuration (`smoke`) | CPU, ~2 min |

## Run

```bash
bash experiments/synthetic/ablations/run.sh smoke
bash experiments/synthetic/ablations/run.sh full          # after main_grid/run.sh full
sbatch experiments/synthetic/slurm/slurm_run.sh ablations # the same as one GPU job
```

Outputs go to `outputs/` (gitignored): `table8_{norefine,annealed,oracle}.*` per arm and
`table8.{md,csv}`, the joined table.

## Committed results (`results/`)

Written by `../export_results.py`: `table8.md`, `table8.csv` (one row per scenario and
method, mean and std per arm), the three per-arm `table8_*.json` / `.md`, `run_info.txt`.

## Paper numbers

From the earlier version's Table 8 `tab:sim_ablation` (TV of the returned x̂, mean ± std over
five runs; the first five scenarios were transcribed, the PDF has all nine; "refined" is
Table 1). The current `tab:sim_full` adds ‖x̂ − x*‖ and the relative L², not transcribed here.

| Setting | Method | no refinement | refined | annealed resampling | oracle f_phi |
|---|---|---|---|---|---|
| (1, 1)  | MLGD    | 0.01 ± 0.01 | 0.02 ± 0.01 | -- | 0.02 ± 0.01 |
| (1, 1)  | MLGD-F  | 0.08 ± 0.05 | 0.08 ± 0.01 | -- | -- |
| (1, 1)  | PMLGD   | 0.02 ± 0.02 | 0.02 ± 0.01 | 0.02 ± 0.01 | 0.02 ± 0.01 |
| (1, 1)  | PMLGD-F | 0.07 ± 0.01 | 0.08 ± 0.02 | 0.10 ± 0.03 | -- |
| (1, 5)  | MLGD    | 0.02 ± 0.01 | 0.02 ± 0.01 | -- | 0.02 ± 0.01 |
| (1, 5)  | MLGD-F  | 0.08 ± 0.05 | 0.08 ± 0.02 | -- | -- |
| (1, 5)  | PMLGD   | 0.03 ± 0.03 | 0.01 ± 0.01 | 0.03 ± 0.03 | 0.02 ± 0.01 |
| (1, 5)  | PMLGD-F | 0.07 ± 0.04 | 0.07 ± 0.03 | 0.07 ± 0.04 | -- |
| (1, 10) | MLGD    | 0.12 ± 0.08 | 0.07 ± 0.07 | -- | 0.07 ± 0.04 |
| (1, 10) | MLGD-F  | 0.42 ± 0.08 | 0.38 ± 0.05 | -- | -- |
| (1, 10) | PMLGD   | 0.12 ± 0.13 | 0.06 ± 0.05 | 0.12 ± 0.13 | 0.06 ± 0.05 |
| (1, 10) | PMLGD-F | 0.41 ± 0.06 | 0.37 ± 0.07 | 0.41 ± 0.06 | -- |
| (5, 1)  | MLGD    | 0.10 ± 0.06 | 0.08 ± 0.01 | -- | 0.17 ± 0.17 |
| (5, 1)  | MLGD-F  | 0.22 ± 0.11 | 0.25 ± 0.09 | -- | -- |
| (5, 1)  | PMLGD   | 0.19 ± 0.09 | 0.17 ± 0.11 | 0.12 ± 0.03 | 0.51 ± 0.02 |
| (5, 1)  | PMLGD-F | 0.30 ± 0.10 | 0.27 ± 0.09 | 0.27 ± 0.06 | -- |
| (5, 5)  | MLGD    | 0.28 ± 0.09 | 0.06 ± 0.02 | -- | 0.24 ± 0.15 |
| (5, 5)  | MLGD-F  | 0.49 ± 0.03 | 0.50 ± 0.05 | -- | -- |
| (5, 5)  | PMLGD   | 0.43 ± 0.28 | 0.17 ± 0.19 | 0.23 ± 0.07 | 0.63 ± 0.25 |

Specified by the paper text; no counterpart in `orineo1/conditional-matching-paper`.
