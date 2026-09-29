# main_grid

**Produces:** Table 1 (`tab:sim_main`) and the full-grid appendix table (`tab:sim_full`):
the 9 scenarios `(dim X, dim Y) in {1, 5, 10}^2` x 4 algorithms (MLGD, MLGD-F, PMLGD, PMLGD-F),
scored by the exact total variation between `P(Y | X = x_hat)` and the target.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/make_table1.py`, which drives `generate_data.py`, `train_models.py`, `check_models.py` and `find_xstar.py` per scenario |
| Full configuration | 9 scenarios x 4 methods x 5 runs; 10 trajectories per run, `n_cond = 250`, Adam guidance step 0.05, 100 refinement steps (lr 0.02, 1,000 samples); `config/paper_config.txt` records it |
| GPU | yes. About 8 min per MLGD run and 50 s per MLGD-F run on an L40S, plus training three networks per scenario (10,000 batches each) |
| Reduced configuration (`smoke`) | CPU, a few minutes: one scenario, tiny networks, 2 trajectories, 1 run |

## Run

```bash
bash experiments/synthetic/main_grid/run.sh smoke        # reduced configuration, checks the code path
bash experiments/synthetic/main_grid/run.sh full         # the paper's configuration, all 9 scenarios on one GPU (~1 day)
SETTINGS="5x5" bash experiments/synthetic/main_grid/run.sh full   # one scenario
```

On a SLURM cluster, one array task per scenario, from the repository root:

```bash
sbatch experiments/synthetic/slurm/slurm_table1.sh
```

Every step whose output already exists is skipped, so a killed run can be resubmitted.
Outputs go to `outputs/` (gitignored): `outputs/runs/<scenario>/` holds data, networks, the
model check (`check.json`, `check.png`) and one JSON per method; `outputs/table1.{json,md,tex}`
is the table. `model_quality` and `ablations` read the networks trained here.

## Committed results (`results/`)

Written by `../export_results.py` from a finished full run:

| file | contents |
|---|---|
| `table1.md`, `table1.csv`, `table1.json` | the table: one row per scenario and method (TV mean ± std over the 5 runs, relative L2, ‖x̂ − x*‖ in sd units, time per run, trajectories with TV < 0.5, in the x* basin, resamplings); the JSON also holds every attempt |
| `cells.csv` | one row per scenario, method and run: TV, relative L2, ‖x̂ − x*‖, final loss, time |
| `table1_appendix.md` | per-scenario model checks and search diagnostics |
| `table1*.tex` | the LaTeX rows of Tables 1, 5 and 7 |
| `check_<scenario>.png` | the networks against the analytic mixture, per scenario |
| `run_info.txt` | date, commit, GPU, torch version |

`python experiments/synthetic/pipeline/compare_to_paper.py --results` prints these next
to the paper's numbers.

`table1_v3.md` is an earlier grid output (from `orineo1/GlobalConditional` `master`
`9ddf570`, before this layout) kept until a full run replaces it; it has no PMLGD rows and
differs from the paper in several cells.

## Paper numbers

Table 1 `tab:sim_main`: TV(P(Y | X = x_hat), G), mean ± std over five runs.

| (dim X, dim Y) | MLGD | MLGD-F | PMLGD | PMLGD-F |
|---|---|---|---|---|
| (1, 1)   | 0.02 ± 0.01 | 0.08 ± 0.01 | 0.02 ± 0.01 | 0.08 ± 0.02 |
| (1, 5)   | 0.02 ± 0.01 | 0.08 ± 0.02 | 0.01 ± 0.01 | 0.07 ± 0.03 |
| (1, 10)  | 0.07 ± 0.07 | 0.38 ± 0.05 | 0.06 ± 0.05 | 0.37 ± 0.07 |
| (5, 1)   | 0.08 ± 0.01 | 0.25 ± 0.09 | 0.17 ± 0.11 | 0.27 ± 0.09 |
| (5, 5)   | 0.06 ± 0.02 | 0.50 ± 0.05 | 0.17 ± 0.19 | 0.48 ± 0.03 |
| (5, 10)  | 0.14 ± 0.03 | 0.41 ± 0.04 | 0.16 ± 0.06 | 0.45 ± 0.03 |
| (10, 1)  | 0.49 ± 0.08 | 0.56 ± 0.03 | 0.99 ± 0.02 | 0.78 ± 0.22 |
| (10, 5)  | 0.13 ± 0.04 | 0.30 ± 0.01 | 0.18 ± 0.12 | 0.30 ± 0.03 |
| (10, 10) | 0.20 ± 0.05 | 0.92 ± 0.02 | 0.43 ± 0.13 | 0.81 ± 0.12 |

Timing in the caption: ~8 min per run with the conditional-diffusion sampler, ~50 s with
the consistency sampler, NVIDIA L40S. Table 7 `tab:sim_full` is the same runs with three
more columns (‖x̂ − x*‖ in sd units, relative L2, seconds per run). The same numbers are
in `pipeline/paper_numbers.py`.

The experiment is specified by the paper text; `orineo1/conditional-matching-paper` has no
counterpart (no particle population, no resampling, no refinement).
