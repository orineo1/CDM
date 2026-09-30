# paper_comparison

Runs every synthetic experiment of the paper on a SLURM cluster and compares the outputs with the numbers
printed in the paper (`ConditionalDistributionMatchingNeurIPS2026.pdf`). Nothing in the method or in
`pipeline/` is changed: the jobs call the existing scripts with the published flags.

| paper | what | job | output read by the comparison |
|---|---|---|---|
| Table 1 (p. 6) | TV after refinement, 9 scenarios x 4 methods x 5 runs | `01_grid.sh` (array 0-8), `02_grid_collect.sh` | `main_grid/outputs/runs/<s>/<METHOD>.json` |
| Table 4 (p. 19) | trained networks vs the analytic mixture | `05_model_quality.sh` (01 also writes `check.json`) | `model_quality/outputs/checks__<s>.json` |
| Table 5 (p. 21) | `‖x̂-x*‖`, relative L2; TV without refinement: plain / annealed / oracle | `03_ablations.sh` (array 0-8), `04_ablations_collect.sh` | `main_grid/outputs/runs/<s>/<METHOD>{_norefine,_annealed,_oracle}.json` |
| Table 6 (p. 22) | no-star target, % of 25 restarts at `x_bi` | `06_infeasible.sh` | `infeasible_target/outputs/arm_*.json` |
| Figure 1, App. B.8 | toy example, trajectories in the `x*` basin | `07_toy.sh` | `toy_example/outputs/fig1_hard_sim.json` |
| App. B.8 | twisted-SMC correction, TV to `Q_beta` | `08_smc.sh` (after 07) | `smc_correction/outputs/fig_smc_hard.json` |
| all of the above | the comparison | `09_compare.sh` | writes `paper_comparison/outputs/comparison.{md,csv}` |

`paper_numbers.json` holds the paper's values, transcribed from the PDF.

## Run

From the repository root, on the cluster's login node:

```bash
pip install -r requirements.txt && pip install -e .          # once, in the venv the jobs will use
PARTITION=<gpu partition> VENV=<path to venv> bash experiments/synthetic/paper_comparison/sbatch/submit_all.sh
```

Optional variables: `ACCOUNT`, `CPU_PARTITION` (for the collect / compare jobs), `CONDA_ENV` instead of `VENV`.
`submit_all.sh` chains the jobs with `--dependency`; logs go to `logs/`. A single job can be resubmitted on its
own (`sbatch --partition=... experiments/synthetic/paper_comparison/sbatch/0X_*.sh`): every step whose output
exists is skipped. When everything has finished (or at any point, on what exists so far):

```bash
python experiments/synthetic/paper_comparison/compare_to_paper.py
```

## Cost (NVIDIA L40S, from the paper)

- `01_grid`: per scenario, three trainings of 10,000 batches + 5 runs x (MLGD ~8 min, PMLGD ~8 min, the two
  -F methods ~50 s) = about 1.5-2.5 h per array task.
- `03_ablations`: per scenario, the four methods again without refinement, the two particle methods with
  annealing, and the two oracle runs: about 2-3 h per array task.
- `05`-`08`: under an hour each.

## Choices that differ from the per-experiment `run.sh` files

- The ablation arms (`03`) run on the grid's `runs/` directory, so they use the same trained networks as
  Table 1, as in the paper. `ablations/run.sh full` would write a separate `runs/` directory and retrain all
  27 networks.
- The toy example (`07`) runs 10 trajectories per method, as the paper reports (`toy_example/run.sh` uses 25),
  and only the `hard` setting, which is the published figure.
- Table 5's "annealed" column is TV **before** refinement (paper caption; the text's "0.80 -> 0.62 at (10,10)
  before refinement"). The annealed arm is run with refinement, as in `ablations/run.sh`, and the comparison
  reads the pre-refinement point that `find_xstar.py` stores (`refine.x_before`): per attempt, the trajectory
  with the lowest pre-refinement loss, which is what a run without refinement would have returned.
- `model_quality/run.sh` defaults to `RUNS_DIR=runs/table1v2`; `05` points it to the grid's runs.

## Reading the comparison

Each cell reports paper vs ours and a verdict: with `d = |ours - paper|` and
`s = sqrt((sd_ours² + sd_paper²) / 5)`, **OK** if `d <= 0.02` (the paper's rounding) or `d <= 2s`, **NEAR** if
`d <= 3s`, **DIFF** otherwise. Table 4 and the SMC TVs are single numbers and are judged by the ratio (OK
within x1.5, NEAR within x2). Table 6 and the toy counts are judged against binomial noise.

The seeds are fixed (42), but trained weights depend on the GPU, CUDA and torch versions, so the numbers
should agree within run-to-run noise, not to the last digit. The noisiest cells are (10, 1), which the paper
itself calls a search problem decided by the starting basin, and the particle methods at dim X = 10.

Checked: the TV and relative L2 are computed with `make_table1.scale_free_distances`, the function behind
the paper's tables, and equal `make_table1.py --collect` to the digit; the repository's reference
`model_quality/results/checks__*.json` reproduce Table 4 exactly for the four scenarios they cover.
