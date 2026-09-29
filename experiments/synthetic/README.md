# `experiments/synthetic/`

One directory per Gaussian experiment in the paper. Each holds `README.md` (what it
produces, the command, runtime, GPU), `run.sh` (a reduced `smoke` configuration and the
full one, every flag literal), `config/` where the experiment has parameters, `results/`
(reference numbers **imported** from a named GitHub commit) and `outputs/` (gitignored,
written by a local run). The two are never mixed.

**Every `run.sh` calls the scripts in `pipeline/`**; the experiment directories hold
no code of their own. `configs/mog/` holds the paper's fixed mixture settings and
`slurm/` the cluster templates.

**Basis.** `orineo1/conditional-matching-paper` is the specification: it says what each
experiment is, with which constants and seeds. Each directory's `PROVENANCE.md` names the
branch, commit and path it was read from, or records that no counterpart exists there.
The nine-scenario MLGD/MLGD-F/PMLGD/PMLGD-F grid, the particle methods and the twisted-SMC
correction have **no counterpart upstream** and are GlobalConditional-native.

| directory | paper artifact (ICLR numbering; AISTATS will renumber) | producer |
|---|---|---|
| `main_grid` | Table 1, Table 7 | `make_table1.py` |
| `ablations` | Table 8 | `make_table1.py` (no `_ablation` emitter) |
| `model_quality` | Table 5 | `check_models.py` |
| `toy_example` | Figure 1 | `fig1_toy_example.py` |
| `smc_correction` | Figure 3 | `fig_smc_correction.py` |
| `infeasible_target` | Table 9, Figure 4 | `generate_data` + `train_models` + `find_xstar --loss l2` |
| `subsampling_control` | App. G control (no float) | `control_subsample.py` |
| `detached_bandwidth_grid` | App. B.6 prose | `make_table1.py --extra "--bandwidth-grad"` |
| `ncond_ramp` | App. B.6 prose | `make_table1.py --extra "--n-cond-schedule linear ..."` |
| `swd_loss_control` | App. B.6/B.7 prose | `make_table1.py --extra "--loss swd"` |
| `step_rule_sweep` | App. B.6 prose | `make_table1.py --guidance-optim sgd\|adam` |
| `retraining_long_wide` | App. B.6 prose | `make_table1.py --epochs / --nunits` |
| `restarts_100` | App. B.7 prose | `make_table1.py --settings 10x1 --budget 100` |
| `retrain_50k` | App. B.7 prose | `make_table1.py --epochs 50000 --force` |
| `toy_model_checks` | App. B.8 prose | `check_models.py --setting` |
| `grid_model_training` | App. B.4 prose | `generate_data.py` + `train_models.py` |
| `training_times` | Table 10 | `generate_data.py --params-file` + `train_models.py` |
| `tv_sensitivity_calibration` | App. B.7 prose | **none** — `run.sh` exits 2 |
| `prior_draw_baseline` | App. B.7 prose | **none** — `run.sh` exits 2 |

**Not covered.** Tables 4 (`tab:sim_2d_arch`) and 6 (`tab:sim_2d_opt`) are configuration
tables with no run behind them; their contents are the arguments recorded in
`training_times/config/`. Which beta-sweep sampler produced the published
Figure 1(d) is an open provenance question, noted in `toy_example/README.md`.

## Running an experiment

```
bash <dir>/run.sh smoke   # a reduced configuration: tiny networks and sample counts, CPU
bash <dir>/run.sh full    # the paper's configuration; most arms want a GPU
```

The reduced configuration exercises the script chain end to end; its output is not a
result. Per-experiment runtimes and GPU requirements are in each `README.md`.

## The documents at this level

- `REFERENCE-ARTIFACTS.md` — which reference artifacts for these experiments are in the
  repository and which are not.

For the experiments behind a table or figure of the paper (`main_grid`, `ablations`,
`model_quality`, `infeasible_target`, `training_times`, `toy_example`, `smc_correction`) the
README carries the paper's numbers, `results/` holds the committed output of the last full
run (written by `export_results.py`), and
`python experiments/synthetic/pipeline/compare_to_paper.py --results` prints the two side
by side. The other directories back appendix prose only; their `results/PUBLISHED.md`
carries the transcribed numbers and `PROVENANCE.md` names imported files. `outputs/` is
gitignored and ships empty.

## Reproducing the tables and figures on a SLURM cluster

From the repository root, each as one GPU job (`slurm/slurm_run.sh <experiment>` runs that
experiment's `run.sh full`); `main_grid` is an array job, one task per scenario:

```bash
sbatch experiments/synthetic/slurm/slurm_table1.sh                     # Tables 1, 7 (9 tasks, ~2 h each)
sbatch experiments/synthetic/slurm/slurm_run.sh smc_correction         # Figure 3
sbatch experiments/synthetic/slurm/slurm_run.sh infeasible_target      # Table 9, Figure 4
sbatch --time=24:00:00 experiments/synthetic/slurm/slurm_run.sh training_times   # Table 10
# after smc_correction has finished:
sbatch experiments/synthetic/slurm/slurm_run.sh toy_example            # Figure 1
# after all nine main_grid tasks have finished:
sbatch experiments/synthetic/slurm/slurm_run.sh model_quality          # Table 5
sbatch --time=48:00:00 experiments/synthetic/slurm/slurm_run.sh ablations   # Table 8
```

Then `python experiments/synthetic/export_results.py` and `compare_to_paper.py --results`.
