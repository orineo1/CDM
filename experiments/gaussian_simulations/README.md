# `experiments/gaussian_simulations/`

One directory per Gaussian experiment in the paper. Each holds `README.md` (what it
produces, the command, runtime, GPU), `run.sh` (a reduced `smoke` configuration and the
full one, every flag literal), `config/` where the experiment has parameters, `results/`
(reference numbers **imported** from a named GitHub commit) and `outputs/` (gitignored,
written by a local run). The two are never mixed.

**`experiments/` invokes `scripts/`; it never duplicates it.** Every `run.sh` calls
`python scripts/<name>.py`.

**Basis.** `orineo1/conditional-matching-paper` is the specification: it says what each
experiment is, with which constants and seeds. Each directory's `PROVENANCE.md` names the
branch, commit and path it was read from, or records that no counterpart exists there.
The nine-scenario MLGD/MLGD-F/PMLGD/PMLGD-F grid, the particle methods and the twisted-SMC
correction have **no counterpart upstream** and are GlobalConditional-native.

| directory | paper artifact | producer |
|---|---|---|
| `table1_main_grid` | Table 1, Table 7 | `make_table1.py` |
| `table8_ablations` | Table 8 | `make_table1.py` (no `_ablation` emitter) |
| `table5_model_checks` | Table 5 | `check_models.py` |
| `fig1_toy_example` | Figure 1 | `fig1_toy_example.py` |
| `fig3_smc_correction` | Figure 3 | `fig_smc_correction.py` |
| `infeasible_target` | Table 9, Figure 4 | `generate_data` + `train_models` + `find_xstar --loss l2` |
| `appendix_g_subsampling` | App. G control (no float) | `control_subsample.py` |
| `detached_bandwidth_grid` | App. B.6 prose | `make_table1.py --extra "--bandwidth-grad"` |
| `ncond_ramp` | App. B.6 prose | `make_table1.py --extra "--n-cond-schedule linear ..."` |
| `swd_loss_control` | App. B.6/B.7 prose | `make_table1.py --extra "--loss swd"` |
| `step_rule_sweep` | App. B.6 prose | `make_table1.py --guidance-optim sgd\|adam` |
| `retraining_long_wide` | App. B.6 prose | `make_table1.py --epochs / --nunits` |
| `restarts_100_10x1` | App. B.7 prose | `make_table1.py --settings 10x1 --budget 100` |
| `retrain_10x1_50k` | App. B.7 prose | `make_table1.py --epochs 50000 --force` |
| `toy_model_checks` | App. B.8 prose | `check_models.py --setting` |
| `grid_model_training` | App. B.4 prose | `generate_data.py` + `train_models.py` |
| `legacy_training_times` | Table 10 | `generate_data.py --params-file` + `train_models.py` |
| `tv_sensitivity_calibration` | App. B.7 prose | **none** — `run.sh` exits 2 |
| `prior_draw_baseline` | App. B.7 prose | **none** — `run.sh` exits 2 |

**Not covered.** Tables 4 (`tab:sim_2d_arch`) and 6 (`tab:sim_2d_opt`) are configuration
tables with no run behind them; their contents are the arguments recorded in
`legacy_training_times/config/`. Which beta-sweep sampler produced the published
Figure 1(d) is an open provenance question, noted in `fig1_toy_example/README.md`.

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

Each experiment's `results/PUBLISHED.md` carries the paper's numbers for that experiment,
transcribed with their location in the source. Other files in `results/` are artifacts
imported from a named commit, listed in that directory's `PROVENANCE.md`. `outputs/` is
gitignored and ships empty.
