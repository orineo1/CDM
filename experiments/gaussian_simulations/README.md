# `experiments/gaussian_simulations/`

One directory per Gaussian experiment in the verified inventory. Each holds
`README.md` (what it produces, the command, runtime, GPU), `run.sh` (smoke and full
tiers, every flag literal), `config/` where the experiment has parameters, `results/`
(reference numbers **imported** from a named GitHub commit) and `outputs/`
(gitignored, produced by a local run). The two are never mixed.

**`experiments/` invokes `scripts/`; it never duplicates it.** Every `run.sh` calls
`python scripts/<name>.py`. Where a paper experiment needed something the library could
not express, the library was changed -- see `CHANGES-TO-LIBRARY.md`.

**Basis.** `orineo1/conditional-matching-paper` is the specification: it says what each
experiment is, with which constants and seeds. Each directory's `PROVENANCE.md` names the
branch, commit and path it was read from, or records that no counterpart exists there.
The nine-scenario MLGD/MLGD-F/PMLGD/PMLGD-F grid, the particle methods and the twisted-SMC
correction have **no counterpart upstream** and are GlobalConditional-native.


| directory | paper artifact | producer | smoke |
|---|---|---|---|
| `table1_main_grid` | Table 1, Table 7 | `make_table1.py` | yes |
| `table8_ablations` | Table 8 | `make_table1.py` (no `_ablation` emitter) | yes |
| `table5_model_checks` | Table 5 | `check_models.py` | yes |
| `fig1_toy_example` | Figure 1 | `fig1_toy_example.py` | yes |
| `fig3_smc_correction` | Figure 3 | `fig_smc_correction.py` | yes |
| `infeasible_target` | Table 9, Figure 4 | `generate_data` + `train_models` + `find_xstar --loss l2` | yes |
| `appendix_g_subsampling` | App. G control (no float) | `control_subsample.py` | yes |
| `detached_bandwidth_grid` | App. B.6 prose | `make_table1.py --extra "--bandwidth-grad"` | yes |
| `ncond_ramp` | App. B.6 prose | `make_table1.py --extra "--n-cond-schedule linear ..."` | yes |
| `swd_loss_control` | App. B.6/B.7 prose | `make_table1.py --extra "--loss swd"` | yes |
| `step_rule_sweep` | App. B.6 prose | `make_table1.py --guidance-optim sgd\|adam` | yes |
| `retraining_long_wide` | App. B.6 prose | `make_table1.py --epochs / --nunits` | yes |
| `restarts_100_10x1` | App. B.7 prose | `make_table1.py --settings 10x1 --budget 100` | yes |
| `retrain_10x1_50k` | App. B.7 prose | `make_table1.py --epochs 50000 --force` | yes |
| `toy_model_checks` | App. B.8 prose | `check_models.py --setting` | yes |
| `grid_model_training` | App. B.4 prose | `generate_data.py` + `train_models.py` | yes |
| `legacy_training_times` | Table 10 | `generate_data.py --params-file` + `train_models.py` | yes |
| `tv_sensitivity_calibration` | App. B.7 prose | **none** | n/a, `run.sh` exits 2 |
| `prior_draw_baseline` | App. B.7 prose | **none** | n/a, `run.sh` exits 2 |

**Not covered, deliberately.** Tables 4 (`tab:sim_2d_arch`) and 6 (`tab:sim_2d_opt`) are
configuration tables with no run behind them (inventory SIM-08, SIM-09); their contents are
the arguments recorded in `legacy_training_times/config/`. Inventory SIM-25 (which
beta-sweep sampler produced the published Figure 1(d)) is an unresolved provenance
question, noted in `fig1_toy_example/README.md`, not a separate experiment.

## The documents at this level

- `SMOKE-RUNS.md` — whether each experiment's script chain runs, and the failures hit
  while wiring them up. Execution only.
- `MISSING-RESULTS.md` — a plain inventory of which artifacts are absent from the
  repository.
- `CHANGES-TO-LIBRARY.md` — what Item 9 changed outside `experiments/`.

Each experiment's `results/PUBLISHED.md` carries the paper's numbers for that experiment,
transcribed with their location in the source. Other files in `results/` are artifacts
imported from a named commit, listed in that directory's `PROVENANCE.md`. `outputs/` is
gitignored and ships empty.
