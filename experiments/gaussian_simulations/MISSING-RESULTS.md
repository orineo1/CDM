# Missing artifacts — `experiments/gaussian_simulations/`

A plain inventory of what is absent from the repository. Checked 2026-09-25 against
`orineo1/GlobalConditional` `master` `9ddf570` and `orineo1/conditional-matching-paper`
(all 70 remote branches). No inference is drawn here beyond what is or is not on disk.

## What `GlobalConditional`'s `results/` contains at `9ddf570`

Twelve `checks__*.json`; `fig_smc_hard.json`; `fig_smc_hard_cpu.json`;
`legacy_pasted_2d.json`; `runs.jsonl`; `smc_sweep/`; and six `table1_*.md`
(`10x1_r100`, `detach`, `long`, `v3`, `w1_5x1`, `wide`).

## Absent files

| file | referenced by |
|---|---|
| `results/table1.json` | paper, `main.tex:1047`; `make_table1.py` would write it |
| `results/table1_appendix.md` | `make_table1.py` docstring (per-scenario checks, basin hit rates, resamplings) |
| `iclr2027/tables/table1_compact_rows.tex` | paper, Table 1 |
| `iclr2027/tables/table1_rows.tex` | paper, Table 7 |
| `iclr2027/tables/table1_models_rows.tex` | paper, Table 5 |
| `iclr2027/tables/table1_ablation_rows.tex` | paper, Table 8 |

None of the four `.tex` fragments is present anywhere on this machine.

## Absent per experiment

| experiment | generating script | reference artifact in repo |
|---|---|---|
| `table1_main_grid` | `make_table1.py` | none for Tables 1/7; `table1_v3.md` is a grid output at `9ddf570` |
| `table5_model_checks` | `check_models.py` | `checks__*.json` for 4 of the 9 scenarios (x5_y5, x5_y10, x10_y5, x10_y10) |
| `table8_ablations` | `make_table1.py` runs the four arms; it has no `_ablation` emitter | none |
| `infeasible_target` | `generate_data.py` + `train_models.py` + `find_xstar.py --loss l2` | none in this repo |
| `fig1_toy_example` | `fig1_toy_example.py` | no `fig1_*_sim.json` or `_sweep.npz` |
| `fig3_smc_correction` | `fig_smc_correction.py` | `fig_smc_hard.json`, `fig_smc_hard_cpu.json`, `smc_sweep/*` |
| `legacy_training_times` | `generate_data.py --params-file` + `train_models.py`; neither emits a timing | none |
| `detached_bandwidth_grid` | `make_table1.py` | `table1_detach.md` (26 rows: MLGD/MLGD-F × 9, PMLGD/PMLGD-F × 4) |
| `ncond_ramp` | `make_table1.py` | none |
| `swd_loss_control` | `make_table1.py` | `table1_w1_5x1.md` for (5,1); none for (10,1) |
| `step_rule_sweep` | `make_table1.py` | none |
| `retraining_long_wide` | `make_table1.py` | `table1_long.md`, `table1_wide.md`, eight `checks__{long,wide}_*` |
| `restarts_100_10x1` | `make_table1.py` | `table1_10x1_r100.md` |
| `retrain_10x1_50k` | `make_table1.py` | none; `checks__x10_y1.json` exists in no variant |
| `toy_model_checks` | `check_models.py --setting` | none for the toy setting |
| `grid_model_training` | `generate_data.py` + `train_models.py` | checkpoints are on the HuggingFace repo the paper cites as `\hfsim`, not in git |
| `appendix_g_subsampling` | `control_subsample.py` | none |
| `tv_sensitivity_calibration` | **no script in either repository** | none |
| `prior_draw_baseline` | **no script in either repository** | none |

## Notes on locating producers

- `infeasible_target`: the scenario constants and an implementation of the same experiment
  are at `orineo1/conditional-matching-paper` `main` `fc4389d`,
  `simulations/notebooks/Exp_nonstar_mmd_target_ablation.ipynb`. `config/nonstar_params.txt`
  transcribes the constants. INVENTORY SIM-07 records this row as having no producer in
  either repository; that search did not cover this branch.
- `ncond_ramp` and `step_rule_sweep` are recorded in the inventory as script-MISSING. The
  flags exist (`--n-cond-schedule linear`, `--n-cond-min`, `--step-decay`,
  `--guidance-optim adam`, `--step-size`); only a driver loop was absent, and the
  relevant `run.sh` is now that loop.
- `tv_sensitivity_calibration` and `prior_draw_baseline`: `git grep` over `simulations/`
  on all 70 branches for `sensitivity`, `delta u`, `prior_draw` and the quoted values
  returns zero hits.
- `appendix_g_subsampling`: `conditional-matching-paper` `main` `fc4389d` additionally
  carries `simulations/scripts/run_backsel_witness_sweep.py`,
  `backsel_state_gradient_variance.py` and `src/witness_utils.py` on this same 10D setting.
