# Reference artifacts — `experiments/synthetic/`

Which reference artifacts for these experiments are in the repository, and which are not.
The sources are `orineo1/GlobalConditional` `master` `9ddf570` and
`orineo1/conditional-matching-paper`.

## Present: `GlobalConditional`'s `results/` at `9ddf570`

Twelve `checks__*.json`; `fig_smc_hard.json`; `fig_smc_hard_cpu.json`;
`legacy_pasted_2d.json`; `runs.jsonl`; `smc_sweep/`; and six `table1_*.md`
(`10x1_r100`, `detach`, `long`, `v3`, `w1_5x1`, `wide`).

## Absent, referenced elsewhere

| file | referenced by |
|---|---|
| `results/table1.json` | paper, `main.tex:1047`; `make_table1.py` would write it |
| `results/table1_appendix.md` | `make_table1.py` docstring (per-scenario checks, basin hit rates, resamplings) |
| `iclr2027/tables/table1_compact_rows.tex` | paper, Table 1 |
| `iclr2027/tables/table1_rows.tex` | paper, Table 7 |
| `iclr2027/tables/table1_models_rows.tex` | paper, Table 5 |
| `iclr2027/tables/table1_ablation_rows.tex` | paper, Table 8 |

None of the four `.tex` fragments is in either code repository.

## Per experiment

| experiment | generating script | reference artifact in repo |
|---|---|---|
| `main_grid` | `make_table1.py` | none for Tables 1/7; `table1_v3.md` is a grid output at `9ddf570` |
| `model_quality` | `check_models.py` | `checks__*.json` for 4 of the 9 scenarios (x5_y5, x5_y10, x10_y5, x10_y10) |
| `ablations` | `make_table1.py` runs the four arms; it has no `_ablation` emitter | none |
| `infeasible_target` | `generate_data.py` + `train_models.py` + `find_xstar.py --loss l2` | none in this repo |
| `toy_example` | `fig1_toy_example.py` | no `fig1_*_sim.json` or `_sweep.npz` |
| `smc_correction` | `fig_smc_correction.py` | `fig_smc_hard.json`, `fig_smc_hard_cpu.json`, `smc_sweep/*` |
| `training_times` | `generate_data.py --params-file` + `train_models.py`; neither emits a timing | none |
| `detached_bandwidth_grid` | `make_table1.py` | `table1_detach.md` (26 rows: MLGD/MLGD-F × 9, PMLGD/PMLGD-F × 4) |
| `ncond_ramp` | `make_table1.py` | none |
| `swd_loss_control` | `make_table1.py` | `table1_w1_5x1.md` for (5,1); none for (10,1) |
| `step_rule_sweep` | `make_table1.py` | none |
| `retraining_long_wide` | `make_table1.py` | `table1_long.md`, `table1_wide.md`, eight `checks__{long,wide}_*` |
| `restarts_100` | `make_table1.py` | `table1_10x1_r100.md` |
| `retrain_50k` | `make_table1.py` | none; `checks__x10_y1.json` exists in no variant |
| `toy_model_checks` | `check_models.py --setting` | none for the toy setting |
| `grid_model_training` | `generate_data.py` + `train_models.py` | checkpoints are on the HuggingFace repo the paper cites as `\hfsim`, not in git |
| `subsampling_control` | `control_subsample.py` | none |
| `tv_sensitivity_calibration` | **no script in either repository** | none |
| `prior_draw_baseline` | **no script in either repository** | none |

## Notes

- `infeasible_target`: the scenario constants and an implementation of the same experiment
  are at `orineo1/conditional-matching-paper` `main` `fc4389d`,
  `simulations/notebooks/Exp_nonstar_mmd_target_ablation.ipynb`. `config/nonstar_params.txt`
  transcribes the constants.
- `ncond_ramp` and `step_rule_sweep` have no upstream driver script. The flags they need
  exist (`--n-cond-schedule linear`, `--n-cond-min`, `--step-decay`,
  `--guidance-optim adam`, `--step-size`); the relevant `run.sh` is the driver loop.
- `tv_sensitivity_calibration` and `prior_draw_baseline` have no producing script in
  either repository; their numbers are transcribed from the paper only.
- `subsampling_control`: `conditional-matching-paper` `main` `fc4389d` additionally
  carries `simulations/scripts/run_backsel_witness_sweep.py`,
  `backsel_state_gradient_variance.py` and `src/witness_utils.py` on this same 10D setting.
