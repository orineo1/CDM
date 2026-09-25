# Smoke-tier run record — `experiments/gaussian_simulations/`

Whether each experiment's script chain starts and runs to completion. **Execution only.**
Nothing here is about numbers: the smoke tier uses tiny networks, tiny sample counts and
1–2 attempts, and its output is not a result.

Run 2026-09-25 on CPU (macOS) as `bash <dir>/run.sh smoke`, with `PYTHON` pointing at the
session venv and `CMP_SIM_DIR` set for the one experiment that needs it. `outputs/` was
cleared before each run, and has been cleared again since — the tree ships empty.

| experiment | runs | wall clock |
|---|---|---|
| `grid_model_training` | **yes** | 10 s |
| `legacy_training_times` | **yes** | 9 s |
| `table5_model_checks` | **yes** | 10 s |
| `infeasible_target` | **yes** | 13 s |
| `restarts_100_10x1` | **yes** | 17 s |
| `step_rule_sweep` | **yes** | 18 s |
| `ncond_ramp` | **yes** | 29 s |
| `swd_loss_control` | **yes** | 29 s |
| `retraining_long_wide` | **yes** | 28 s |
| `table1_main_grid` | **yes** | 47 s |
| `detached_bandwidth_grid` | **yes** | 49 s |
| `retrain_10x1_50k` | **yes** | 59 s |
| `table8_ablations` | **yes** | 54 s |
| `appendix_g_subsampling` | **yes** | 64 s |
| `fig1_toy_example` | **yes** | 64 s |
| `fig3_smc_correction` | **yes** | 75 s |
| `toy_model_checks` | **yes** | 67 s |
| `tv_sensitivity_calibration` | n/a — no producer; `run.sh` exits 2 | — |
| `prior_draw_baseline` | n/a — no producer; `run.sh` exits 2 | — |

**17 run, 0 fail, 2 have nothing to run.**

`table1_main_grid`, `infeasible_target` and `swd_loss_control` were re-run after commit
`4e0e603` (Item 12 changed `MMDLoss`) and after the environment's torch moved 2.12.0 →
2.14.0. All three still run.

## Failures encountered during development, and their cause

Both were wiring bugs in `run.sh`, fixed and re-run.

1. `detached_bandwidth_grid`, exit 2:

   ```
   make_table1.py: error: argument --extra: expected one argument
   ```

   `--extra "--bandwidth-grad"`. argparse refuses a value beginning with `-` unless it
   contains a space, so a single-token flag needs `--extra=--bandwidth-grad`.

2. `detached_bandwidth_grid`, exit 127 on the retry:

   ```
   run.sh: line 30: --tag: command not found
   ```

   A trailing `#` comment before the line-continuation backslash swallowed the backslash
   and split the command. The orphaned half ran with the default `--out` and wrote stray
   `results/table1*` files into the repo root; those were deleted. Comment moved above the
   command.

## Defect in `scripts/`, not fixed here

`fig_smc_correction.py` exits non-zero when given a single `--betas` value:

```
  File ".../scripts/fig_smc_correction.py", line 182, in main
    ax = axes[i]
         ~~~~^^^
TypeError: 'Axes' object is not subscriptable
```

`plt.subplots(1, len(args.betas))` returns a bare `Axes` rather than an array when
`len == 1`. `scripts/` belongs to another item, so this is reported rather than patched;
`fig3_smc_correction/run.sh` passes four betas in both tiers.
