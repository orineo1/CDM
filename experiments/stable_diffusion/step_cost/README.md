# step_cost

Memory and time of one MLGD-F guidance step, measured in isolation on random latents and target vectors (the
content does not change the cost), with the settings of `../four_targets/config/balanced.env`:

1. **Turbo vs Base** (All selection): peak memory split into models + fixed part + the n_cond samples, and the
   step time; measured for SDXL Turbo (K = 2 steps) and projected for SDXL Base (K* = 30) by scaling the
   per-sample part by K*/K (same architecture). A full run is 125 guided steps.
2. **All vs Uniform** selection (n_sel/n_cond = 0.5): time per step, mean ± std over 5 repeats, n_cond = 100 and 120.

```bash
bash experiments/stable_diffusion/step_cost/run.sh                     # GPU, about an hour
bash experiments/stable_diffusion/step_cost/run.sh --num_variations 100 --rules all --n_repeats 1
bash experiments/stable_diffusion/step_cost/run.sh help                # every flag
```

A configuration that runs out of memory is recorded as OOM. Fast (non-deterministic) kernels are the default;
`--deterministic` uses the optimization runs' kernels. Output: `outputs/<GPU>_<id>/` (`step_cost.md`,
`step_cost.json`, `run_info.txt`); `../export_results.py` copies it into `results/`. The committed run is
`results/L40S/` (NVIDIA L40S). The run time of a full optimization run needs no step_cost run: each run records it
in its `metrics.json` (`../four_targets/README.md`, Run time).
