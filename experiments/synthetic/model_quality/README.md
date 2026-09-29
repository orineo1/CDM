# model_quality

**Produces:** Table 5 `tab:model_quality`: the trained networks of each scenario against the
analytic mixture (score network, prior samples, the two conditional samplers at x* and at
random x), independently of the x* search.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/check_models.py` per scenario, then `make_table5.py` (smoke also trains tiny networks with `generate_data.py` and `train_models.py`) |
| Full configuration | one check per scenario over the 27 networks trained by `../main_grid` (`RUNS_DIR`, default `../main_grid/outputs/runs`) |
| GPU | yes; minutes per scenario |
| Reduced configuration (`smoke`) | CPU, ~1 min |

## Run

```bash
bash experiments/synthetic/model_quality/run.sh smoke
bash experiments/synthetic/model_quality/run.sh full            # after main_grid/run.sh full
sbatch experiments/synthetic/slurm/slurm_run.sh model_quality   # the same as one GPU job
```

Outputs go to `outputs/` (gitignored): `checks__<scenario>.{json,png}` and `table5.{md,csv}`.

## Committed results (`results/`)

Written by `../export_results.py`: `table5.md`, `table5.csv`, the nine `checks__*.json` and
`.png`, `run_info.txt`. Until a full run replaces them, `results/` holds the four
`checks__{x5_y5,x5_y10,x10_y5,x10_y10}.json` of the run behind the paper (imported from
`orineo1/GlobalConditional` `master` `9ddf570`); `make_table5.py results/` reproduces the
paper's four rows from them exactly.

## Paper numbers

Table 5 `tab:model_quality`:

| Setting | score net Fisher (rel.) | prior SWD | cond DDPM at x* | cond DDPM random x | iCT at x* | iCT random x |
|---|---|---|---|---|---|---|
| dim X = 1, dim Y = 1   | 0.001 | 0.020 | 0.012 | 0.166 | 0.041 | 1.234 |
| dim X = 1, dim Y = 5   | 0.005 | 0.036 | 0.041 | 0.043 | 0.166 | 0.161 |
| dim X = 1, dim Y = 10  | 0.003 | 0.036 | 0.032 | 0.041 | 0.105 | 0.097 |
| dim X = 5, dim Y = 1   | 0.006 | 0.026 | 0.082 | 0.193 | 0.112 | 0.882 |
| dim X = 5, dim Y = 5   | 0.007 | 0.030 | 0.048 | 0.097 | 0.213 | 0.258 |
| dim X = 5, dim Y = 10  | 0.010 | 0.033 | 0.040 | 0.088 | 0.093 | 0.164 |
| dim X = 10, dim Y = 1  | 0.011 | 0.036 | 0.990 | 0.766 | 0.563 | 1.673 |
| dim X = 10, dim Y = 5  | 0.014 | 0.029 | 0.038 | 0.181 | 0.165 | 0.406 |
| dim X = 10, dim Y = 10 | 0.017 | 0.032 | 0.057 | 0.142 | 0.279 | 0.401 |

"score net Fisher" is the worst relative Fisher divergence over t > 1; "at x*" is the W1
(dim Y = 1) or SWD distance of the sampler's draws at x* to the analytic conditional.
