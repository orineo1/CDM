# Search × gradient estimator (2 × 2): independent rerun, all four columns complete

A full rerun, from scratch, of the four corners of `search_x_estimator.md`: retrained samplers, MLGD\*,
GB-FESO (MMD) and the two missing corners. It ran on the HUJI cluster, `salmon` partition, NVIDIA L40S, from
`/sci/labs/orzuk/ori_m/CDM_rerun` at `master` `c1d7af1`. The samplers are diffusion, 256 units, 4 blocks,
100,000 batches, seed 7. The data are the Table 1 data (`generate_data`, seed 42).

TV(P(Y | x̂), G): mean ± population std over 5 runs (seeds 42–46, 30 starts each); lower is better. In every
arm the loss is the MMD, with 250 samples per search step, 3,000 per refinement / phase-1 step, and the same
selection (the 5 lowest re-scored 5 × 5,000). Every cell has 5 runs.

| dim X × dim Y | MLGD\*, FiniteDiff | MLGD\*, Backprop | GB-FESO, FiniteDiff | GB-FESO, Backprop |
|---|---|---|---|---|
| 1×1 | 0.024 ± 0.015 | 0.031 ± 0.018 | **0.014 ± 0.017** | 0.030 ± 0.018 |
| 1×5 | 0.017 ± 0.008 | 0.018 ± 0.006 | **0.014 ± 0.009** | 0.018 ± 0.012 |
| 1×10 | 0.048 ± 0.033 | 0.047 ± 0.034 | 0.045 ± 0.012 | **0.042 ± 0.032** |
| 5×1 | **0.023 ± 0.007** | 0.029 ± 0.011 | 0.026 ± 0.006 | 0.024 ± 0.010 |
| 5×5 | **0.063 ± 0.020** | 0.075 ± 0.028 | 0.073 ± 0.027 | 0.067 ± 0.022 |
| 5×10 | **0.070 ± 0.034** | 0.087 ± 0.042 | 0.109 ± 0.042 | 0.092 ± 0.024 |
| 10×1 | **0.070 ± 0.027** | 0.075 ± 0.016 | 0.345 ± 0.331 | 0.666 ± 0.408 |
| 10×5 | 0.066 ± 0.018 | 0.076 ± 0.028 | **0.066 ± 0.013** | 0.082 ± 0.026 |
| 10×10 | **0.193 ± 0.177** | 0.405 ± 0.147 | 0.412 ± 0.167 | 0.407 ± 0.151 |

LaTeX rows: `search_x_estimator_rerun.tex`, written by `pipeline/make_table_2x2.py`.

## Against `search_x_estimator.md`

- **Three columns reproduce to within ±0.002.** These are MLGD\* FiniteDiff, MLGD\* Backprop and GB-FESO Backprop.
  ±0.002 is the Monte Carlo noise of the TV estimate.
- **GB-FESO, FiniteDiff matches in eight cells.** At (10, 10) it gives 0.412 against 0.336, which is within one
  run-to-run std (0.14–0.17). Its finite differences make it the least repeatable arm.
- **The reading is unchanged.** At (10, 1) the search decides: the diffusion search reaches 0.07 with either
  estimator, direct search 0.35–0.67. At (10, 10) finite differences help only together with the diffusion
  search: MLGD\* goes from 0.41 to 0.19, while GB-FESO is at 0.41 with either estimator in this rerun.

## Commands

All jobs ran with `-A orzuk`, from the repository root:

```bash
# samplers (one per cell); runs/night/x10_y1/models/d256 -> ../../grid/x10_y1/d256
ARCH=diffusion BATCHES=100000 UNITS=256 BLOCKS=4 DATA=runs/table1v2/$C/data.pt OUTD=runs/night/grid/$C/d256 \
  sbatch -A orzuk --partition=salmon --export=ALL experiments/synthetic/slurm/slurm_train_sampler.sh
# MLGD* (FiniteDiff) and GB-FESO (Backprop): runs/gbfeso_d256/<cell>/{MLGDstar,GBFESO-MMD}_s*.json,
# the recipes of the README section "GB-FESO on the retrained samplers"
# MLGD* (Backprop) and GB-FESO (FiniteDiff): runs/search_x_estimator/<cell>/{mlgd_pw,gbfeso_zo_pw}_s*.json
CELLS="x1_y1 x1_y5 x1_y10 x5_y1 x5_y5 x5_y10 x10_y1 x10_y5 x10_y10" sbatch -A orzuk --partition=salmon \
  --time=12:00:00 --array=<tasks> experiments/synthetic/slurm/slurm_search_x_estimator.sh
python experiments/synthetic/pipeline/make_table_2x2.py --data-root runs/table1v2
```

Two nodes, salmon-06 and salmon-07, failed jobs at start without writing a log, so the (10, 10) tasks were
resubmitted with `--exclude=salmon-07`.
