# smc_correction

**Produces:** Figure 3 `fig:smc_correction`: the twisted-SMC correction of the β-sweep of
Figure 1(d), and the TV of the plain and corrected histograms to Q_β.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/fig_smc_correction.py` (smoke first trains the toy networks with `fig1_toy_example.py --quick`) |
| Full configuration | 4 β x 10 pooled populations x 2,000 particles x 2 arms; `--max-shift 0.25 --ess-frac 1.0` |
| GPU | yes; tens of minutes |
| Reduced configuration (`smoke`) | CPU, ~5 min |

## Run

```bash
bash experiments/synthetic/smc_correction/run.sh smoke
bash experiments/synthetic/smc_correction/run.sh full
sbatch experiments/synthetic/slurm/slurm_run.sh smc_correction
```

Outputs go to `outputs/` (gitignored): `fig_smc_hard.png`, `fig_smc_hard.json` (the TV per β)
and `fig_smc_hard.npz` (the populations, read by `../toy_example` for panel d). The full
tier needs the toy networks under `outputs/ckpt/hard/`; the smoke tier trains them, the
full tier trains them on first use.

## Committed results (`results/`)

Written by `../export_results.py`: the three files above and `run_info.txt`. Until a full
run replaces them, `results/` holds the run behind the paper, imported from
`orineo1/GlobalConditional` `master` (`fig_smc_hard.json`, `fig_smc_hard_cpu.json` at
`9ddf570`; `fig_smc_hard.npz` at `1a5edcd`) and its `--max-shift`/`--ess-frac` sweep
(`ms*.json`; `ms0.25_ess1.0` is the published configuration).

## Paper numbers

# Figure 3 `fig:smc_correction` and App. B.8 (paper.txt:1053-1063)

Twisted-SMC reweighting of the Figure 1(d) beta-sweep. Ten independent populations of
2,000 trajectories, pooled, per beta; resampled at every step; the corrected arm bounds
the guided shift to sigma_t / 4 (the plain arm is never clipped).

TV of the histogram to Q_beta:

| beta | plain | corrected |
|---|---|---|
| 2  | 0.17 | 0.03 |
| 8  | 0.19 | 0.03 |
| 16 | 0.56 | 0.06 |

beta = 0's 0.03 is the histogram noise floor.
