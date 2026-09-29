# training_times

**Produces:** the training-time table `tab:training_times` (wall-clock training time of the
three networks per setting). **Commented out in the current paper**; kept in case it returns.

| | |
|---|---|
| Script | `generate_data.py --params-file` + `train_models.py`; `run.sh` times each training with the shell and writes `outputs/timings.csv` |
| Full configuration | 3 parameter files (`configs/mog/mog_2d_full`, `mog_10d_full`, `ldm_10D_full`) x 3 networks, 10,000 batches of 1,024 each |
| GPU | yes; up to 4 h per network |
| Reduced configuration (`smoke`) | CPU, ~30 s |

## Run

```bash
bash experiments/synthetic/training_times/run.sh smoke
bash experiments/synthetic/training_times/run.sh full
sbatch --time=24:00:00 experiments/synthetic/slurm/slurm_run.sh training_times
```

Outputs go to `outputs/` (gitignored): `timings.csv` (params, network, seconds) and the
networks. No 5D parameter file exists in `configs/mog/`, so the 5D column of Table 10
cannot be reproduced from this repository.

## Committed results (`results/`)

Written by `../export_results.py`: `timings.csv`, `run_info.txt`.

## Paper numbers

# Table 10 `tab:training_times` (paper.txt:1153-1156)

Approximate training times on an NVIDIA L4 GPU.

| Model | 2D | 5D | 10D |
|---|---|---|---|
| Diffusion (unconditional) | ~11 min | ~3 h  | ~4 h  |
| Diffusion (conditional)   | ~12 min | ~3 h  | ~4 h  |
| Consistency Model (CM)    | ~3 min  | ~8 min | ~9 min | The settings come from
`orineo1/conditional-matching-paper` `CLEAN-SIMU` `83de561`,
`simulations/notebooks/Exp_{2D,5D,10D}_cond_1D.ipynb`.
