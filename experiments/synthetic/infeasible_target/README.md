# infeasible_target

**Produces:** Table 9 `tab:nonstar_results` and Figure 4 `fig:nonstar`: a point-mass target
`y* = 4` that no `x` can realise. The scenario (`config/nonstar_params.txt`) has a design
`x_bi = -2` whose conditional is bimodal (modes at y = 2 and 4) and a design `x_uni = 2`
that is unimodal at y = 3. The distribution-matching loss (MMD) should return `x_bi`, the
best available design; the pointwise loss `E(y - y*)^2` of standard inverse design converges
to `x_uni`, closer to `y*` on average but with no mass on it.

| | |
|---|---|
| Script | `generate_data.py` + `train_models.py` (prior and iCT sampler), `find_xstar.py` for the three arms, then `make_table9.py` (Table 9 and Figure 4) |
| Full configuration | 2 trainings of 20,000 batches + 3 arms x 25 restarts (MMD n_cond 250; pointwise n_cond 250; pointwise n_cond 1) |
| GPU | yes; under an hour |
| Reduced configuration (`smoke`) | CPU, ~1 min |

## Run

```bash
bash experiments/synthetic/infeasible_target/run.sh smoke
bash experiments/synthetic/infeasible_target/run.sh full
sbatch experiments/synthetic/slurm/slurm_run.sh infeasible_target
```

Outputs go to `outputs/` (gitignored): `arm_{MMD_n250,L2_n250,L2_n1}.json` (every restart),
`table9.{md,csv}`, `table9_restarts.csv` (x̂ of every restart) and `table9_fig4.png`.

## Committed results (`results/`)

Written by `../export_results.py`: the files above plus `run_info.txt`.

## Paper numbers

Table 9 `tab:nonstar_results`, R = 25 restarts per arm. "% at x_bi" is the fraction of
restarts landing closer to `x_bi` than to `x_uni`.

| Loss | n_cond | % at x_bi | Time (s) |
|---|---|---|---|
| MMD       | 250 | 96% | 7.40 ± 1.06 |
| Pointwise | 250 |  0% | 4.31 ± 0.60 |
| Pointwise |   1 | 12% | 1.65 ± 0.46 |

Figure 4 `fig:nonstar` plots the same run. The scenario constants come from
`orineo1/conditional-matching-paper` `main` `fc4389d`,
`simulations/notebooks/Exp_nonstar_mmd_target_ablation.ipynb` (transcribed in `config/`).
