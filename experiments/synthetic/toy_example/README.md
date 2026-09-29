# toy_example

**Produces:** Figure 1 `fig:sim_2d`, four panels: the joint and the recovered x̂ of each
method, the loss landscape, the induced conditionals, and the β-sweep of CDM-S. Panel (d)
of the published figure (`fig1_toy_hard_smc.png`) overlays the twisted-SMC populations of
`../smc_correction`; the full tier runs that experiment itself on the same networks.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/fig1_toy_example.py` (`--smc-npz` for panel d) |
| Full configuration | settings `hard` (the paper's), `bimodal`, `paper`; 10 trajectories per method, `n_cond = 250`; β in {0, 2, 8, 16} x 2,000 trajectories |
| GPU | yes; minutes per setting |
| Reduced configuration (`smoke`) | CPU, ~4 min (the script's own `--quick`) |

## Run

```bash
bash experiments/synthetic/toy_example/run.sh smoke
bash experiments/synthetic/toy_example/run.sh full          # trains, simulates, runs ../smc_correction, draws
sbatch experiments/synthetic/slurm/slurm_run.sh toy_example
```

Outputs go to `outputs/` (gitignored): `fig1_<setting>.png`, `fig1_<setting>_sim.json` (all
numbers: x̂, hits of the x* basin, times) and `fig1_<setting>_sweep.npz`; toy networks are
cached under `outputs/ckpt/<setting>/`.

## Committed results (`results/`)

Written by `../export_results.py`: the three files per setting and `run_info.txt`.

## Paper numbers

# Figure 1 `fig:sim_2d` and App. B.8 (paper.txt:1014-1051)

**Setup.** Four clusters in x at -6, -2, 2, 6, a quarter of the mass each; x* = 2 gives the
bimodal target. Every component has x-variance 0.25 and non-zero x-y correlation
(rho = 0.5, -0.7, +-0.4, -0.5). Decoys: three narrow modes at y = 2, 0, -2 (var 0.15) at
x = -6; one mode at y = 2 (var 0.9) at x = -2; one broad mode at y = 0 (var 3) at x = 6.
Networks: MLP 3 x 128, 3,000 epochs, batch 1,024, fresh draws.

**Panel (b), population loss at the cluster centres:** 0.12 at x = -6, 0.24 at x = -2,
0.21 at x = 6.

**Panels (a)/(c).** Ten trajectories per method from x_T ~ N(0,1), MMD with n_cond = 250.
MLGD reaches the x* basin in 1 restart out of 10 (exact L2 = 6e-4); the other nine and all
ten MLGD-F restarts end in the broad decoy at x ~ 5.5. PMLGD and PMLGD-F reach the x*
basin with 6 and 7 of their ten particles.

**Panel (d).** 2,000 trajectories per beta, beta in {0, 2, 8, 16}, 80 bins, analytic
L2-GMM loss as guidance, eta = 1, DPS step convention. At beta = 8 and 16 the samples
leave the x = -2 cluster, where Q_beta keeps 7% and 2% of its mass. The toy's training configuration also appears at
`orineo1/conditional-matching-paper` `main` `fc4389d`,
`simulations/notebooks/toy_example_with_beta_sweep.ipynb`.
