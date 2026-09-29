> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

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
leave the x = -2 cluster, where Q_beta keeps 7% and 2% of its mass.

Transcribed from `paper.txt`. The toy's training configuration also appears at
`orineo1/conditional-matching-paper` `main` `fc4389d`,
`simulations/notebooks/toy_example_with_beta_sweep.ipynb`.
