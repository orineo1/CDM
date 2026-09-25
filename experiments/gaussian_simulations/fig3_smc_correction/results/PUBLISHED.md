> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

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

Transcribed from `paper.txt`. Alongside, imported from `orineo1/GlobalConditional`
`master` `9ddf570`: `fig_smc_hard.json`, `fig_smc_hard_cpu.json`, and the
`--max-shift`/`--ess-frac` sweep `ms{0.25_ess1.0, 0.5_ess0.5, 0.5_ess1.0, 1.0_ess1.0}.json`
(from `results/smc_sweep/`). `ms0.25_ess1.0` is the published configuration.
