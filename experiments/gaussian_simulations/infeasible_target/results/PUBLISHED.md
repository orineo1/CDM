> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# Table 9 `tab:nonstar_results` (paper.txt:1124-1127)

No-star target, R = 25 restarts per arm. `% at x_bi` is the fraction of restarts landing
closer to `x_bi` (L* = 280.805) than to `x_uni` (283.505).

| Loss | n_cond | % at x_bi | Time (s) |
|---|---|---|---|
| MMD       | 250 | 96% | 7.40 +- 1.06 |
| Pointwise | 250 |  0% | 4.31 +- 0.60 |
| Pointwise |   1 | 12% | 1.65 +- 0.46 |

Figure 4 `fig:nonstar` plots the same run.

Transcribed from `paper.txt`. The scenario constants in `../config/nonstar_params.txt`
come from `orineo1/conditional-matching-paper` `main` `fc4389d`,
`simulations/notebooks/Exp_nonstar_mmd_target_ablation.ipynb`.
