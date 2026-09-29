> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# Table 10 `tab:training_times` (paper.txt:1153-1156)

Approximate training times on an NVIDIA L4 GPU.

| Model | 2D | 5D | 10D |
|---|---|---|---|
| Diffusion (unconditional) | ~11 min | ~3 h  | ~4 h  |
| Diffusion (conditional)   | ~12 min | ~3 h  | ~4 h  |
| Consistency Model (CM)    | ~3 min  | ~8 min | ~9 min |

Transcribed from `paper.txt`. The settings come from
`orineo1/conditional-matching-paper` `CLEAN-SIMU` `83de561`,
`simulations/notebooks/Exp_{2D,5D,10D}_cond_1D.ipynb`.
