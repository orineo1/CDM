> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# App. B.6 (paper.txt:866-869)

> "Ramping ncond from 50 to 250 along the trajectory together with a cosine decay of the
> guidance step changes no cell beyond noise at dim X <= 5 and hurts MLGD at (10, 10)
> (0.20 -> 0.43; it helps MLGD-F at (5, 1), 0.25 -> 0.13)."

| cell | method | constant | ramp + cosine decay |
|---|---|---|---|
| (10, 10) | MLGD   | 0.20 | 0.43 |
| (5, 1)   | MLGD-F | 0.25 | 0.13 |

Transcribed from `paper.txt`. The cosine-decay floor is not given in the text;
`run.sh` uses `--step-decay 0.1`, the value in the `scripts/slurm_table1.sh` example.
