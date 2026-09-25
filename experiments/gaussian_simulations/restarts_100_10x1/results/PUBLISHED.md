> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# Section 5.1 and App. B.7 (paper.txt:915-917)

> "with 100 restarts per run instead of 10 (three runs), MLGD reaches TV 0.15 +- 0.12
> (70/300 trajectories below 0.5, at 10x the cost), the exact-conditional oracle
> 0.39 +- 0.10 and MLGD-F 0.52 +- 0.12."

| method | TV at (10, 1), 100 restarts |
|---|---|
| MLGD            | 0.15 +- 0.12 |
| MLGD-X (oracle) | 0.39 +- 0.10 |
| MLGD-F          | 0.52 +- 0.12 |

Transcribed from `paper.txt`. Alongside: `table1_10x1_r100.md`, imported from
`orineo1/GlobalConditional` `master` `9ddf570`.
