> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# App. B.6, "Kernel bandwidth and schedules" (paper.txt:862-866)

> "Repeating the full grid with the bandwidths detached (now the default of the library)
> reproduces every cell within run-to-run noise (e.g. MLGD TV 0.053 +- 0.030 against
> 0.060 +- 0.018 at (5, 5) and 0.33 +- 0.13 against 0.20 +- 0.05 at (10, 10))."

| cell | method | detached | non-detached |
|---|---|---|---|
| (5, 5)   | MLGD | 0.053 +- 0.030 | 0.060 +- 0.018 |
| (10, 10) | MLGD | 0.33 +- 0.13   | 0.20 +- 0.05   |

The paper states that the Table 1 runs used the non-detached bandwidth, and that detached
is now the library default -- so `--extra=--bandwidth-grad` selects the Table 1 setting.

Transcribed from `paper.txt`. Alongside: `table1_detach.md`, imported from
`orineo1/GlobalConditional` `master` `9ddf570`.
