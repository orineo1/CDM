> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# App. B.6 and B.7

**(5, 1)**, paper.txt:869-870:

> "Replacing the MMD by the sliced Wasserstein loss at (5, 1) changes nothing either
> (MLGD 0.072 against 0.076, MLGD-F 0.20 against 0.25)."

| cell | method | SWD | MMD |
|---|---|---|---|
| (5, 1) | MLGD   | 0.072 | 0.076 |
| (5, 1) | MLGD-F | 0.20  | 0.25  |

**(10, 1)**, paper.txt:910-911: the sliced-Wasserstein loss gives **0.81 +- 0.27**.

Transcribed from `paper.txt`. Alongside: `table1_w1_5x1.md`, imported from
`orineo1/GlobalConditional` `master` `9ddf570`. The projection count is not given in the
text, so `run.sh full` leaves `--swd-projections` at the library default.
