> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# App. B.6, "Sampler training" (paper.txt:872-880)

Retraining all three networks at (5,5), (5,10), (10,5), (10,10) for 50,000 instead of
10,000 batches ("long"), or with 256 instead of 128 hidden units ("wide").

Consistency sampler's normalised SWD at x*: 0.20-0.36, against 0.09-0.28 for the reported
models; diffusion sampler 0.04-0.06 throughout.

MLGD-F, Table 1 value -> long -> wide:

| cell | Table 1 | long | wide |
|---|---|---|---|
| (5, 5)   | 0.50 | 0.82 | 0.63 |
| (5, 10)  | 0.41 | 0.41 | 0.32 |
| (10, 5)  | 0.30 | 0.64 | 0.47 |
| (10, 10) | 0.92 | 0.88 | 0.96 |

MLGD "stays within noise of Table 1".

Transcribed from `paper.txt`. Alongside: `table1_long.md`, `table1_wide.md` and eight
`checks__{long,wide}_x{5,10}_y{5,10}.json`, the outputs of the run.
