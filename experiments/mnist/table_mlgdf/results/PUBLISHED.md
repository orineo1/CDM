# Published reference -- MNIST rotated digits (paper Table 2)

Declared reference for this experiment. These are the numbers **as published**.

Source: the paper, Table 2, `\label{tab:mnist}`
(`main.tex` in the paper source; caption: "MNIST rotated-digits task (15
seeds). Digit percentages are over classified seeds only (up to 3 per target
fell below the classifier threshold). Timing varies with hyperparameters.").

| Target G | SWD_all | SWD_top5 | Time (s) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform  | 0.052 ± .03 | 0.033 ± .01 | 19.2 ± 0.1 | 79 | -- | -- | -- | -- | -- | -- | 7 | -- | 14 |
| bimodal  | 0.081 ± .07 | 0.031 ± .01 | 22.4 ± 3.5 | 15 | 15 | -- | -- | -- | 8 | 15 | -- | 38 | 8 |
| unimodal | 0.049 ± .02 | 0.033 ± .01 | 9.3 ± 0.2 | -- | -- | 38 | 31 | -- | -- | -- | 23 | 8 | -- |

Digit columns are percentages over the classified seeds of that target.

Run configuration behind the table: paper Table 13 `tab:mnist_lgd_hparams`,
transcribed into `../config/targets.tsv`.

## Provenance of artifacts

See `../PROVENANCE.md` for the source repository, branch and commit of the code
and of the committed run records.
