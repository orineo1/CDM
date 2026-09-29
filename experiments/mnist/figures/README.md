# figures

Draws the MNIST figures from Table 2's run records (`../table_mlgdf/outputs/full/`).
Nothing is optimised again; the angle histograms sample the conditional model, so a
small GPU helps but a CPU works.

Per target (one file set per row of Table 2):

| file | contents |
|---|---|
| `<run>_all_seeds.png` | all 15 recovered digits, each titled `Seed \| Loss \| label` |
| `<run>_all_row1..3.png` | the same digits, one row of 5 per file |
| top-k images | the 5 digits with the lowest SWD |
| top-k distributions | for those 5, the sampled angle histogram (cartesian and polar) against the target |

`_notitle` copies are written without titles, for the paper.

## Run

```bash
bash run.sh smoke              # after ../table_mlgdf/run.sh smoke
bash run.sh full               # after ../table_mlgdf/run.sh full; writes outputs/full/
bash run.sh full <dir>         # draw from any directory of run records
```

## Committed results (`results/`)

The figures of the last full run, copied by `../export_results.py`.
