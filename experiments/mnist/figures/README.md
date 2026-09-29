# figures

Draws the MNIST figures from Table 2's run records (`../table_mlgdf/outputs/full/`).
Nothing is optimised again; the angle histograms sample the conditional model, so a
small GPU helps but a CPU works.

Three files per target (one set per row of Table 2):

| file | contents |
|---|---|
| `<run>_all_seeds.png` | all 15 recovered digits, each titled `Seed \| Loss \| label` |
| `<run>_top5_notitle.png` | the 5 digits with the lowest SWD, without titles (for the paper) |
| `<run>_dist_polar_notitle.png` | for those 5, the sampled angle histogram against the target, polar, without titles |

The digits are displayed clipped to [-1, 1] and mapped to [0, 1]; the histograms sample
the conditional model given those displayed images (as the paper's figure code did).

## Run

```bash
bash run.sh smoke              # after ../table_mlgdf/run.sh smoke
bash run.sh full               # after ../table_mlgdf/run.sh full; writes outputs/full/
bash run.sh full <dir>         # draw from any directory of run records
```

## Committed results (`results/`)

The figures of the last full run, copied by `../export_results.py`.
