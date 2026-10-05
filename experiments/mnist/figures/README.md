# figures

Draws the MNIST figures from the run records of `../table_mlgdf` (nothing is optimized again). Per target:

| file | contents |
|---|---|
| `<run>_top5_notitle.png` | the 5 digits with the lowest SWD |
| `<run>_dist_polar_notitle.png` | for those 5, the sampled angle histogram against the target (polar) |
| `<run>_all_seeds.png` | all 15 recovered digits, titled `Seed \| Loss \| label` |

```bash
bash run.sh smoke              # after ../table_mlgdf/run.sh smoke
bash run.sh full               # after ../table_mlgdf/run.sh full; writes outputs/full/
bash run.sh full <dir>         # from any directory of run records
```

`results/` holds the figures of the committed run (`../export_results.py` copies them).
