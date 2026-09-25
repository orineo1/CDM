# table8_ablations

**Produces:** Table 8 `tab:sim_ablation` — four arms: no refinement, refined (= Table 1), annealed resampling, oracle `f_phi`.

| | |
|---|---|
| Script | `scripts/make_table1.py` |
| Full tier | about three grids |
| GPU | yes |
| Smoke tier | CPU, ~1 min |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
The refined arm is not re-run here; it is `table1_main_grid`.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
