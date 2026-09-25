# table8_ablations

**Produces:** Table 8 `tab:sim_ablation` — four arms: no refinement, refined (= Table 1), annealed resampling, oracle `f_phi`.

| | |
|---|---|
| Script | `scripts/make_table1.py` |
| Full configuration | about three grids |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~1 min |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
The refined arm is not re-run here; it is `table1_main_grid`.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
