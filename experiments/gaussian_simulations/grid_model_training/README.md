# grid_model_training

**Produces:** The 27 networks behind the grid experiments (App. B.4).

| | |
|---|---|
| Script | `scripts/generate_data.py` + `scripts/train_models.py` |
| Full configuration | 27 trainings |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~10 s |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
`table1_main_grid` performs these trainings itself when its runs-dir is empty; this directory runs the training stage alone.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
