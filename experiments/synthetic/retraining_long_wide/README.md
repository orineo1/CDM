# retraining_long_wide

**Produces:** The App. B.6 longer/wider retraining study (paper prose, no float).

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/make_table1.py` (`--epochs` / `--nunits`, one runs-dir per arm) |
| Full configuration | 24 trainings + 8 grids |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~30 s |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
Outputs go to `outputs/`, which is gitignored and ships empty.

## Results

`results/` holds the outputs of `run.sh full` behind the paper, copied from `outputs/`.
