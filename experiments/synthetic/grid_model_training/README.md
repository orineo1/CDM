# grid_model_training

**Produces:** The 27 networks behind the grid experiments (App. B.4).

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/generate_data.py` + `experiments/synthetic/pipeline/train_models.py` |
| Full configuration | 27 trainings |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~10 s |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
`main_grid` performs these trainings itself when its runs-dir is empty; this directory runs the training stage alone.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Results

`results/` holds the outputs of `run.sh full` behind the paper, copied from `outputs/`.
