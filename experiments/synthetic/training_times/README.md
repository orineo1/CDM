# training_times

**Produces:** Table 10 `tab:training_times`.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/generate_data.py --params-file` + `experiments/synthetic/pipeline/train_models.py` |
| Full configuration | 9 trainings |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~10 s |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
`train_models.py` emits no timing, so `run.sh` times each invocation with the shell. `experiments/synthetic/configs/mog/` has no 5D file.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Results

`results/` holds the outputs of `run.sh full` behind the paper, copied from `outputs/`.
