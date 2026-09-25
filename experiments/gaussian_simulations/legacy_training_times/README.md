# legacy_training_times

**Produces:** Table 10 `tab:training_times`.

| | |
|---|---|
| Script | `scripts/generate_data.py --params-file` + `scripts/train_models.py` |
| Full tier | 9 trainings |
| GPU | yes |
| Smoke tier | CPU, ~10 s |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
`train_models.py` emits no timing, so `run.sh` times each invocation with the shell. `SimulationParameters/` has no 5D file.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
