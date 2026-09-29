# model_quality

**Produces:** Table 5 `tab:model_quality`.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/check_models.py` (smoke also uses `generate_data.py` and `train_models.py`) |
| Full configuration | one check per scenario, once the 27 networks exist |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~1 min |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
The full tier reads trained scenarios from `RUNS_DIR` (default `runs/table1v2`).

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
