# table5_model_checks

**Produces:** Table 5 `tab:model_quality`.

| | |
|---|---|
| Script | `scripts/check_models.py` (smoke also uses `generate_data.py` and `train_models.py`) |
| Full tier | one check per scenario, once the 27 networks exist |
| GPU | yes |
| Smoke tier | CPU, ~1 min |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
The full tier reads trained scenarios from `RUNS_DIR` (default `runs/table1v2`).

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
