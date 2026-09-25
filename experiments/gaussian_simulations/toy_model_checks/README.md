# toy_model_checks

**Produces:** The App. B.8 network check for the toy setting (paper prose, no float).

| | |
|---|---|
| Script | `scripts/check_models.py --setting` (smoke first trains the toy networks via `fig1_toy_example.py --quick`) |
| Full configuration | two settings |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~1 min |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
The paper does not name the setting; the full configuration runs both `hard` and `paper`.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
