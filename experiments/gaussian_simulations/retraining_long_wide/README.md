# retraining_long_wide

**Produces:** The App. B.6 longer/wider retraining study (paper prose, no float).

| | |
|---|---|
| Script | `scripts/make_table1.py` (`--epochs` / `--nunits`, one runs-dir per arm) |
| Full tier | 24 trainings + 8 grids |
| GPU | yes |
| Smoke tier | CPU, ~30 s |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
