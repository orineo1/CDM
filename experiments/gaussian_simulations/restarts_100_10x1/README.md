# restarts_100_10x1

**Produces:** The App. B.7 100-restart run at (10,1) (paper prose, no float).

| | |
|---|---|
| Script | `scripts/make_table1.py --settings 10x1 --budget 100` |
| Full tier | one cell at 10x budget, 3 runs, 3 methods |
| GPU | yes |
| Smoke tier | CPU, ~20 s |
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
