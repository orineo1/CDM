# restarts_100

**Produces:** The App. B.7 100-restart run at (10,1) (paper prose, no float).

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/make_table1.py --settings 10x1 --budget 100` |
| Full configuration | one cell at 10x budget, 3 runs, 3 methods |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~20 s |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
