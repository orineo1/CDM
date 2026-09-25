# retrain_10x1_50k

**Produces:** The App. B.7 retrained-sampler arm at (10,1) (paper prose, no float).

| | |
|---|---|
| Script | `scripts/make_table1.py --settings 10x1 --epochs 50000 --force` |
| Full tier | one retraining + one cell |
| GPU | yes |
| Smoke tier | CPU, ~1 min |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
`--force` retrains all three networks, not the conditional alone.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
