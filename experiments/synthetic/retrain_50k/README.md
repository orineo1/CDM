# retrain_50k

**Produces:** The App. B.7 retrained-sampler arm at (10,1) (paper prose, no float).

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/make_table1.py --settings 10x1 --epochs 50000 --force` |
| Full configuration | one retraining + one cell |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~1 min |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
`--force` retrains all three networks, not the conditional alone.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
