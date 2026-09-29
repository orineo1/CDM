# ncond_ramp

**Produces:** The App. B.6 n_cond-ramp comparison (paper prose, no float).

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/make_table1.py` with `--extra "--n-cond-schedule linear --n-cond-min 50 --step-decay ..."` |
| Full configuration | one grid |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~30 s |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
`run.sh` is the sweep driver; the flags are in `find_xstar.py`.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
