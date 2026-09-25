# ncond_ramp

**Produces:** The App. B.6 n_cond-ramp comparison (paper prose, no float).

| | |
|---|---|
| Script | `scripts/make_table1.py` with `--extra "--n-cond-schedule linear --n-cond-min 50 --step-decay ..."` |
| Full tier | one grid |
| GPU | yes |
| Smoke tier | CPU, ~30 s |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
`run.sh` is the sweep driver; the flags are in `find_xstar.py`.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
