# swd_loss_control

**Produces:** The App. B.6 and B.7 SWD-for-MMD comparisons (paper prose, no float).

| | |
|---|---|
| Script | `scripts/make_table1.py` with `--extra "--loss swd"` |
| Full tier | two cells |
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
