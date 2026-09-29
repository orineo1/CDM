# swd_loss_control

**Produces:** The App. B.6 and B.7 SWD-for-MMD comparisons (paper prose, no float).

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/make_table1.py` with `--extra "--loss swd"` |
| Full configuration | two cells |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~30 s |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
