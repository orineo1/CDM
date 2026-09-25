# step_rule_sweep

**Produces:** The App. B.6 step-rule sweeps, constant-zeta and Adam (paper prose, no float).

| | |
|---|---|
| Script | `scripts/make_table1.py` (`--guidance-optim sgd|adam` + `--step-size`) |
| Full tier | 3 cells x the values in `config/zeta_grid.txt` x 3 runs |
| GPU | yes |
| Smoke tier | CPU, ~20 s |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
`config/zeta_grid.txt` is read by `run.sh full`.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
