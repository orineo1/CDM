# infeasible_target

**Produces:** Table 9 `tab:nonstar_results` and Figure 4 `fig:nonstar`.

| | |
|---|---|
| Script | `scripts/generate_data.py`, `scripts/train_models.py`, `scripts/find_xstar.py` |
| Full configuration | 2 trainings (20,000 batches) + 3 arms x 25 restarts |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~15 s |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
The scenario is `config/nonstar_params.txt`. The pointwise arms use `find_xstar.py --loss l2`.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
