# infeasible_target

**Produces:** Table 9 `tab:nonstar_results` and Figure 4 `fig:nonstar`.

| | |
|---|---|
| Script | `scripts/generate_data.py`, `scripts/train_models.py`, `scripts/find_xstar.py` |
| Full tier | 2 trainings (20,000 batches) + 3 arms x 25 restarts |
| GPU | yes |
| Smoke tier | CPU, ~15 s |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
The scenario is `config/nonstar_params.txt`. The pointwise arms use `find_xstar.py --loss l2` (see `../CHANGES-TO-LIBRARY.md`).

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
