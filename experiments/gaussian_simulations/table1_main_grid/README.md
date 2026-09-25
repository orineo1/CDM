# table1_main_grid

**Produces:** Table 1 `tab:sim_main` and Table 7 `tab:sim_full`.

| | |
|---|---|
| Script | `scripts/make_table1.py` (drives `generate_data.py`, `train_models.py`, `check_models.py`, `find_xstar.py`) |
| Full configuration | 9 scenarios x 4 methods x 5 runs |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~1 min |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
Or on a cluster: `sbatch scripts/slurm_table1.sh`.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
