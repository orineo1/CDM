# fig1_toy_example

**Produces:** Figure 1 `fig:sim_2d` — four panels.

| | |
|---|---|
| Script | `scripts/fig1_toy_example.py` |
| Full tier | four beta values x 2,000 trajectories, plus the four methods |
| GPU | yes |
| Smoke tier | CPU, ~1 min (the script's own `--quick`) |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
Or on a cluster: `sbatch scripts/slurm_fig1.sh`. Toy networks are cached under `outputs/ckpt/<setting>/`.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
