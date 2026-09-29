# toy_example

**Produces:** Figure 1 `fig:sim_2d` — four panels.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/fig1_toy_example.py` |
| Full configuration | four beta values x 2,000 trajectories, plus the four methods |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~1 min (the script's own `--quick`) |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
Or on a cluster: `sbatch experiments/synthetic/slurm/slurm_fig1.sh`. Toy networks are cached under `outputs/ckpt/<setting>/`.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
