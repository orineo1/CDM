# smc_correction

**Produces:** Figure 3 `fig:smc_correction`.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/fig_smc_correction.py` (smoke first trains the toy networks via `fig1_toy_example.py --quick`) |
| Full configuration | 4 betas x 10 pooled populations x 2,000 particles x 2 arms |
| GPU | optional — a CPU variant of the reference output exists |
| Reduced configuration (`smoke`) | CPU, ~1 min |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
Both configurations pass four betas; the script requires more than one `--betas` value.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
