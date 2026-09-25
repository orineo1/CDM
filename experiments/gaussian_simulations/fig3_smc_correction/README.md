# fig3_smc_correction

**Produces:** Figure 3 `fig:smc_correction`.

| | |
|---|---|
| Script | `scripts/fig_smc_correction.py` (smoke first trains the toy networks via `fig1_toy_example.py --quick`) |
| Full tier | 4 betas x 10 pooled populations x 2,000 particles x 2 arms |
| GPU | optional — a CPU variant of the reference output exists |
| Smoke tier | CPU, ~1 min |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
Both tiers pass four betas: the script exits non-zero on a single `--betas` value (see `../SMOKE-RUNS.md`).

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
