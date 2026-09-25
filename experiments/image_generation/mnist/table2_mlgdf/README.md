# table2_mlgdf

**Produces:** Table 2 `tab:mnist` (inventory MN-01) -- MLGD-F on rotated MNIST
for the three angle targets, 15 seeds each, reporting SWD over all seeds, SWD
over the top 5 by final SWD, wall-clock time per seed, and the digit classes
the rotation-robust classifier reads off the recovered images.

| | |
|---|---|
| Script | `run_mlgdf.py` (this directory); the loop is `guidance.core.optimize_guided` |
| Full tier | 3 targets x 15 seeds, hyperparameters from `config/targets.tsv` (paper Table 13) |
| GPU | yes (L40S in the paper); the paper reports under 15 min per target |
| Smoke tier | CPU, ~1 min: 1 seed per target, 4 DDIM steps, 32 conditional samples, untrained prior |
| **Tier actually run here** | **smoke only** (2026-09-25, CPU, PASS) |

## Run

```
bash run.sh smoke
bash run.sh full
```

`run.sh full` needs both checkpoints (see the area README) and writes one `.pkl`
run record per target into `outputs/full/`, plus one `gcio` JSONL record per seed.
`figures/` reads those records.

## Reference numbers

`results/PUBLISHED.md` -- the paper's Table 2 as published, with provenance.
Nothing in `outputs/` is a published number.
