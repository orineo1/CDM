# table2_mlgdf

**Produces:** Table 2 `tab:mnist` -- MLGD-F on rotated MNIST
for the three angle targets, 15 seeds each, reporting SWD over all seeds, SWD
over the top 5 by final SWD, wall-clock time per seed, and the digit classes
the rotation-robust classifier reads off the recovered images.

| | |
|---|---|
| Script | `run_mlgdf.py` (this directory); the loop is `guidance.core.optimize_guided` |
| Full configuration | 3 targets x 15 seeds, hyperparameters from `config/targets.tsv` (paper Table 13) |
| GPU | yes (L40S in the paper); the paper reports under 15 min per target |
| Reduced configuration (`smoke`) | CPU, ~1 min: 1 seed per target, 4 DDIM steps, 32 conditional samples, untrained prior |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```

`run.sh full` needs both checkpoints (see the area README) and writes one `.pkl`
run record per target into `outputs/full/`, plus one `gcio` JSONL record per seed.
`figures/` reads those records.

## Reference numbers

`results/PUBLISHED.md` -- the paper's Table 2 as published, with provenance.
`outputs/` holds local runs and is never a published number.
