# table_mlgdf

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
run record per target into `outputs/full/`.
`figures/` reads those records.

## Reference numbers

`results/PUBLISHED.md` -- the paper's Table 2 as published, with provenance.
`outputs/` holds local runs and is never a published number.

## The table

`run.sh full` ends by printing Table 2 (SWD over all seeds and the best 5, time per
seed, and the % of classified seeds recovered as each digit) and writing it to
`outputs/full/table2.md`. To rebuild it from existing run records without re-running:

```bash
python experiments/mnist/table_mlgdf/make_table.py experiments/mnist/table_mlgdf/outputs/full
```

## Settings checked against the paper's own run records

The run records behind Table 2 are committed in `orineo1/conditional-matching-paper`,
branch `gread-search-mnist`, `MNIST/checkpoints_and_results/`: uniform =
`unif_ns600_st290_ssoriginal_xt3.pkl`, bimodal = `Bimodal.pkl`, unimodal =
`uni_var515_st130_ssdouble_xt3.pkl`. Their SWDs and times reproduce Table 2 exactly.

- **No per-step clamp.** The records' images reach pixel values of 2.0-3.3; with the
  per-step clamp to [-1, 1] the final image stays within about +-1.01 (the last DDIM
  step, at t = 0, is almost the identity). `run.sh full` passes `--no-clamp`.
- **The paper's classifier.** `experiments/mnist/checkpoints/robust_classifier.pth` is
  the classifier committed next to those records. Applied to them at threshold 0.75 it
  reproduces the bimodal and unimodal digit columns of Table 2 exactly; for uniform it
  gives 0: 83 %, 1: 8 %, 9: 8 % (3 unlabelled) against the published 0: 79 %, 7: 7 %,
  9: 14 %, a one-to-two seed difference that no threshold removes.

