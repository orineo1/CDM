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

## Timing

On a GPU, `run_mlgdf.py` first does one short untimed run (3 DDIM steps) so that CUDA
start-up is not charged to seed 0, and synchronises the GPU around each timed seed. The
warm-up cannot change any result: every random draw in the seed loop comes from torch,
and each seed re-seeds torch. `--no-warmup` turns it off. Note that the paper's own
records include the warm-up in seed 0 (bimodal: 35.4 s against 21.5 s for the other
seeds), which is part of the published 22.4 +- 3.5 s.

## Bit-for-bit agreement with the source

The records behind Table 2 were written by the grid-search scripts on branch
`gread-search-mnist` (`mnist_uniform_gridsearch.py`, `sample_mnist_gridsearch.py`,
`mnist_unimodal_gridsearch.py`), not by `main`'s `run_mlgdf.py`. Their optimisation
loops are checked here bit for bit (CPU, same networks and seeds, max difference 0.0
for all three targets). One setting had to follow the branch rather than `main`: the
bimodal modes are [180, 360] degrees in that order (`main` has [0, 180]; same
distribution, different component order, different random draws).

The optimisation is checked against the source's own `optimize_LGD`
(`MNIST/run_mlgdf.py` of conditional-matching-paper): with the same networks and seeds,
on CPU, all three targets give identical images and losses (max difference 0.0). Two
float32 details had to match for that: the SWD is one mean over all sorted differences
(not a mean of per-projection means), and target angles are cast to float32 before
cos/sin. Same code, seeds and settings on the same GPU and software versions give the
same run every time; a different GPU or torch/CUDA version can still change the last
digits, and near a tipping point that can change which digit a seed ends on.

## Relabelling with another classifier or threshold

The classifier only labels the recovered images, so labels can be recomputed without
re-running anything. `relabel.py` writes relabelled copies of the records and a new
`table2.md`, and prints each seed's top class and confidence:

```bash
python experiments/mnist/table_mlgdf/relabel.py \
    --results-dir experiments/mnist/table_mlgdf/outputs/full \
    --out experiments/mnist/table_mlgdf/outputs/full_thr0.5 --threshold 0.5
```

`--classifier` picks the weights (default: the paper's `robust_classifier.pth`; a path
that does not exist is trained by the code and saved there). The paper used 0.75. The
figures can then be drawn from `--out` with `figures/make_figures.py --results-dir`.

