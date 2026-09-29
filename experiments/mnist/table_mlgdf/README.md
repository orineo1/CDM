# table_mlgdf: Table 2

MLGD-F on rotated MNIST for three angle targets, 15 seeds each. For every target the
table reports the sliced Wasserstein distance (SWD) between the angle distribution of
the recovered image and the target, over all seeds and over the 5 best seeds, the
time per seed, and the digits the classifier reads off the recovered images.

## Settings (`config/targets.tsv`)

| target | n_cond | T (DDIM steps) | n_MC | step size | target variance (deg^2) |
|---|---|---|---|---|---|
| unimodal | 1500 | 130 | 3 | 2 zeta_t | 515 |
| bimodal | 1500 | 125 | 3 | zeta_t | 300 |
| uniform | 600 | 300 | 3 | 2 zeta_t | - |

zeta_t = r_t / (1 + r_t^2) + 5t/1000. Fixed for all targets: SWD with 50 projections,
iCT sampling steps [150, 50, 20, 10, 5, 1], seeds 0-14, no clamping during the
optimisation, classifier threshold 0.8 (below it an image is counted as `None`).

## Run

```bash
bash run.sh smoke                  # CPU, seconds: checks the code path only
bash run.sh full                   # GPU (about 5 to 20 s per seed on an L4)
CLASSIFIER=<file.pth> bash run.sh full   # pick the classifier (default ../checkpoints/robust_classifier.pth)
```

`run.sh full` writes to `outputs/full/` (not in git): one `.pkl` run record per
target, `table2.md` (printed at the end of the run) and `run_info.txt` (commit,
classifier, GPU, settings). `make_table.py <dir>` rebuilds the table from the records.

## Committed results (`results/`)

Written by `../export_results.py` from a finished run:

| file | contents |
|---|---|
| `table2.md` | Table 2 |
| `seeds.csv` | one row per target and seed: SWD, time, in top-5, label, confidence |
| `images.npz` | the recovered images and the same per-seed values; `np.load` it (no pickle) |
| `run_info.txt` | the commit, classifier, GPU, torch version and settings of the run |

## Notes

- The digit label depends only on the saved image and the classifier; the SWDs and
  images do not depend on the classifier.
- Per-seed results are exactly repeatable on the same GPU model and software
  versions. A different GPU or torch/CUDA version can change the last digits, and a
  trajectory near a tipping point between two digits can then end on the other one.
- On a GPU, one short untimed warm-up run precedes seed 0, so that CUDA start-up is
  not counted in its time (`--no-warmup` turns it off). It cannot change any result.
