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
| `table2.md` | Table 2 (also shown in `../README.md`) |
| `seeds.csv` | one row per target and seed: run name, SWD, time, in top-5, label (`None` below the threshold), confidence |
| `images.npz` | the recovered images (`<target>_images`, 15x28x28, as optimised, not clamped) and the same per-seed values (`<target>_seed`, `_swd`, `_time_s`, `_label` with -1 for None, `_confidence`); `np.load` it (no pickle) |
| `run_info.txt` | the date, commit, classifier, threshold, GPU, torch version and settings of the run |

The run name in `seeds.csv` and in the figure file names encodes the settings, e.g.
`uniform_st300_ssdouble_xt3_ns600_cl0`: T = 300 steps, step size 2 zeta_t, n_MC = 3,
n_cond = 600, no clamping; `var300` is the target variance where there is one.

## Notes

- The digit label depends only on the saved image and the classifier; the SWDs and
  images do not depend on the classifier. If `../checkpoints/robust_classifier.pth` is
  missing, the first run trains it (10 epochs on MNIST, seed 42, a few minutes on a
  GPU) and saves it there; MNIST is downloaded to `./data` under the directory the
  run is started from.
- Per-seed results are exactly repeatable on the same GPU model and software
  versions. A different GPU or torch/CUDA version can change the last digits, and a
  trajectory near a tipping point between two digits can then end on the other one.
- On a GPU, one short untimed warm-up run precedes seed 0, so that CUDA start-up is
  not counted in its time (`--no-warmup` turns it off). It cannot change any result.
