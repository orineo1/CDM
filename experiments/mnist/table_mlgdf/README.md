# table_mlgdf: the MNIST table

MLGD-F on rotated MNIST for three angle targets, 15 seeds each: the SWD between the angle distribution of the
recovered image and the target (all seeds and the 5 best), the time per seed, and the digits the classifier reads
off the recovered images.

## Settings (`config/targets.tsv`)

| target | n_cond | T (DDIM steps) | n_MC | step size | target variance (deg²) |
|---|---|---|---|---|---|
| unimodal | 1500 | 130 | 3 | 2 ζ_t | 515 |
| bimodal | 1500 | 125 | 3 | ζ_t | 300 |
| uniform | 600 | 300 | 3 | 2 ζ_t | - |

ζ_t = r_t / (1 + r_t²) + 5t/1000. For all targets: SWD with 50 projections, iCT sampling steps
[150, 50, 20, 10, 5, 1], seeds 0–14, no clamping during the optimization, classifier threshold 0.8.

## Run

```bash
bash run.sh smoke                        # CPU, seconds: checks the code path only
bash run.sh full                         # GPU, about 5 to 20 s per seed on an L4
CLASSIFIER=<file.pth> bash run.sh full   # another classifier (default ../checkpoints/robust_classifier.pth)
```

`run.sh full` writes `outputs/full/` (not in git): one `.pkl` run record per target, `table2.md` and `run_info.txt`.
`make_table.py <dir>` rebuilds the table from the records.

## Committed results (`results/`)

| file | contents |
|---|---|
| `table2.md` | the table |
| `seeds.csv` | per target and seed: run name, SWD, time, in top 5, label (`None` below the threshold), confidence |
| `images.npz` | the recovered images (`<target>_images`, 15×28×28, not clamped) and the per-seed values; `np.load` it |
| `run_info.txt` | date, commit, classifier, threshold, GPU, torch version and settings |

Run names encode the settings, e.g. `uniform_st300_ssdouble_xt3_ns600_cl0`: T = 300, step size 2 ζ_t, n_MC = 3,
n_cond = 600, no clamping. Results repeat exactly on the same GPU model and software versions; elsewhere the last
digits can change, and a trajectory near a tipping point can end on another digit.
