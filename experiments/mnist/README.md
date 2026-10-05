# MNIST: rotated digits

`X` is a 28×28 MNIST image and `Y` its rotation angle. The prior `P(X)` is an unconditional DDPM over randomly
rotated digits; the conditional sampler `P(Y | X)` is a consistency (iCT) model of the angle. MLGD-F searches for
an image whose angle distribution matches a target on the circle: uniform, bimodal (0° and 180°) or unimodal (0°).
A rotation-robust classifier then reads off which digit was found.

| folder | contents |
|---|---|
| `table_mlgdf/` | the MNIST table: 3 targets × 15 seeds; settings, runner and `results/` |
| `figures/` | the digit and angle-histogram figures, and `results/` |
| `train/` | training of the two networks (not needed: the published checkpoints are downloaded) |
| `checkpoints/` | where the networks and the classifier are placed (not in git) |
| `export_results.py` | copies a finished run into the committed `results/` folders |

## Results

Committed in `table_mlgdf/results/` and `figures/results/` (NVIDIA L4, torch 2.11; `table_mlgdf/results/run_info.txt`).

| Target | SWD_all | SWD_top5 | Time (s) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | None |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 0.049 ± 0.043 | 0.029 ± 0.006 | 18.0 ± 0.1 | 91 | -- | -- | -- | -- | -- | 9 | -- | -- | -- | 4 |
| bimodal | 0.081 ± 0.066 | 0.034 ± 0.008 | 8.0 ± 0.1 | 18 | 18 | -- | -- | -- | 9 | 18 | -- | 27 | 9 | 4 |
| unimodal | 0.050 ± 0.016 | 0.034 ± 0.008 | 8.1 ± 0.1 | -- | -- | 38 | 31 | -- | -- | -- | 23 | 8 | -- | 2 |

Digit columns: % of the classified seeds. None: seeds below the classifier threshold of 0.8, excluded from the %.
SWD: mean ± std over all 15 seeds and over the 5 best.

## Reproduce

```bash
bash experiments/mnist/table_mlgdf/run.sh full   # GPU: the table   -> table_mlgdf/outputs/full/
bash experiments/mnist/figures/run.sh full       # the figures      -> figures/outputs/full/
python experiments/mnist/export_results.py       # copy both into the committed results/
```

The first full run downloads the two networks and trains the classifier (a few minutes). Each `run.sh` also has a
`smoke` tier: seconds on CPU with an untrained prior, to check the code path only. On SLURM:
`sbatch --gres=gpu:1 --wrap "bash experiments/mnist/table_mlgdf/run.sh full"`.

The MNIST code is the package `cdm.models.mnist`; `cdm/models/mnist/guidance.py` builds the `GuidanceConfig` for
the shared optimizer `cdm.guidance.core.optimize_guided`.
