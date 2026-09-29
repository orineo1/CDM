# MNIST: rotated digits

The paper's rotated-digits experiment (Section 3.2). `X` is a 28x28 MNIST image and
`Y` its rotation angle. The prior `P(X)` is an unconditional DDPM over randomly
rotated digits; the conditional sampler `P(Y|X)` is a consistency (iCT) model that
samples the angle of an image. MLGD-F searches for an image whose angle
distribution matches a target `G` on the circle: uniform, bimodal (modes at 0 and
180 degrees) or unimodal (at 0 degrees). A rotation-robust classifier then reads off
which digit was found.

| folder | what it holds |
|---|---|
| `table_mlgdf/` | Table 2: 3 targets x 15 seeds; the settings, the runner, and `results/` |
| `figures/` | the digit and angle-histogram figures drawn from Table 2's runs, and `results/` |
| `train/` | training of the two networks (not needed to reproduce the results) |
| `checkpoints/` | where the networks and the classifier are placed; not in git (see its README) |
| `export_results.py` | copies a finished run into the committed `results/` folders |

## Results

The results of the last full run are committed and can be read without running
anything: `table_mlgdf/results/` (table, per-seed CSV, images, run info) and
`figures/results/` (three figures per target). The run used the settings in
`table_mlgdf/config/targets.tsv`, 15 seeds per target, the classifier trained by the
code, threshold 0.8, on an NVIDIA L4 with torch 2.11 (see `results/run_info.txt`).

| Target | SWD_all | SWD_top5 | Time (s) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | None |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 0.049 ± 0.043 | 0.029 ± 0.006 | 18.0 ± 0.1 | 91 | -- | -- | -- | -- | -- | 9 | -- | -- | -- | 4 |
| bimodal | 0.081 ± 0.066 | 0.034 ± 0.008 | 8.0 ± 0.1 | 18 | 18 | -- | -- | -- | 9 | 18 | -- | 27 | 9 | 4 |
| unimodal | 0.050 ± 0.016 | 0.034 ± 0.008 | 8.1 ± 0.1 | -- | -- | 38 | 31 | -- | -- | -- | 23 | 8 | -- | 2 |

Digit columns: % of the classified seeds; None: seeds below the classifier threshold
(excluded from the %). SWD is mean ± std over all 15 seeds and over the 5 best.

## Reproducing the results

From the repository root, after `pip install -r requirements.txt && pip install -e .`:

```bash
bash experiments/mnist/table_mlgdf/run.sh full   # GPU; Table 2 -> table_mlgdf/outputs/full/
bash experiments/mnist/figures/run.sh full       # figures   -> figures/outputs/full/
python experiments/mnist/export_results.py       # copy both into the committed results/ folders
```

On the first full run the two networks are downloaded and the classifier is trained
(a few minutes; see `checkpoints/README.md`). Each `run.sh` also has a `smoke` tier:
a few seconds on CPU, with an untrained prior, to check the code path. It produces
no result.

On a SLURM cluster, run each step as a job on a GPU node, in order, e.g.
`sbatch --gres=gpu:1 --wrap "bash experiments/mnist/table_mlgdf/run.sh full"`.

## Code

The MNIST-specific code is the package `cdm.models.mnist`: the angle model, the UNet
prior, the classifier, the datasets, the three targets and the plotting.
`cdm/models/mnist/guidance.py` builds the `GuidanceConfig` (DDIM step, perturbation,
step-size rule zeta_t, SWD loss, target sampler) and hands it to the shared optimiser
`cdm.guidance.core.optimize_guided`. The optimisation is bit-for-bit identical to the
original implementation in `orineo1/conditional-matching-paper`.
