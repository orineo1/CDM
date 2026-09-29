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

The results of the last full run are committed: `table_mlgdf/results/` (table,
per-seed CSV, images, run info) and `figures/results/` (figures). They can be read
without running anything.

## Reproducing the results

From the repository root, after `pip install -r requirements.txt && pip install -e .`:

```bash
bash experiments/mnist/table_mlgdf/run.sh full   # GPU; Table 2 -> table_mlgdf/outputs/full/
bash experiments/mnist/figures/run.sh full       # figures   -> figures/outputs/full/
python experiments/mnist/export_results.py       # copy both into the committed results/ folders
```

Each `run.sh` also has a `smoke` tier: a few seconds on CPU, with an untrained prior,
to check the code path. It produces no result.

## Code

The MNIST-specific code is the package `cdm.models.mnist`: the angle model, the UNet
prior, the classifier, the datasets, the three targets and the plotting.
`cdm/models/mnist/guidance.py` builds the `GuidanceConfig` (DDIM step, perturbation,
step-size rule zeta_t, SWD loss, target sampler) and hands it to the shared optimiser
`cdm.guidance.core.optimize_guided`. The optimisation is bit-for-bit identical to the
original implementation in `orineo1/conditional-matching-paper`.
