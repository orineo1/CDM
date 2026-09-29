# MNIST rotated-digits experiment

The paper's Section 5.2 / Appendix C experiment: the prior `P(X)` is an
unconditional DDPM over randomly rotated MNIST digits, the conditional sampler
`P(Y|X)` is an iCT model that samples the rotation angle of an image, and
MLGD-F searches for an image whose induced angle distribution matches a target
`G` on the circle.

| directory | produces |
|---|---|
| `table_mlgdf/` | Table 2 -- 3 targets x 15 seeds |
| `figures/` | Figure 5, and Figures 6-8 (all seeds per target) |
| `train/` | the two frozen networks (Tables 11 and 12) |
| `cdm/models/mnist/` | the MNIST-specific code all three import (not an experiment) |

Each experiment directory has the same shape: `README.md`, `run.sh` with a
reduced `smoke` configuration and the full one, `config/`, `results/` (imported
reference numbers) and `outputs/` (whatever a local run wrote -- never a
result).

## What is MNIST-specific and what is shared

`cdm/models/mnist/` holds only what is specific to this task: the two rotated-MNIST
datasets, the two classifiers, the conditional iCT angle model, the UNet prior
configuration, the three angle targets, and the figure code.

The method is not duplicated there. It is imported from the repo's own packages:

| need | comes from |
|---|---|
| guided optimisation loop | `guidance.core.optimize_guided` + `guidance.config.GuidanceConfig` |
| sliced-Wasserstein distance | `losses.sliced.SupportAwareSlicedWassersteinLoss` |
| angle <-> (cos, sin), circular MoG density | `losses.circular` |
| mixture sampling | `utils.dist.generate_mog_samples_not_differentiable` |

`cdm/models/mnist/guidance.py` is the seam: it builds the `GuidanceConfig` -- the DDIM
step against a diffusers scheduler, the Tweedie perturbation scale, the zeta_t
rule, the clamped update and the target generator -- and then hands it to the
shared optimiser.

## Checkpoints

Both frozen networks are downloaded from the HuggingFace repo named in the
paper's appendix (`anon-submission-cdm/cdm-inverse-design`, subfolder `MNIST`),
unless a local copy is found first. `experiments/mnist/checkpoints/` in this repo already holds
`MnistConditional500Epoch.pt`; the unconditional `MnistUncond100Epoch.pth` is
not in the repo and is fetched. Set `HF_TOKEN` if the repo is gated.

## Imports

The MNIST-specific code is the package `cdm.models.mnist` (the conditional iCT
angle model, the UNet prior, the rotation-robust classifier, the datasets, the
targets, plotting). The entry points here import it like any other part of `cdm`,
after `pip install -e .`.
