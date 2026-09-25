# MNIST rotated-digits experiment

The paper's Section 5.2 / Appendix C experiment: the prior `P(X)` is an
unconditional DDPM over randomly rotated MNIST digits, the conditional sampler
`P(Y|X)` is an iCT model that samples the rotation angle of an image, and
MLGD-F searches for an image whose induced angle distribution matches a target
`G` on the circle.

| directory | produces |
|---|---|
| `table2_mlgdf/` | Table 2 -- 3 targets x 15 seeds |
| `figures/` | Figure 5, and Figures 6-8 (all seeds per target) |
| `model_training/` | the two frozen networks (Tables 11 and 12) |
| `mnist_cdm/` | the MNIST-specific code all three import (not an experiment) |

Each experiment directory has the same shape: `README.md`, `run.sh` with a
`smoke` and a `full` tier, `config/`, `results/` (imported reference numbers)
and `outputs/` (whatever a local run wrote -- never a result).

## What is MNIST-specific and what is shared

`mnist_cdm/` holds only what is specific to this task: the two rotated-MNIST
datasets, the two classifiers, the conditional iCT angle model, the UNet prior
configuration, the three angle targets, and the figure code.

The method is not duplicated there. It is imported from the repo's own packages:

| need | comes from |
|---|---|
| guided optimisation loop | `guidance.core.optimize_guided` + `guidance.config.GuidanceConfig` |
| sliced-Wasserstein distance | `losses.sliced.SupportAwareSlicedWassersteinLoss` |
| angle <-> (cos, sin), circular MoG density | `losses.circular` |
| mixture sampling | `utils.dist.generate_mog_samples_not_differentiable` |
| per-run records | `gcio.save_run` |

`mnist_cdm/guidance.py` is the seam: it builds the `GuidanceConfig` -- the DDIM
step against a diffusers scheduler, the Tweedie perturbation scale, the zeta_t
rule, the clamped update and the target generator -- and then hands it to the
shared optimiser.

## Checkpoints

Both frozen networks are downloaded from the HuggingFace repo named in the
paper's appendix (`anon-submission-cdm/cdm-inverse-design`, subfolder `MNIST`),
unless a local copy is found first. `MnistModels/` in this repo already holds
`MnistConditional500Epoch.pt`; the unconditional `MnistUncond100Epoch.pth` is
not in the repo and is fetched. Set `HF_TOKEN` if the repo is gated.

## Imports

There is no packaging config in this repo, so the entry-point scripts put the
repo root and this directory on `sys.path` themselves (`_path.py`), exactly as
`scripts/` does.
