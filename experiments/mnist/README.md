# MNIST rotated-digits experiment

The paper's rotated-digits experiment (Section 3.2 and the MNIST appendix): the prior `P(X)` is an
unconditional DDPM over randomly rotated MNIST digits, the conditional sampler
`P(Y|X)` is an iCT model that samples the rotation angle of an image, and
MLGD-F searches for an image whose induced angle distribution matches a target
`G` on the circle.

| directory | produces |
|---|---|
| `table_mlgdf/` | Table 2 -- 3 targets x 15 seeds |
| `figures/` | Figure 5, and Figures 6-8 (all seeds per target) |
| `train/` | the two frozen networks (Tables 11 and 12) |
| `checkpoints/` | the conditional checkpoint shipped with the repo; the prior is downloaded here on first use |

Each experiment directory has the same shape: `README.md`, `run.sh` with a
reduced `smoke` configuration and the full one, `config/`, `results/` (the
published numbers) and `outputs/` (whatever a local run wrote, gitignored).

## Reproducing Table 2

```bash
bash experiments/mnist/table_mlgdf/run.sh full     # GPU, ~15 min per target on an L40S
bash experiments/mnist/figures/run.sh full         # reads the run records; small GPU or CPU
```

Training the two networks (`train/run.sh full`) is not needed: the published
checkpoints are used.

## What is MNIST-specific and what is shared

`cdm/models/mnist/` holds only what is specific to this task: the two rotated-MNIST
datasets, the two classifiers, the conditional iCT angle model, the UNet prior
configuration, the three angle targets, and the figure code.

The method is not duplicated there. It is imported from the shared packages:

| need | comes from |
|---|---|
| guided optimisation loop | `cdm.guidance.core.optimize_guided` + `cdm.guidance.config.GuidanceConfig` |
| sliced-Wasserstein distance | `cdm.losses.sliced.SupportAwareSlicedWassersteinLoss` |
| angle <-> (cos, sin), circular MoG density | `cdm.losses.circular` |
| mixture sampling | `cdm.utils.dist.generate_mog_samples_not_differentiable` |

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
