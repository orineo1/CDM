# `cdm/models/`

The frozen networks the method uses; the search itself is in `cdm/guidance/`.

| path | contents |
|---|---|
| `diffusion.py` | `DiffusionModel`, a timestep-indexed DDPM/DDIM model: the prior `P(X)` (`condition=False`) and the conditional diffusion sampler `P(Y\|X)` of MLGD (`condition=True`) |
| `priors/` | re-exports `DiffusionModel` as the prior |
| `samplers/consistency.py` | consistency-model samplers `P(Y\|X)` (MLGD-F) |
| `samplers/exact.py` | `ExactConditionalSampler`: exact draws from `P(Y\|X)` of a known Gaussian mixture (an oracle) |
| `mnist/` | the MNIST experiment: networks, datasets, targets, figures |
| `sd/` | the Stable Diffusion experiment (see its README) |

The search calls only `model_cond.sample(nsamples=..., condition_x=..., device=...)`, which every conditional
sampler implements. Checkpoints written by `experiments/synthetic/pipeline/train_models.py` store an
`"architecture"` key: `"diffusion"` (`DiffusionModel`) or `"cm"` (`ConsistencyModeliCT`).
