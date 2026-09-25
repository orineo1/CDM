# `sdxl/` — the Stable Diffusion experiment's own code

SD-specific code, copied in so the repo is self-contained. Nothing here points at
another repository at runtime.

Ported from `orineo1/conditional-matching-paper`, branch
`claude/sd-shared-controlnet-hvt-uniform`, commit `cb8bc91`, path
`SD_cond_SD_controlnet/`.

## What is SD-specific (carried here)

| file | what |
|---|---|
| `models.py` | loads the Architect (SDXL Base, DDIM) and Sprinter (SDXL Turbo + ControlNet-Scribble) pipelines |
| `generation.py` | Sprinter forward passes, `pred_x0`, the DDIM step, and the guided step `run_dps_step_clip` |
| `clip_utils.py` | CLIP ViT-L/14 loading and a **differentiable** image encode (resize + normalise + ViT) |
| `image_utils.py` | Sobel proxy, latent→PIL decode, the base conditioning image |
| `hed.py` | HED scribble extraction (`lllyasviel/Annotators`) and weighted scribble averaging |
| `metrics.py` | SWD + the full decode→generate→CLIP→MMD evaluation helper; re-exports the MMD seam |
| `mmd.py` | **the seam to the shared estimator — see below** |
| `analysis.py`, `visualization.py` | scribble-diff heatmaps and step plots |
| `run_mlgd_f.py`, `eval_baselines.py` | the two entry points |

SDXL does **not** use `models/DiffusionModel`. It uses Stable Diffusion's own
pipelines, which is correct and expected for this experiment.

## What is CALLED from the shared packages, not duplicated

`sdxl/mmd.py` is the whole point. The **estimator** — the unbiased U-statistic —
is `losses.functions.MMDLoss` and is called, never copied. What stays SD-specific is
only what genuinely differs:

- **the kernel.** SD uses a single median-heuristic bandwidth with a generalised
  exponent, `exp(-(||a-b||²/2bw²)^alpha)`. The shared `losses.RBF` is a 5-bandwidth
  mean-based mixture. `MMDLoss` accepts an injected kernel, which is the seam used.
- **the sqrt wrapper.** SD reports `sqrt(|MMD²| + 1e-8)`. `mmd_report` does that, for
  display only; the loss and its gradient use the plain squared statistic.

SWD is *not* aliased onto the shared one: `losses.functions.sliced_wasserstein_distance`
is a different estimator with a different signature and return type, so SD's is carried
in `metrics.py`.

## Check

```
python -m sdxl.check_mmd_equivalence
```

CPU-only, seconds, no model downloads. Asserts that routing through the shared
`MMDLoss` reproduces the upstream vendored formula to 1e-6, that the estimator is
unbiased (`MMD²(X,X) < 0`), and that gradients flow through the generated samples
while the target stays detached.

## Dependencies

`sdxl/requirements.txt`. A GPU is required to run anything that loads the pipelines.
