# `cdm/models/sd/` — the Stable Diffusion experiment's own code

SD-specific code, carried here so the repository is self-contained. Nothing here points
at another repository at runtime.

Source: `orineo1/conditional-matching-paper`, branch
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
| `mmd.py` | thin alias onto the shared MMD, plus the bandwidth helper and witness scores |
| `visualization.py` | the per-step plot, preview rows, the scribble-diff heatmap |
| `interpolation.py` | the interpolation figures' pieces: target on a scribble, gender / age axis, sorted grid, PCA scatter |
| `run_mlgd_f.py` | entry point: the MLGD-F optimisation (`run.sh smoke|full`) |
| `eval_baselines.py` | entry point: the baseline scribbles, `sdedit_best` and `avg_prompt` (`run.sh baselines`) |
| `eval_all_experiments.py` | entry point: the paper's N = 2,000 evaluation, MMD / SWD / p(male) (`run.sh eval`) |
| `saliency.py` | entry point: where MLGD-F edited the scribble and why there: scribble change, gender saliency, their correlation, the latent view (Figs. 7 and 8; `run.sh saliency`) |
| `step_cost.py` | entry point: memory and time of one guidance step, Turbo measured vs Base projected, All vs Uniform selection (ablation; `step_cost/run.sh`) |
| `check_imports.py`, `check_mmd_equivalence.py` | the CPU checks (`run.sh check`, `tests/test_sd.py`) |

`models.py` also holds `make_deterministic` (seeds + deterministic kernels) and the pinned
model revisions' resolution; `hed.py` the pinned HED download.

SDXL does **not** use `cdm.models.DiffusionModel`. It uses Stable Diffusion's own
pipelines, which is correct and expected for this experiment.

## What is CALLED from the shared packages, not duplicated

**MMD is not implemented here.** `losses.functions.compute_mmd` is the paper repo's
`SD_cond_SD_controlnet/src/metrics.py::compute_mmd` — SD's own kernel
(single median-heuristic bandwidth, generalised exponent `exp(-(||a-b||²/2bw²)^alpha)`)
and SD's own `sqrt(|MMD²| + 1e-8)` return are part of that function, so there is nothing
SD-specific left to build. `cdm/models/sd/mmd.py::sd_mmd` is a thin alias that keeps the name the
pipeline code uses; `mmd_report` is a pass-through onto it. `estimate_bandwidth` and
`compute_witness_scores` live here.

SWD is *not* aliased onto the shared one: `losses.functions.sliced_wasserstein_distance`
is a different estimator with a different signature and return type, so SD's own lives in
`metrics.py`.

## Check

```
python -m cdm.models.sd.check_mmd_equivalence
```

CPU-only, seconds, no model downloads. Asserts that the shared `compute_mmd`
reproduces the upstream formula to 1e-6 and that gradients flow through the generated
samples while the target stays detached.

## Dependencies

`experiments/stable_diffusion/requirements.txt`. A GPU is required to run anything that loads
the pipelines.
