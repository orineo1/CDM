# `cdm/models/sd/`

The Stable Diffusion experiment's code. Run it through `experiments/stable_diffusion/` (see its README).

| file | contents |
|---|---|
| `models.py` | loads the Architect (SDXL Base, DDIM) and the Sprinter (SDXL Turbo + ControlNet-Scribble); pinned revisions; deterministic kernels |
| `generation.py` | Sprinter sampling, `pred_x0`, the DDIM step and the guided step `run_dps_step_clip` |
| `clip_utils.py`, `image_utils.py`, `hed.py` | differentiable CLIP encoding, image helpers, HED scribble extraction |
| `metrics.py`, `mmd.py` | SWD and evaluation helpers; the MMD is the shared `cdm.losses.functions.compute_mmd` |
| `visualization.py`, `interpolation.py` | per-step plots; the interpolation figures |
| `run_mlgd_f.py` | entry point: the MLGD-F optimization |
| `eval_baselines.py` | entry point: the Average-prompt and SDEdit-Best baseline scribbles |
| `eval_all_experiments.py` | entry point: the N = 2,000 evaluation (MMD, SWD, p(male)) |
| `saliency.py` | entry point: where MLGD-F edits the scribble, and gradient saliency |
| `step_cost.py` | entry point: memory and time of one guidance step |
| `check_imports.py`, `check_mmd_equivalence.py` | CPU checks, no downloads (`four_targets/run.sh check`) |

The guided loop runs inside the `diffusers` pipelines, so it does not use `cdm.guidance` or `cdm.models.DiffusionModel`;
the inner estimator (the MMD) is shared.
