# Stable Diffusion: sketch-to-portrait

`X` is the SDXL VAE latent of a face scribble, and `Y` the CLIP ViT-L/14 embedding of a portrait generated from it
by SDXL Turbo + ControlNet-Scribble (2 steps). MLGD-F edits the scribble so that its portraits follow a target
distribution: balanced (50/50) or skewed (25/75) gender, gender interpolation, or age interpolation (40–79).

| directory | results in the paper | entry point |
|---|---|---|
| [`four_targets/`](four_targets/README.md) | the results table (MMD, p(male)) against the Average-prompt and SDEdit-Best baselines; MMD and SWD for every target; qualitative examples | `four_targets/run.sh full`, `baselines`, `eval` |
| [`four_targets/`](four_targets/README.md#saliency-where-mlgd-f-edits-the-scribble) (saliency) | where MLGD-F edits the scribble, gradient saliency and their correlation (balanced target); results in `four_targets/results/full_BalancedTarget_seed4/saliency/` | `four_targets/run.sh saliency` |
| [`interpolation/`](interpolation/README.md) | the gender and age interpolation grids and CLIP projections | `interpolation/interpolation.ipynb` |
| [`step_cost/`](step_cost/README.md) | memory and time of SDXL Turbo vs Base in the inner loop; All vs Uniform selection | `step_cost/run.sh` |
| [`jacobian_error/`](jacobian_error/README.md) | the Jacobian error of SDXL-Lightning vs SDXL Base | `jacobian_error/eps_g_experiment.ipynb` |

The code is the package `cdm.models.sd`; the MMD is the shared `cdm.losses.functions.compute_mmd`.

## Setup

A CUDA GPU (the committed runs: NVIDIA L40S, 48 GB), network access for the model downloads, and, from the
repository root:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt && pip install -e .
pip install -r experiments/stable_diffusion/requirements.txt
wandb login                  # or export WANDB_API_KEY=...; or WANDB_MODE=offline
```

Every run logs to Weights & Biases and saves its outputs in its own directory under `<experiment>/outputs/`
(git-ignored). Model weights and caches go to `<repo>/.cache/` (`CACHE_DIR=/path` moves them), not to `$HOME`.
`bash experiments/stable_diffusion/four_targets/run.sh check` runs the CPU checks (seconds, no downloads).
`python experiments/stable_diffusion/export_results.py` copies finished runs into the committed `results/` folders.

The runs use deterministic GPU kernels, so they repeat exactly but take about twice as long. Each run records its
own run time; see [`four_targets/README.md`](four_targets/README.md#run-time), "Run time".
