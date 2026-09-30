# `experiments/stable_diffusion/`

The Stable Diffusion experiment. One directory per experiment, each with `README.md`
(what it produces, the command, runtime, GPU), `run.sh` (explicit configurations, every
flag literal), `config/`, `results/` (the paper's reference numbers, **imported** with provenance, and the
exported runs of this repository, kept in separate files) and `outputs/` (local runs,
gitignored).

| directory | paper artifact | entry point |
|---|---|---|
| `four_targets` | Table 3, Table 15 | `cdm.models.sd.run_mlgd_f`, `cdm.models.sd.eval_baselines` |
| `jacobian_error` | theory appendix, Jacobian-fidelity figure and table | `eps_g_experiment.ipynb` (self-contained) |

**The code lives in the `cdm.models.sd` package**, taken from
`orineo1/conditional-matching-paper` @ `cb8bc91` so this repository is self-contained —
nothing points at another repo at runtime. `cdm/models/sd/README.md` says what is carried there and
what is called from the shared packages; in short, the SDXL + ControlNet-Scribble
pipeline, HED extraction, SDEdit init, target construction, CLIP evaluation and the
baselines are SD-specific and carried here, while the **MMD estimator is the shared
`cdm.losses.functions.compute_mmd`** and is called, not duplicated.

The guided loop is Algorithm 1 implemented inside the `diffusers` pipeline
(`generation.py`): SDEdit initialisation, the DPS step-size normalisation and
CLIP-space guidance. It does not use `cdm.guidance.optimize_guided` or
`cdm.models.DiffusionModel`; the shared part is the inner estimator.

## Requirements

This experiment needs a CUDA GPU — the paper's runs used an L40S (48 GB) for the
optimisation and an A100 (80 GB) for the 2,000-image evaluation — plus network access to
download SDXL Base, SDXL Turbo, ControlNet-Scribble and CLIP ViT-L/14, and the SD
dependencies in `cdm/models/sd/requirements.txt` (`transformers`, `diffusers`, `peft`,
`controlnet_aux`, `wandb`). A CPU-only `check` configuration that needs none of that is
described in the experiment's README.

In `four_targets/results/`, `table3_sd_results.md` and `table15_sd_swd.md` are transcribed
from the paper; `mlgdf_runs.md` and the `<Scenario>/` folders are runs of this code,
written by `export_results.py` (see below).

## Saving results

As for MNIST: `run.sh full` writes everything to `four_targets/outputs/` (not in git),
then

```bash
python experiments/stable_diffusion/export_results.py   # outputs/full_* -> four_targets/results/
```

copies the finished runs into the committed `four_targets/results/` (table, per-step CSV,
images, run info). The per-step visualisations are in wandb only.

## Not covered here

These parts of the SD experiment have **no entry point in this repository**. The upstream
code for them is in notebooks under `SD_cond_SD_controlnet/notebooks/` at the same commit:

| artifact | upstream notebook |
|---|---|
| Fig. 10, saliency + correlation | `gender_saliency_eval.ipynb` |
| Table 17 step timing; memory ablation | `measure_dps_step_memory.ipynb` |
| Table 18, Fig. 20 Jacobian fidelity | `eps_g_experiment.ipynb` |
| the 2,000-image evaluation protocol | `eval_all_experiments.ipynb` |

`cdm/models/sd/analysis.py` and `cdm/models/sd/visualization.py` carry the scribble-diff heatmap and the
step plots, but the saliency computation itself is not among them.
