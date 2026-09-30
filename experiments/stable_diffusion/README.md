# `experiments/stable_diffusion/`

The Stable Diffusion experiment. One directory per experiment, each with `README.md`
(what it produces, the command, runtime, GPU), `run.sh` (explicit configurations, every
flag literal), `config/`, `results/` (the paper's reference numbers, **imported** with provenance, and the
exported runs of this repository, kept in separate files) and `outputs/` (local runs,
gitignored).

| directory | paper artifact | entry point |
|---|---|---|
| `four_targets` | Table 3, Table 15 | `cdm.models.sd.run_mlgd_f` (optimisation), `cdm.models.sd.eval_all_experiments` (the N = 2,000 evaluation), `cdm.models.sd.eval_baselines` |
| `interpolation` | the gender / age interpolation figures: portraits sorted along the target axis, PCA scatter | `interpolation.ipynb` over `cdm.models.sd.interpolation` |
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

## Running it (cluster, W&B)

Everything below is run from the repository root. `four_targets/README.md` has the
flags, the config files and what each run saves.

**1. Environment, once.** A venv with CUDA torch and the SD packages:

```bash
python3 -m venv /path/to/venvs/cdm && source /path/to/venvs/cdm/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt && pip install -e .
pip install -r experiments/stable_diffusion/requirements.txt
```

**2. Weights & Biases, once.** Every run logs its config, the per-step MMD / gradient
norm / ζ, the per-step visualisation, the input and final scribbles and the final MMD to
W&B (project `sdxl-smoke` for smoke runs, `sdxl-mlgdf` for full runs). Log in with the
key from https://wandb.ai/authorize; `wandb login` stores it in `~/.netrc`, or keep it
out of `$HOME`:

```bash
wandb login                                                     # or:
export WANDB_API_KEY=$(cat /path/to/.wandb_key)                 # sbatch passes exported variables to the job
```

**3. Where things go.** `run.sh` writes nothing to `$HOME`: model weights (tens of GB)
and every cache go to `<repo>/.cache/` (`CACHE_DIR=/path` or `HF_HOME=/path` move
them), W&B run files to `four_targets/outputs/wandb/`, results to
`four_targets/outputs/<tier>_<Scenario>/`.

**4. Check, then a smoke run, then the paper configuration.** Submit `run.sh` itself
with `sbatch --wrap`; partition, GPU and time go on the `sbatch` line. The full
configuration needs a large GPU (the paper used an L40S, 48 GB) and hours (about 4 h in
the paper; an L4 is slower).

```bash
bash experiments/stable_diffusion/four_targets/run.sh check          # CPU, seconds: imports + MMD equivalence
mkdir -p logs
sbatch --partition=salmon --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
       --output=logs/sd-%j.out \
       --wrap "bash experiments/stable_diffusion/four_targets/run.sh smoke balanced"
sbatch --partition=salmon --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
       --output=logs/sd-%j.out \
       --wrap "bash experiments/stable_diffusion/four_targets/run.sh full balanced"
```

Scenarios: `balanced`, `skewed`, `gender_interpolation`, `age_interpolation`, or `all`
(with `--array=0-3` one scenario per task). Defaults come from
`four_targets/config/<scenario>.env`; any `run_mlgd_f` flag after the scenario overrides
them, e.g. `run.sh full skewed --base_zeta 5 --seed 2`. `DRY_RUN=1 bash .../run.sh ...`
prints the command without running it. Watch a job with `squeue -u $USER` and
`tail -f logs/sd-<jobid>.out`; the W&B link is printed near the top of the log.

**5. Evaluate as the paper does (Table 3 / 15).** About an hour per scenario:

```bash
sbatch --partition=salmon --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 \
       --output=logs/sd-eval-%j.out \
       --wrap "bash experiments/stable_diffusion/four_targets/run.sh eval balanced"
```

**6. Keep the results.** When the runs and evaluations are done:

```bash
python experiments/stable_diffusion/export_results.py                # outputs/full_*, eval_*, interpolation -> results/
```

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

Ported: the 2,000-image evaluation (`eval_all_experiments.ipynb` ->
`cdm/models/sd/eval_all_experiments.py`, `four_targets/run.sh eval`) and the interpolation
notebooks (`eval_scribbl_interpolation*.ipynb` -> `interpolation/`, without their MMD).

`cdm/models/sd/analysis.py` and `cdm/models/sd/visualization.py` carry the scribble-diff heatmap and the
step plots, but the saliency computation itself is not among them.
