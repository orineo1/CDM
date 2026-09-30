# four_targets

**Produces:** Table 3 (`tab:sd_results`) and Table 15 (`tab:sd_swd`) — MLGD-F against four
distributional targets (balanced, skewed, gender interpolation, age interpolation),
with the Average-prompt and SDEdit-Best baselines.

| | |
|---|---|
| Entry points | `python -m cdm.models.sd.run_mlgd_f`, `python -m cdm.models.sd.eval_baselines` |
| Config | `config/<scenario>.env` — T=250, t_start=125, ζ=10.0, uniform selection at n_sel/n_cond = 0.5, ControlNet scale 0.5 |
| Full configuration | the paper's settings; GPU: L40S 48 GB for the optimisation, A100 80 GB for the 2,000-image evaluation |
| Reduced configuration (`smoke`) | 6 guided steps (12-step schedule from step 6), 6 variations; GPU required, loads the same pipelines |
| Static check (`check`) | CPU, seconds, no model downloads |

## Run

```
bash run.sh check                          # CPU, no GPU, no downloads
bash run.sh smoke  balanced                # GPU; reduced configuration, for a quick run
bash run.sh full   balanced                # GPU; the full, published configuration
bash run.sh full   all                     # the four scenarios one after another
bash run.sh full   skewed --base_zeta 5 --seed 2   # override any default from the CLI
bash run.sh help                           # every flag run_mlgd_f accepts
```

Scenarios: `balanced`, `skewed`, `gender_interpolation`, `age_interpolation`, `all`.
Defaults come from `config/<scenario>.env`; any `run_mlgd_f` flag after the scenario
overrides them (it is passed last, and the last value wins). `DRY_RUN=1` prints the
command without running it. Set `PYTHON=/path/to/python` if the interpreter on `PATH` is
not the project one.

Every run logs to Weights & Biases (`sdxl-smoke` / `sdxl-mlgdf`, or `WANDB_PROJECT`): the
config, per-step MMD, gradient norm, ζ and correction norm, the per-step visualisation
(x_t, pred x₀, conditional portraits, CLIP PCA against the target), the input and final
scribbles and the final MMD. Runs are named `<tier>_<scenario>` (`WANDB_NAME` overrides).
The per-step visualisations go to W&B only. Everything else is also saved to
`outputs/<tier>_<Scenario>/` (see *Saved results* below).

On a SLURM cluster, submit `run.sh` itself with `sbatch --wrap`, from the repo root, with
the venv active (sbatch passes it and exported variables such as `HF_HOME` to the job):

```
mkdir -p logs
sbatch --partition=salmon --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --output=logs/sd-%j.out \
       --wrap "bash experiments/stable_diffusion/four_targets/run.sh smoke balanced"
sbatch --partition=salmon --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 --output=logs/sd-%A_%a.out \
       --array=0-3 --wrap "bash experiments/stable_diffusion/four_targets/run.sh full all"
```

Under `--array`, `all` runs scenario `SLURM_ARRAY_TASK_ID` (0 balanced, 1 skewed,
2 gender_interpolation, 3 age_interpolation). `run.sh` stops at once if no GPU is visible
and warns if there is no W&B login. On nodes without internet use `WANDB_MODE=offline`,
then `wandb sync experiments/stable_diffusion/four_targets/outputs/wandb/offline-run-*` from
a login node.

Nothing is written to `$HOME`: `run.sh` puts the model weights (~tens of GB) and the
matplotlib, torch and wandb caches in `<repo>/.cache/` (gitignored; `CACHE_DIR=/path` moves
them, and `HF_HOME` etc. set in the environment are kept), and the wandb run files in
`outputs/wandb/`.

The `smoke` configuration reproduces no published number; its output is not a result.

## Requirements

Both the `smoke` and the `full` configuration load SDXL Base, SDXL Turbo,
ControlNet-Scribble and CLIP ViT-L/14 and backpropagate through the whole chain, so they
need a CUDA GPU (the paper used an L40S, 48 GB) and network access for the model
downloads. The SD dependencies are in `experiments/stable_diffusion/requirements.txt` (`transformers`,
`diffusers`, `peft`, `controlnet_aux`, `wandb`).

`bash run.sh check` needs none of that. It runs two CPU-only checks in seconds:

- `cdm/models/sd/check_imports.py` — every intra-package import in `cdm/models/sd/` resolves and all 13
  modules byte-compile;
- `cdm/models/sd/check_mmd_equivalence.py` — the SD MMD, which is the **shared**
  `losses.functions.compute_mmd`, reproduces the upstream formula to 1e-6, and gradients
  flow through the generated samples while the target stays detached.

## Saved results

A run writes to `outputs/<tier>_<Scenario>/` (not in git):

| file | contents |
|---|---|
| `steps.csv` | one row per guided step, written as the run goes: MMD, gradient norm, ζ, loss norm, correction norm, elapsed time (plus unguided MMD with `--run_unguided`, Adam stats with `--use_adam`) — the per-step values logged to W&B |
| `metrics.json` | all args, every step, final MMD (n=250), optimisation time, the W&B run name / id / url |
| `run_info.txt` | date, commit, host, SLURM job, GPU, config and overrides, the exact command |
| `scribble.png`, `source_portrait.png`, `final_scribble_mlgd_f.png` | the input and output scribbles and the source portrait |
| `final_photos_mlgd_f.png`, `photos_mlgd_f/` | portraits generated from the final scribble (250) |
| `target_clip_pca.png`, `target_samples_*.png`, `targets_*/` | the target distribution |
| `npy/` | the images as arrays |

Then, as for MNIST, copy the finished runs into the committed `results/`:

```
python experiments/stable_diffusion/export_results.py                  # outputs/full_* -> results/
python experiments/stable_diffusion/export_results.py --tier smoke --dest /tmp/x   # try it on smoke runs
```

This writes `results/mlgdf_runs.md` (one row per scenario: final MMD next to the paper's,
unguided MMD, steps, settings, time, commit, W&B link) and `results/<Scenario>/` with
`metrics.json`, `steps.csv`, `run_info.txt`, `curves.png` (the per-step curves),
the scribbles (`scribble_source.png` is the name `eval_baselines` expects), the source
portrait, the final photos and the target plots. The large arrays stay in `outputs/`.

## Reproducibility

Every random draw is seeded from `--seed` (`SEED` in `config/<scenario>.env`), and
`run_mlgd_f` also forces deterministic GPU kernels (`make_deterministic` in
`cdm/models/sd/models.py`: deterministic cuDNN, no autotuning, deterministic cuBLAS
workspace, `torch.use_deterministic_algorithms`, math-kernel attention). The guidance
backpropagates through the Sprinter, the VAE and CLIP at every step, and without these
settings the gradient differs in its last bits between two identical runs, which over
the guided steps grows into a different scribble. `--no_deterministic` turns the kernel
settings off (faster, not repeatable).

The per-step plots and the unguided comparison draw their noise from their own
generators, so logging more or less often, or with another `num_cond`, does not change
the guided trajectory.

What a run reproduces exactly: **the same GPU model, the same library versions, the same
model revisions and the same seed give the same numbers and images.** Another GPU or
another `torch`/`diffusers` gives different floating-point results, and the guided
trajectory amplifies them, so across machines only the statistics are comparable (run
several seeds). Every run therefore records what produced it: `run_info.txt` (GPU,
versions, exact command), `pip_freeze.txt` (the environment) and
`metrics.json["model_revisions"]` (the hub commit sha of each model actually loaded).
To freeze the weights for good, pass those shas back as `--architect_revision`,
`--sprinter_revision`, `--controlnet_revision`, `--hed_revision`, `--clip_revision`
(or add them to the config).

**Checking it.** Run the same configuration twice in one job (so both land on the same
GPU) and compare; `compare_runs.py` exits 0 only when every step, the final MMD and the
images are identical:

```
R=experiments/stable_diffusion/four_targets
sbatch --partition=salmon --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 --output=logs/det-%j.out \
  --wrap "bash $R/run.sh smoke balanced --output_dir $R/outputs/det_a && \
          bash $R/run.sh smoke balanced --output_dir $R/outputs/det_b && \
          python experiments/stable_diffusion/compare_runs.py $R/outputs/det_a $R/outputs/det_b"
```

The log ends with `RESULT: IDENTICAL` or a per-column report of where the runs diverge.
Any `UserWarning` naming an operation without a deterministic implementation is printed
during the run (`warn_only`); if one appears, that operation is the remaining source of
drift.

## Reference numbers

`results/table3_sd_results.md`, `results/table15_sd_swd.md` — transcribed from the paper.
Provenance in `results/PROVENANCE.md`.

`outputs/` is for local runs and ships empty; `export_results.py` is what copies a run
into `results/`.
