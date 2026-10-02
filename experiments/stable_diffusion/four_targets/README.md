# four_targets

**Produces:** Table 3 (`tab:sd_results`) and Table 15 (`tab:sd_swd`) — MLGD-F against four
distributional targets (balanced, skewed, gender interpolation, age interpolation),
with the Average-prompt and SDEdit-Best baselines.

| | |
|---|---|
| Entry points | `python -m cdm.models.sd.run_mlgd_f` (the optimisation), `python -m cdm.models.sd.eval_all_experiments` (the N = 2,000 evaluation), `python -m cdm.models.sd.eval_baselines` (the baseline scribbles) |
| Config | `config/<scenario>.env` — T=250, t_start=125, ζ=10.0, uniform selection at n_sel/n_cond = 0.5, ControlNet scale 0.5; `config/models.env` — the pinned model revisions |
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
bash run.sh eval   balanced                # GPU, ~1 h: the paper's N = 2,000 evaluation of the run's scribbles
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
the run's own directory, `outputs/<tier>_<Scenario>_seed<S>_<jobid>/` (see *Saved results*
below).

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

## Baselines: `run.sh baselines`

Builds the baseline scribbles of a finished run (`cdm/models/sd/eval_baselines.py`) into
`<run dir>/baselines/`, where `run.sh eval` finds them. The run directory is `--run_dir`,
or by default the newest finished `full` run of the scenario:

- **`sdedit_best`**: the best of N unguided SDEdit candidates of the source scribble. Each
  candidate is scored by MMD on 250 portraits against a search target built from the
  source scribble; N is chosen so the search takes as long as the MLGD-F run did (read
  from the run's `metrics.json`; `--lgd_cm_minutes` overrides, `--max_candidates` caps it
  for a quick check). The run's own seed, steps, `t_start`, ControlNet scale and target
  prompts are used (`run.sh` passes `config/<scenario>.env`; repeat any `run_mlgd_f` flag
  the run had, e.g. `--seed 3`).
- **`avg_prompt`**: the HED scribble of one portrait generated from the frac-weighted
  average of the groups' CLIP text embeddings (prompt interpolation).
- `--methods avg sdedit` adds the older pixel-average scribble and one guided SDEdit.

Every scribble, the source and MLGD-F ones included, gets `preview_<name>.png`: 5 portraits
under the neutral prompt. Also saved: `scribble_<method>.png`, `scribbles_all.png`,
`sdedit_search.png`, `target_<group>.png`, `baselines_meta.json` (budget, candidates, best
seed and MMD, every candidate's MMD). Logged to W&B (`sdxl-baselines`). It takes about as
long as the MLGD-F run.

```
bash run.sh baselines balanced
bash run.sh baselines balanced --run_dir experiments/stable_diffusion/four_targets/outputs/smoke_BalancedTarget_seed1_<id> --max_candidates 3   # quick check
```

The whole scenario is then three steps: `run.sh full balanced`, `run.sh baselines balanced`,
`run.sh eval balanced`.

## Running the whole pipeline on SLURM

The three steps as three jobs, each starting only after the previous one **finished
successfully** (`afterok`; `--kill-on-invalid-dep=yes` cancels the rest if one fails).
From the repo root, venv active, `WANDB_API_KEY` and `HF_HOME` exported:

```
G=$PWD; R=$G/experiments/stable_diffusion/four_targets; SEED=1
common="--partition=salmon --gres=gpu:1 --cpus-per-task=8 --mem=64G --chdir=$G --kill-on-invalid-dep=yes"
jf=$(sbatch --parsable $common --time=24:00:00 --output=$G/logs/sd-full-%j.out \
     --wrap "bash $R/run.sh full balanced --seed $SEED")
RUN=$R/outputs/full_BalancedTarget_seed${SEED}_${jf%%;*}
jb=$(sbatch --parsable $common --time=24:00:00 --output=$G/logs/sd-base-%j.out --dependency=afterok:${jf%%;*} \
     --wrap "bash $R/run.sh baselines balanced --run_dir $RUN")
je=$(sbatch --parsable $common --time=03:00:00 --output=$G/logs/sd-eval-%j.out --dependency=afterok:${jb%%;*} \
     --wrap "bash $R/run.sh eval balanced --run_dir $RUN")
echo "full $jf -> baselines $jb -> eval $je   run dir: $RUN"
```

`squeue -u $USER` shows the later two as `PENDING (Dependency)` until their turn.

## Final evaluation: `run.sh eval` (Table 3 / Table 15)

The optimisation's own numbers (the per-step MMD, the final MMD on 250 images against the
100-image target pool) are the optimisation's signal, not the paper's measurement. The
tables use the protocol of `cdm/models/sd/eval_all_experiments.py`, a port of the paper's
`eval_all_experiments.ipynb`:

1. a **2,000-image target** generated from the **source** scribble with the scenario's
   prompts and fractions (age: 40 ages × 50, seed 42), CLIP-encoded;
2. for every scribble found in the run directory (and its `baselines/`): **2,000 portraits** under the
   neutral prompt, then **MMD** and **SWD** against the target and, in the gender scenarios,
   **p(male)** by CLIP zero-shot with a binomial 95% CI; Δ% is the improvement over the
   source scribble.

Right after a run the scribbles are `source` and `mlgd_f`; `avg_prompt` and `sdedit_best`
are picked up from `baselines/` once `run.sh baselines` has produced them. Any other
scribble (e.g. the paper's older `avg`, `avg_latent`, `sdedit`, or `final_scribble_regular.png`
of a `--run_unguided` run) is evaluated only when passed with `--scribbles name=path`.
The paper's own scribbles (`SD_cond_SD_controlnet/experiments/<Scenario>/` in the paper
repository) can be evaluated with `--run_dir <that directory>`.

```
bash run.sh eval balanced                                           # outputs/full_BalancedTarget -> outputs/eval_BalancedTarget
bash run.sh eval balanced --n_eval 200 --run_dir experiments/stable_diffusion/four_targets/outputs/smoke_BalancedTarget_seed1_<id> --output_dir experiments/stable_diffusion/four_targets/outputs/eval_smoke   # quick check
bash run.sh eval all                                                # the four scenarios
```

It writes `eval_results.json`, `eval_table.md` (the Table 3 / 15 rows), `embeddings.npz`
(target and per-method CLIP embeddings, to recompute the metrics without a GPU),
`preview_<method>.png` and `target_preview_<group>.png` (5 images each), the MMD / SWD bars
and the p(male) CI plot, `run_info.txt` and `pip_freeze.txt`. **The 2,000 photos per method
are not saved.** Logged to W&B (`sdxl-eval`): the table, previews, plots and every number.
Deterministic and pinned like the run (`--seed 42` is the notebook's).

## Saved results

Every run gets its own directory, `outputs/<tier>_<Scenario>_seed<S>_<jobid>/` (not in git;
the seed is the config's or `--seed`, the id the SLURM job id or a timestamp), and
`run.sh baselines` / `run.sh eval` add `baselines/` and `eval/` inside it:

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

It writes `results/eval_table.md` (every run's N = 2,000 table) and, per run,
`results/<run>/` with only what the results are read from, each file once:

| file | contents |
|---|---|
| `metrics.json`, `run_info.txt`, `curves.png` | the run's record (args, every step, model revisions, W&B link), the exact command, the per-step curves |
| `scribble_source.png`, `source_portrait.png`, `final_scribble_mlgd_f.png` | the source scribble and portrait, the MLGD-F scribble |
| `baselines/` | `scribble_avg_prompt.png`, `scribble_sdedit_best.png`, `avg_prompt_portrait.png`, `sdedit_search.png`, `baselines_meta.json`, `run_info.txt` |
| `eval/` | `eval_results.json` (every number), `run_info.txt`, `preview_<method>.png` and `target_preview_<group>.png` (5 portraits each) |

`<run>` is the run directory's name; to publish a run without its job id, point
`--run-dir` at a folder of symlinks named e.g. `full_BalancedTarget_seed4`. Everything else
(`steps.csv`, also inside `metrics.json`; photo grids; target galleries; the bar and CI
plots of numbers that are in `eval_table.md`; `npy/`; embeddings) stays in `outputs/`.

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
The weights are frozen: `config/models.env` pins every model to a hub commit, and
`run.sh` passes them as `--architect_revision`, `--sprinter_revision`,
`--controlnet_revision`, `--hed_revision`, `--clip_revision`. To move to newer weights,
change the sha there (or remove the line to follow the hub's `main`).

**Checked** on 2026-09-30, NVIDIA L40S, torch 2.11 (cu128): two `smoke balanced` runs in one
job gave identical per-step values (`steps.csv`), the same final MMD and pixel-identical
scribbles. To re-check, run the same configuration twice with two `--output_dir`s in one
job and `diff` the two `steps.csv` (ignoring `elapsed_s`) and `cmp` the two
`final_scribble_mlgd_f.png`. Any `UserWarning` naming an operation without a deterministic
implementation is printed during the run (`warn_only`); if one appears, that operation is
the remaining source of drift.

## The paper's runs

The paper reports one run per scenario, all with `base_zeta` 7.5; `results/` holds exactly
these four (`results/eval_table.md` has their N = 2,000 tables):

| scenario | `results/` | seed | SDEdit budget (min) |
|---|---|---|---|
| balanced | `full_BalancedTarget_seed4` | 4 | 150 |
| skewed | `full_SkewedTarget_seed6` | 6 | 150 |
| gender_interpolation | `full_GenderInterpolation_seed8` | 8 | 150 |
| age_interpolation | `full_AgeInterpolation_seed2` (source portrait: age 40) | 2 | 178 |

To reproduce one (GPU: NVIDIA L40S; the weights are pinned by `config/models.env`):

```
bash experiments/stable_diffusion/four_targets/run.sh full balanced --seed 4 --base_zeta 7.5
bash experiments/stable_diffusion/four_targets/run.sh baselines balanced --run_dir <run> --lgd_cm_minutes 150
bash experiments/stable_diffusion/four_targets/run.sh eval balanced --run_dir <run>
```

and likewise `skewed --seed 6`, `gender_interpolation --seed 8`, and
`age_interpolation --seed 2 --base_zeta 7.5 --age_source min` (budget 178). The other
settings are `config/<scenario>.env`; each run's exact commands are in its `run_info.txt`,
`baselines/run_info.txt` and `eval/run_info.txt`.

The code in `cdm/models/sd/` is ported from `orineo1/conditional-matching-paper` @ `cb8bc91`
(branch `claude/sd-shared-controlnet-hvt-uniform`, `SD_cond_SD_controlnet/`);
`cdm/models/sd/README.md` lists what is carried there and what is shared.

`outputs/` is for local runs and ships empty; `export_results.py` is what copies a run
into `results/`.
