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
Local copies and `run_info.txt` (commit, GPU, exact command) go to
`outputs/<tier>_<Scenario>/`.

On a SLURM cluster, `slurm_sd.sh` takes the same arguments; partition, GPU and time go on
the `sbatch` line. Run from the repo root:

```
mkdir -p logs
sbatch --partition=<p> experiments/stable_diffusion/four_targets/slurm_sd.sh smoke balanced
sbatch --partition=<p> --array=0-3 experiments/stable_diffusion/four_targets/slurm_sd.sh full all
sbatch --partition=<p> experiments/stable_diffusion/four_targets/slurm_sd.sh full skewed --base_zeta 5
```

`VENV=/path/to/venv` picks the environment, `HF_HOME` the model cache, and
`WANDB_MODE=offline` (then `wandb sync wandb/offline-run-*`) covers nodes without internet.

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

## Reference numbers

`results/table3_sd_results.md`, `results/table15_sd_swd.md` — transcribed from the paper.
Provenance in `results/PROVENANCE.md`.

`outputs/` is for local runs and ships empty. It is never a result.
