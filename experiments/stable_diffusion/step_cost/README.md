# step_cost

**Produces:** the cost numbers of the paper's ablation and selection appendix, from one
guidance step measured in isolation:

1. **Memory and run time, Turbo vs Base** (All selection): peak GPU memory decomposed as
   models + fixed part + the n_cond samples, the time of one step, and the full run
   (125 guided steps), measured for the Turbo Sprinter (K = 2) and projected for SDXL Base
   (K* = 30): the two share the architecture, so the per-sample memory and time are
   multiplied by K*/K and the rest is unchanged.
2. **Per-step time, All vs Uniform** (n_sel/n_cond = 0.5), mean ± std over 5 repeats, for
   n_cond = 100 and 120.

| | |
|---|---|
| Entry point | `run.sh` -> `cdm/models/sd/step_cost.py` |
| Input | none: the paper's settings (`../four_targets/config/balanced.env`) |
| GPU | yes; the default (2 x n_cond, 2 rules, 5 repeats) is about an hour. Numbers depend on the GPU: the paper's ablation used an A100 (80 GB) |
| W&B | project `sdxl-step-cost` |

Ported from the paper's `measure_dps_step_memory.ipynb` (memory) and from
`scripts/time_dps_step.py` / `time_dps_step.sh` on branch `test-witness-result` of
`orineo1/conditional-matching-paper` (the per-step timing of the selection table), and
measured the same way: random latents and 250 random target CLIP vectors (the content does
not change the cost), fast kernels, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, one
untimed call of each configuration, then 5 timed calls, mean ± std.

## Run

From the repository root, with the venv active:

```bash
mkdir -p logs
sbatch --partition=<A100 partition> --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 \
       --output=logs/sd-cost-%j.out \
       --wrap "bash experiments/stable_diffusion/step_cost/run.sh"
```

Flags override the defaults; a configuration that runs out of memory is recorded as OOM
and the next one runs (on a 48 GB L40S, n_cond = 120 with All may not fit):

```bash
bash experiments/stable_diffusion/step_cost/run.sh --num_variations 100 --rules all --n_repeats 1
bash experiments/stable_diffusion/step_cost/run.sh --deterministic         # the runs' kernels: slower
```

Every flag: `bash experiments/stable_diffusion/step_cost/run.sh help`. `DRY_RUN=1` prints
the command.

## What happens

One guidance step as `run_mlgd_f.py` runs it (mid-schedule timestep, step 125 of 250), on
random latents and 250 random target CLIP vectors; for every n_cond and rule, one untimed
warm-up call, then 5 timed calls:

| part | what | measured |
|---|---|---|
| models | every model loaded | memory |
| fixed | Architect UNet forward, pred_x0, VAE decode | peak memory above models, time |
| samples | the n_cond Sprinter samples (K = 2), the MMD, the backward through the n_sel selected ones | peak memory above the fixed part, time |

Fast GPU kernels by default: a cost measurement needs no bit-identical results, and the
deterministic kernels of the optimisation runs (`--deterministic`) are slower and use more
memory. `step_cost.md` states which were used.

## Outputs

`outputs/<GPU>_<SLURM job id>/` (gitignored): `step_cost.md` (the two tables),
`step_cost.json` (every repeat, the settings, the GPU, model revisions, versions),
`run_info.txt`. `python experiments/stable_diffusion/export_results.py` copies the three
into `results/<GPU>_<id>/`.

## Differences from the notebook

- The step is the pipeline's (SDXL Base Architect, ControlNet scale 0.5; the notebook used
  Turbo as Architect and scale 0.8); the notebook unpacked six return values of
  `run_dps_step_clip`, which returns five.
- As in `time_dps_step.py`: random inputs instead of generated targets and a scribble, a
  warm-up call per configuration, the Uniform gradient not rescaled (cost-neutral).
- Base is projected from the decomposition (fixed part unchanged, per-sample part x K*/K)
  for memory and time alike; the notebook's separate Turbo-vs-Base sampling timing and
  sample images are left out.
- Repeats with mean ± std, a warm-up step, both selection rules, OOM recorded.
