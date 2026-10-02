#!/usr/bin/env bash
# GPU memory and time of one MLGD-F guidance step (paper: ablation, Turbo vs Base; selection
# rules, per-step time). Settings of the paper's runs (../four_targets/config/balanced.env).
#
#   bash run.sh [flags...]     GPU. Default: n_cond 100 and 120, All and Uniform (n_sel/n_cond
#                              0.5), 5 repeats each = 20 guidance steps; ~1 h on an A100.
#   bash run.sh help           every flag cdm.models.sd.step_cost accepts.
#
# Any flag overrides the defaults; an out-of-memory configuration is recorded, not fatal:
#   bash run.sh --num_variations 100 --rules all --n_repeats 1     # just the memory row
#
# Output: outputs/<GPU>_<SLURM job id or time>/ (--output_dir overrides): step_cost.md (the two
# tables), step_cost.json (every measurement), run_info.txt. W&B: project sdxl-step-cost.
# Numbers depend on the GPU: the paper's ablation used an A100 (80 GB).
#
# On SLURM, from the repo root:
#   sbatch --partition=<A100 partition> --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 \
#          --output=logs/sd-cost-%j.out --wrap "bash experiments/stable_diffusion/step_cost/run.sh"
#
# Environment: as four_targets/run.sh (PYTHON, VENV, WANDB_PROJECT, WANDB_ENTITY, WANDB_NAME,
# RUN_TAG, CACHE_DIR, DRY_RUN=1 prints the command instead of running it).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/outputs"
# shellcheck source=../env.sh
source "$HERE/../env.sh"
CFG="$SD_DIR/four_targets/config"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"   # as time_dps_step.sh
# env.sh sets CUBLAS_WORKSPACE_CONFIG for deterministic runs; it limits cuBLAS's workspace and
# slows large matmuls, so drop it unless --deterministic is asked for (time_dps_step.sh has none)
case " $* " in *" --deterministic "*) ;; *) unset CUBLAS_WORKSPACE_CONFIG ;; esac
if [ "${1:-}" = help ]; then "$PY" -m cdm.models.sd.step_cost --help; exit 0; fi
require_gpu
set -a; source "$CFG/models.env"; source "$CFG/balanced.env"; set +a
gpu="$( [ -n "${DRY_RUN:-}" ] && echo GPU || "$PY" -c "import torch; print(torch.cuda.get_device_name(0).split()[-1])")"
tag="${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
outdir="$(flag_value --output_dir "$OUT/${gpu}_${tag}" "$@")"
args=( --output_dir "$outdir" --n_steps "$N_STEPS" --start_step "$START_STEP"
       --guidance_scale "$GUIDANCE_SCALE" --controlnet_scale "$CONTROLNET_SCALE" --seed "$SEED"
       $(revision_args) --wandb_project "${WANDB_PROJECT:-sdxl-step-cost}" )
[ -n "${WANDB_ENTITY:-}" ] && args+=( --wandb_entity "$WANDB_ENTITY" )
args+=( "$@" )
export WANDB_NAME="${WANDB_NAME:-step_cost_$(basename "$outdir")}"
export WANDB_TAGS="${WANDB_TAGS:-step_cost,$gpu}"
cmd=( "$PY" -m cdm.models.sd.step_cost "${args[@]}" )
echo "=== step_cost | $outdir | wandb ${WANDB_PROJECT:-sdxl-step-cost}/$WANDB_NAME ==="
printf '%q ' "${cmd[@]}"; echo
[ -n "${DRY_RUN:-}" ] && exit 0
write_run_info "$outdir" "config/balanced.env  overrides: ${*:-none}" "${cmd[@]}"
"${cmd[@]}"
echo "save into results/: python experiments/stable_diffusion/export_results.py"
