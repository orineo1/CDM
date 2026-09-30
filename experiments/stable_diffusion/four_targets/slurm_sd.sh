#!/usr/bin/env bash
# SLURM wrapper around run.sh: same arguments, same config defaults, same CLI overrides.
#
# From the repo root (logs/ must exist before sbatch -- SLURM opens the log file first):
#   mkdir -p logs
#   sbatch --partition=<p> experiments/stable_diffusion/four_targets/slurm_sd.sh smoke balanced
#   sbatch --partition=<p> experiments/stable_diffusion/four_targets/slurm_sd.sh full  skewed --base_zeta 5
#   sbatch --partition=<p> --array=0-3 experiments/stable_diffusion/four_targets/slurm_sd.sh full all
#
# With --array and scenario `all`, task i runs scenario i of:
#   0 balanced  1 skewed  2 gender_interpolation  3 age_interpolation
# Without --array, `all` runs the four one after another in one job.
# Partition, GPU type, time etc. go on the sbatch command line (--partition, --gres, --time).
#
# Environment (optional; exported variables reach the job):
#   VENV=/path/to/venv       venv to activate (default: .venv in the repo, if present)
#   HF_HOME=/path            model cache (~tens of GB of SDXL / ControlNet / CLIP weights)
#   WANDB_API_KEY=...        if `wandb login` has not been run on this cluster
#   WANDB_MODE=offline       compute nodes without internet; then from a login node:
#                            wandb sync wandb/offline-run-*
#   WANDB_PROJECT, WANDB_ENTITY, WANDB_NAME   see run.sh
#
#SBATCH --job-name=cdm-sd
#SBATCH --gres=gpu:1               # full runs need a large GPU: the paper used an L40S (48 GB)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sd-%j.out
#SBATCH --error=logs/sd-%j.err
set -euo pipefail

# sbatch runs a copy of this script from the spool dir, so locate the repo via the submit dir
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
[ -f experiments/stable_diffusion/four_targets/run.sh ] || { echo "run sbatch from the repo root" >&2; exit 1; }

VENV=${VENV:-.venv}
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi

TIER="${1:-full}"; shift || true
SCEN="${1:-all}";  shift || true
SCENARIOS=(balanced skewed gender_interpolation age_interpolation)
if [ "$SCEN" = all ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  SCEN="${SCENARIOS[$SLURM_ARRAY_TASK_ID]:?array index must be 0-3}"
fi

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
if [ "$TIER" != check ]; then
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA GPU visible: check --gres and the torch build')"
  if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != offline ] && ! grep -q api.wandb.ai ~/.netrc 2>/dev/null; then
    echo "warning: no wandb login found; run 'wandb login' once, or export WANDB_API_KEY / WANDB_MODE=offline" >&2
  fi
fi

echo "=== job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} on $(hostname): $TIER $SCEN $* ==="
PYTHON=python bash experiments/stable_diffusion/four_targets/run.sh "$TIER" "$SCEN" "$@"
