#!/usr/bin/env bash
# Stable Diffusion four-targets experiment (Table 3 / Table 15) on the cluster.
# A thin SLURM wrapper around run.sh; logs every run to Weights & Biases.
#
# From the repo root (logs/ must exist before sbatch, SLURM opens the log file first):
#   mkdir -p logs
#   sbatch experiments/stable_diffusion/four_targets/slurm_sd.sh balanced            # full run, one target
#   TIER=smoke sbatch --time=01:00:00 experiments/stable_diffusion/four_targets/slurm_sd.sh balanced
#   sbatch --array=0-3 experiments/stable_diffusion/four_targets/slurm_sd.sh         # all four targets
#   sbatch --partition=<gpu partition> --gres=gpu:a100:1 experiments/stable_diffusion/four_targets/slurm_sd.sh skewed
#
# Scenarios: balanced | skewed | gender_interpolation | age_interpolation
# (array index 0..3 in that order). Outputs: experiments/stable_diffusion/four_targets/outputs/.
#
# Environment variables (all optional):
#   TIER=full|smoke          which run.sh configuration (default full)
#   VENV=/path/to/venv       venv to activate (default .venv in the repo)
#   WANDB_API_KEY=...        if you have not run `wandb login` on this cluster
#   WANDB_MODE=offline       compute nodes without internet: sync later with
#                            `wandb sync wandb/offline-run-*` from a login node
#   HF_HOME=/scratch/...     where the ~tens of GB of SDXL/ControlNet/CLIP weights are cached
#
#SBATCH --job-name=cdm-sd
#SBATCH --partition=salmon         # override: sbatch --partition=<name> ...
#SBATCH --gres=gpu:1               # needs a large GPU: the paper used an L40S (48 GB)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sd-%j.out
#SBATCH --error=logs/sd-%j.err
set -euo pipefail

# sbatch runs a copy of this script from the spool dir, so locate the repo via the submit dir
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
[ -f experiments/stable_diffusion/four_targets/run.sh ] || { echo "run sbatch from the repo root" >&2; exit 1; }
mkdir -p logs

VENV=${VENV:-.venv}
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi

SCENARIOS=(balanced skewed gender_interpolation age_interpolation)
SCENARIO=${1:-${SCENARIOS[${SLURM_ARRAY_TASK_ID:-0}]}}
TIER=${TIER:-full}

export PYTHONUNBUFFERED=1
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA GPU visible: check --gres and the torch build')"
if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != offline ] && ! grep -q api.wandb.ai ~/.netrc 2>/dev/null; then
    echo "warning: no wandb login found; run 'wandb login' once, or export WANDB_API_KEY / WANDB_MODE=offline" >&2
fi

echo "=== $TIER | $SCENARIO | job ${SLURM_JOB_ID:-local} on $(hostname) ==="
PYTHON=python bash experiments/stable_diffusion/four_targets/run.sh "$TIER" "$SCENARIO"
echo "done: $(ls -d experiments/stable_diffusion/four_targets/outputs/${TIER}_*)"
