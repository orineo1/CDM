#!/usr/bin/env bash
# One grid-search configuration per SLURM array task. Submit with ./submit.sh, which
# sizes the array from the grid; by hand:
#   sbatch --array=0-35 experiments/mnist/grid_search/slurm.sh uniform
#
#SBATCH --job-name=cdm-mnist-grid
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/grid-%x-%A_%a.out
#SBATCH --error=logs/grid-%x-%A_%a.err
set -euo pipefail
export PYTHONUNBUFFERED=1
TARGET="${1:?usage: sbatch --array=0-N slurm.sh <uniform|bimodal> [grid name]}"
GRID="${2:-$TARGET}"            # grids/<GRID>.args; default grids/<target>.args
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
[ -f experiments/mnist/grid_search/grid_search.py ] || { echo "submit from the repo root"; exit 1; }
VENV=${VENV:-/sci/labs/orzuk/ori_m/venvs/cdm}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
CACHE=${CACHE:-/sci/labs/orzuk/ori_m/.cache}
export XDG_CACHE_HOME="$CACHE" HF_HOME="$CACHE/huggingface" MPLCONFIGDIR="$CACHE/matplotlib" TORCH_HOME="$CACHE/torch"

# shellcheck disable=SC2046
python experiments/mnist/grid_search/grid_search.py --target "$TARGET" \
    $(cat "experiments/mnist/grid_search/grids/$GRID.args") \
    --index "$SLURM_ARRAY_TASK_ID" --skip-existing \
    --wandb-mode "${WANDB_MODE:-online}" ${GRID_EXTRA:-}
