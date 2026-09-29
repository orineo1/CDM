#!/usr/bin/env bash
# Table 1 on the cluster: one array task per (dim X, dim Y) scenario, then one collect job.
# Runs experiments/synthetic/main_grid/run.sh full, so the flags live in ONE place.
#
#   sbatch experiments/synthetic/slurm/slurm_table1.sh                                   # nine scenarios, then collect
#   SETTINGS="5x5 10x10" sbatch --array=0-1 experiments/synthetic/slurm/slurm_table1.sh  # a subset (array size = number of settings)
#
# Each task generates its scenario's data, trains the three networks, runs check_models.py
# and the four search algorithms (steps whose outputs exist are skipped, so a task can be
# resubmitted), all under experiments/synthetic/main_grid/outputs/runs/<scenario>/. The last
# task to finish collects the table into experiments/synthetic/main_grid/outputs/table1.*;
# if it finishes before another task, collect again:
#   SETTINGS="1x1 1x5 1x10 5x1 5x5 5x10 10x1 10x5 10x10" bash experiments/synthetic/main_grid/run.sh full --collect
# Logs: cdm-table1-<jobid>_<task>.out in the submit directory.
#
#SBATCH --job-name=cdm-table1
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
[ -f experiments/synthetic/main_grid/run.sh ] || { echo "run sbatch from the repo root"; exit 1; }
VENV=${VENV:-/sci/labs/orzuk/ori_m/venvs/cdm}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"

ALL_SETTINGS=${SETTINGS:-"1x1 1x5 1x10 5x1 5x5 5x10 10x1 10x5 10x10"}
read -r -a ARR <<< "$ALL_SETTINGS"
S=${ARR[$SLURM_ARRAY_TASK_ID]}
echo "=== task $SLURM_ARRAY_TASK_ID: scenario $S on $(hostname) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
SETTINGS="$S" bash experiments/synthetic/main_grid/run.sh full --no-collect

# the last task to finish collects (cheap; a re-run of --collect is always safe)
if [ "$SLURM_ARRAY_TASK_ID" = "$((${#ARR[@]} - 1))" ]; then
    echo "=== collect ==="
    SETTINGS="$ALL_SETTINGS" bash experiments/synthetic/main_grid/run.sh full --collect || true
fi
echo "done"
