#!/usr/bin/env bash
# Table 1 on the cluster: one array task per (dim X, dim Y) scenario, then one collect job.
#
#   sbatch --partition=<gpu partition> scripts/slurm_table1.sh          # nine scenarios, then collect
#   SETTINGS="5x5 10x10" sbatch --array=0-1 scripts/slurm_table1.sh     # a subset (array size = number of settings)
#   TABLE_ARGS="--attempts 3 --epochs 5000" sbatch ...                  # extra flags for make_table1.py
#
# Each task generates its scenario's data, trains the three networks, runs check_models.py
# and the four search algorithms (steps whose outputs exist are skipped, so a task can be
# resubmitted), all under runs/table1/<scenario>/. When the array is done the collect job
# writes results/table1.{json,md,tex} and results/table1_appendix.md.
#
#SBATCH --job-name=cdm-table1
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/table1-%A_%a.out
#SBATCH --error=logs/table1-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -f scripts/make_table1.py ] || { echo "run sbatch from the repo root"; exit 1; }
mkdir -p logs
VENV=${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"

SETTINGS=${SETTINGS:-"1x1 1x5 1x10 5x1 5x5 5x10 10x1 10x5 10x10"}
read -r -a ARR <<< "$SETTINGS"
S=${ARR[$SLURM_ARRAY_TASK_ID]}
echo "=== task $SLURM_ARRAY_TASK_ID: scenario $S on $(hostname) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python scripts/make_table1.py --settings "$S" --device cuda --no-collect ${TABLE_ARGS:-}

# the last task to finish collects (cheap; a re-run of --collect is always safe)
if [ "$SLURM_ARRAY_TASK_ID" = "$((${#ARR[@]} - 1))" ]; then
    echo "=== collect ==="
    python scripts/make_table1.py --settings $SETTINGS --collect ${TABLE_ARGS:-} || true
fi
echo "done"
