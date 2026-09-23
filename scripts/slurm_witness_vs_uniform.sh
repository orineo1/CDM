#!/usr/bin/env bash
# Witness vs. uniform gradient subsampling in the Table 1 format, on the cluster: one array task
# per (dim X, dim Y) scenario, then one collect job.
#
#   sbatch --partition=<gpu partition> scripts/slurm_witness_vs_uniform.sh      # nine scenarios, then collect
#   SETTINGS="5x5 10x10" sbatch --array=0-1 scripts/slurm_witness_vs_uniform.sh # a subset (array size = number of settings)
#   TABLE_ARGS="--k-fracs 0.1 0.5 --n-cond-list 100 250" sbatch ...            # extra flags for witness_vs_uniform_table.py
#
# Each task reuses its scenario's data and three networks from Table 1 (or, if missing, trains
# them as Table 1 does), then runs the search algorithms at every (n_cond, k/n, rule) point
# (outputs that exist are skipped, so a task can be resubmitted), all under
# runs/table1v2/<scenario>/. The last task writes results/witness_vs_uniform.{json,md,tex}.
#
#SBATCH --job-name=cdm-witness
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/witness-%A_%a.out
#SBATCH --error=logs/witness-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -f scripts/witness_vs_uniform_table.py ] || { echo "run sbatch from the repo root"; exit 1; }
mkdir -p logs
VENV=${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"

SETTINGS=${SETTINGS:-"1x1 1x5 1x10 5x1 5x5 5x10 10x1 10x5 10x10"}
RUNS_DIR=${RUNS_DIR:-runs/table1v2}
read -r -a ARR <<< "$SETTINGS"
S=${ARR[$SLURM_ARRAY_TASK_ID]}
echo "=== task $SLURM_ARRAY_TASK_ID: scenario $S on $(hostname) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python scripts/witness_vs_uniform_table.py --settings "$S" --runs-dir "$RUNS_DIR" --device cuda --no-collect ${TABLE_ARGS:-}

# the last task to finish collects (cheap; a re-run of --collect is always safe)
if [ "$SLURM_ARRAY_TASK_ID" = "$((${#ARR[@]} - 1))" ]; then
    echo "=== collect ==="
    python scripts/witness_vs_uniform_table.py --settings $SETTINGS --runs-dir "$RUNS_DIR" --collect ${TABLE_ARGS:-} || true
fi
echo "done"
