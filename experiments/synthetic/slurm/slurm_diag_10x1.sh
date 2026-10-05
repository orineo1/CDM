#!/bin/bash
# Diagnostics for the (dim X, dim Y) = (10, 1) cell, Table 1 flags throughout.
#   task 0: oracle sampler (exact conditional) with refinement, 10 trajectories x 5 attempts
#   task 1: exact-loss oracle (closed-form MMD^2, exact gradient incl. mixture weights), 10 x 5
#   task 2: exact-loss oracle, 100 x 3
#   sbatch --array=0-2 experiments/synthetic/slurm/slurm_diag_10x1.sh
#   then: sbatch --dependency=afterok:<id> --wrap "... diag_trajectories.py ..." (see README of the run)
#SBATCH --job-name=cdm-diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/diag_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
VENV=${VENV:-.venv}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
D=${D:-runs/table1v2/x10_y1}
COMMON="--prior $D/models/uncond/model_unconditional.pt --target $D/data.pt --algo lgd --n-cond 250
        --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 --guidance-optim adam --step-size 0.05 --seed 42 --device cuda"
case "$SLURM_ARRAY_TASK_ID" in
  0) python experiments/synthetic/pipeline/search_mog.py $COMMON --sampler exact --n-restarts 10 --n-attempts 5 --out $D/MLGD-X_diag10.json ;;
  1) python experiments/synthetic/pipeline/search_mog.py $COMMON --sampler exactloss --n-restarts 10 --n-attempts 5 --out $D/MLGD-E_diag10.json ;;
  2) python experiments/synthetic/pipeline/search_mog.py $COMMON --sampler exactloss --n-restarts 100 --n-attempts 3 --out $D/MLGD-E_diag100.json ;;
esac
echo "done $(date)"
