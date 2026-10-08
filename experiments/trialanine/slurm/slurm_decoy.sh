#!/bin/bash
# Deceptive-target benchmark, step 1: deceptiveness check of the candidates (decoy_explore.py), one task per
# (candidate, weight): task = c + NC * j, weight = WEIGHTS[j].
#   NC=48 WEIGHTS="0.05 0.2" sbatch --array=0-95 experiments/trialanine/slurm/slurm_decoy.sh
#SBATCH --job-name=tri-decoy
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --exclude=salmon-02,salmon-09
#SBATCH --output=logs/trialanine/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
D=${D:-runs/trialanine/decoy}; mkdir -p $D/explore
NC=${NC:-48}; read -ra W <<< "${WEIGHTS:-0.05 0.2}"
t=${SLURM_ARRAY_TASK_ID:-0}; c=$((t % NC)); w=${W[$((t / NC))]}
python -W ignore experiments/trialanine/pipeline/decoy_explore.py --meta $D/candidates.json --kind countP --wP $w \
  --starts ${STARTS:-20} --ids $c --thin ${THIN:-2} --out $D/explore/c$(printf %03d $c)_w$w.json
