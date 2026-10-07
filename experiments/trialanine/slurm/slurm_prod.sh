#!/bin/bash
# Frozen-bias production walkers for trialanine (one CPU per array task; OpenMM Reference platform).
#   TAG=x0 NS=50 BIAS=runs/trialanine/metad_bias.npz sbatch --array=0-15 experiments/trialanine/slurm/slurm_prod.sh
#   TAG=val1 X=runs/trialanine/x_val1.npy ... (design as .npy or comma list; default x0 = 0)
# Output: runs/trialanine/prod_${TAG}_w${TASK}.npz/.json (phi, psi, frozen-bias energy V; diagnostics).
#SBATCH --job-name=tri-prod
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --exclude=salmon-02,salmon-09
#SBATCH --output=logs/trialanine/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
[ -f ./cluster_local.sh ] && . ./cluster_local.sh
VENV=${VENV:-$CDM_LAB/venvs/trialanine-venv}; . $VENV/bin/activate
export PYTHONUNBUFFERED=1 OPENMM_CPU_THREADS=1
W=${SLURM_ARRAY_TASK_ID:-0}; TAG=${TAG:-x0}; NS=${NS:-50}; R=runs/trialanine; mkdir -p $R
python experiments/trialanine/pipeline/sample.py prod ${X:+--x $X} ${BIAS:+--bias $BIAS} --ns $NS --seed $((1000 + W)) \
  --out $R/prod_${TAG}_w${W}.npz
