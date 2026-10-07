#!/bin/bash
# Landscape analysis (reweighting of the x0 reference walkers) on one node.
#   TAG=_rand ARGS="--targets 3 --starts 48" sbatch experiments/trialanine/slurm/slurm_landscape.sh
#SBATCH --job-name=tri-land
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --exclude=salmon-02,salmon-09
#SBATCH --output=logs/trialanine/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
[ -f ./cluster_local.sh ] && . ./cluster_local.sh
VENV=${VENV:-$CDM_LAB/venvs/trialanine-venv}; . $VENV/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
R=$PWD/runs/trialanine; cd experiments/trialanine/pipeline
python landscape.py --frames "$R/prod_x0_w*.npz" --out ../results --box ${BOX:-4} --sigma ${SIGMA:-1} --tag ${TAG:-} ${ARGS:-}
