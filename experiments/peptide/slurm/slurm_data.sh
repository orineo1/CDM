#!/bin/bash
# Step 1: simulate the training ensembles (300 random eps) and the 40 held-out targets, then build
# data.pt / prior_data.pt / targets/*.npy for our pipeline.   sbatch experiments/peptide/slurm/slurm_data.sh
#SBATCH --job-name=pep-data
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/peptide/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=${R:-runs/peptide}; P=experiments/peptide/pipeline; mkdir -p $R
python $P/sim_ensembles.py --n-eps 300 --seed 1 --keep 8000 --out $R/train_ens.pt
python $P/sim_ensembles.py --n-eps 40 --seed 2 --n-rep 64 --chunk 200 --keep 20000 --out $R/targets_ens.pt
python $P/make_data.py --train $R/train_ens.pt --targets $R/targets_ens.pt --frames-per-eps 4000 --out $R
echo "done $(date)"
