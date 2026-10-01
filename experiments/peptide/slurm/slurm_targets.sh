#!/bin/bash
# Re-simulate only the 40 targets (64 replicas) and rebuild data.pt / targets/*.npy from the existing
# train_ens.pt (used once: the first targets had 8 replicas, too noisy for metastable eps).
#SBATCH --job-name=pep-targets
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/peptide/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=${R:-runs/peptide}; P=experiments/peptide/pipeline
python $P/sim_ensembles.py --n-eps 40 --seed 2 --n-rep 64 --chunk 200 --keep 20000 --out $R/targets_ens.pt
python $P/make_data.py --train $R/train_ens.pt --targets $R/targets_ens.pt --frames-per-eps 4000 --out $R
echo "done $(date)"
