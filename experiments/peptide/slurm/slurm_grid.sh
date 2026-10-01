#!/bin/bash
# Step 3c: grid-search baseline (30 x 30 grid on our diffusion surrogate, MMD, top-5 reselection).
#   sbatch --array=0-3 experiments/peptide/slurm/slurm_grid.sh       # task k: targets k*PER .. k*PER+PER-1
#SBATCH --job-name=pep-grid
#SBATCH --partition=catfish        # all methods on the same GPU type (L4) so that wall times compare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/peptide/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=${R:-runs/peptide}; PER=${PER:-10}; OUT=${OUT:-$R/results}; mkdir -p $OUT
k=${SLURM_ARRAY_TASK_ID:-0}
python experiments/peptide/pipeline/grid_search.py --prior $R/models/prior/model_unconditional.pt \
  --sampler $R/models/${SAMPLER:-diff}/model_conditional.pt --targets-dir $R/targets \
  --target-ids $((k * PER))-$((k * PER + PER - 1)) --grid 30 --out $OUT/grid${TAG:-}_$k.json
echo "done $(date)"
