#!/bin/bash
# Step 4: re-simulate every returned eps_hat (plus eps* itself, the noise floor) and score it against the
# target ensemble.   sbatch experiments/peptide/slurm/slurm_eval.sh
#SBATCH --job-name=pep-eval
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/peptide/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=${R:-runs/peptide}; P=experiments/peptide/pipeline; RES=${RES:-$R/results}; EV=${EV:-$R/eval}; mkdir -p $EV
python $P/evaluate.py collect --results "$RES/*.json" --targets-ens $R/targets_ens.pt --out $EV/eval_eps.npy
python $P/sim_ensembles.py --eps-file $EV/eval_eps.npy --n-rep 64 --chunk 200 --keep 20000 --seed 3 --out $EV/eval_ens.pt
python $P/evaluate.py metrics --eval-eps $EV/eval_eps.npy --eval-ens $EV/eval_ens.pt --targets-ens $R/targets_ens.pt \
  --out ${TABLE_DIR:-experiments/peptide/results}
echo "done $(date)"
