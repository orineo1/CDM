#!/bin/bash
# Step 4: re-simulate every returned eps_hat (plus eps* itself, the noise floor) and score it against the
# target ensemble.   [N=5] sbatch experiments/peptide/slurm/slurm_eval.sh   (N > 4: results_table_N<N>.md)
#SBATCH --job-name=pep-eval
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/peptide/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
N=${N:-4}; [ "$N" = 4 ] && R0=runs/peptide || R0=runs/peptide_N$N; R=${R:-$R0}; P=experiments/peptide/pipeline; RES=${RES:-$R/results}; EV=${EV:-$R/eval}; mkdir -p $EV
# simulation protocol (training data, targets and evaluation must share it): N = 4 their 4M steps, N >= 5 longer
if [ "$N" = 4 ]; then STEPS=${STEPS:-4000000}; BURN=${BURN:-50000}; STRIDE=${STRIDE:-200}
else STEPS=${STEPS:-16000000}; BURN=${BURN:-1000000}; STRIDE=${STRIDE:-800}; fi
SIM="--n-steps $STEPS --burn $BURN --stride $STRIDE"
python $P/evaluate.py collect --results "$RES/*.json" --targets-ens $R/targets_ens.pt ${NO_ORACLE:+--no-oracle} --out $EV/eval_eps.npy
python $P/sim_ensembles.py --eps-file $EV/eval_eps.npy --design-from $R/targets_ens.pt $SIM --n-rep 64 --chunk $([ "$N" = 4 ] && echo 200 || echo 100) --keep 20000 --seed 3 --out $EV/eval_ens.pt
python $P/evaluate.py metrics --eval-eps $EV/eval_eps.npy --eval-ens $EV/eval_ens.pt --targets-ens $R/targets_ens.pt \
  ${BASINS:+--basins $BASINS} --suffix="${SUFFIX:-$([ "$N" = 4 ] || echo _N$N)}" --out ${TABLE_DIR:-experiments/peptide/results}
echo "done $(date)"
