#!/bin/bash
# Re-simulate only the 40 targets (64 replicas) and rebuild data.pt / targets/*.npy from the existing
# train_ens.pt (used once: the first targets had 8 replicas, too noisy for metastable eps).
#SBATCH --job-name=pep-targets
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/peptide/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv$(grep -q "^13" /etc/debian_version 2>/dev/null && echo _deb13)}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
N=${N:-4}; [ "$N" = 4 ] && R0=runs/peptide || R0=runs/peptide_N$N; R=${R:-$R0}; P=experiments/peptide/pipeline
# simulation protocol (training data, targets and evaluation must share it): N = 4 their 4M steps, N >= 5 longer
if [ "$N" = 4 ]; then STEPS=${STEPS:-4000000}; BURN=${BURN:-50000}; STRIDE=${STRIDE:-200}
else STEPS=${STEPS:-16000000}; BURN=${BURN:-1000000}; STRIDE=${STRIDE:-800}; fi
SIM="--n-steps $STEPS --burn $BURN --stride $STRIDE"
[ "$N" = 4 ] && FRAMES=${FRAMES:-4000} || FRAMES=${FRAMES:-2000}
python $P/sim_ensembles.py --design-from $R/train_ens.pt --n-eps 40 --seed 2 --n-rep 64 --chunk 200 --keep 20000 $SIM --out $R/targets_ens.pt
python $P/make_data.py --train $R/train_ens.pt --targets $R/targets_ens.pt --frames-per-eps $FRAMES --out $R
echo "done $(date)"
