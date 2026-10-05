#!/bin/bash
# Step 1: simulate the training ensembles (300 random eps) and the 40 held-out targets, then build
# data.pt / prior_data.pt / targets/*.npy for our pipeline.   sbatch experiments/peptide/slurm/slurm_data.sh
# N beads: N=5 sbatch ... (all N - 1 eps_d free; FREE=1,2 FIXED=3=0.5,4=0.5 for another design); runs/peptide_N<N>.
# N = 4 defaults: 300 eps x 4000 frames; N >= 5: 600 eps x 2000 frames (the same 1.2M rows, denser in the 4-5-D box).
#SBATCH --job-name=pep-data
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/peptide/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv$(grep -q "^13" /etc/debian_version 2>/dev/null && echo _deb13)}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
N=${N:-4}; [ "$N" = 4 ] && R0=runs/peptide || R0=runs/peptide_N$N; R=${R:-$R0}; P=experiments/peptide/pipeline; mkdir -p $R
if [ "$N" = 4 ]; then N_EPS=${N_EPS:-300}; FRAMES=${FRAMES:-4000}; else N_EPS=${N_EPS:-600}; FRAMES=${FRAMES:-2000}; fi
# simulation protocol (training data, targets and evaluation must share it): N = 4 their 4M steps, N >= 5 longer
if [ "$N" = 4 ]; then STEPS=${STEPS:-4000000}; BURN=${BURN:-50000}; STRIDE=${STRIDE:-200}
else STEPS=${STEPS:-16000000}; BURN=${BURN:-1000000}; STRIDE=${STRIDE:-800}; fi
SIM="--n-steps $STEPS --burn $BURN --stride $STRIDE"
D="--n-beads $N ${FREE:+--free-eps $FREE} ${FIXED:+--fixed-eps $FIXED}"
python $P/sim_ensembles.py $D --n-eps $N_EPS --seed 1 --keep $((2 * FRAMES)) $SIM --out $R/train_ens.pt
python $P/sim_ensembles.py $D --n-eps 40 --seed 2 --n-rep 64 --chunk 200 --keep 20000 $SIM --out $R/targets_ens.pt
python $P/make_data.py --train $R/train_ens.pt --targets $R/targets_ens.pt --frames-per-eps $FRAMES --out $R
echo "done $(date)"
