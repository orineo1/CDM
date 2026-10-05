#!/bin/bash
# Equilibration check of the protocol for N beads: 24 random eps (+ all-low, all-high, all-0.8), 64 replicas from the
# straight chain, every frame kept; populations early vs late and between replica halves (check_equilibration.py).
#   N=6 sbatch experiments/peptide/slurm/slurm_equil.sh       (STEPS=8000000 BURN=... to test a longer protocol)
#SBATCH --job-name=pep-equil
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/peptide/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh             # Python environment (CDM_ENV)
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
N=${N:-4}; [ "$N" = 4 ] && R0=runs/peptide || R0=runs/peptide_N$N; R=${R:-$R0}; P=experiments/peptide/pipeline; mkdir -p $R
# simulation protocol (training data, targets and evaluation must share it): N = 4 their 4M steps, N >= 5 longer
if [ "$N" = 4 ]; then STEPS=${STEPS:-4000000}; BURN=${BURN:-50000}; STRIDE=${STRIDE:-200}
else STEPS=${STEPS:-16000000}; BURN=${BURN:-1000000}; STRIDE=${STRIDE:-800}; fi
SIM="--n-steps $STEPS --burn $BURN --stride $STRIDE"
K=$([ "$N" = 4 ] && echo 2 || echo $((N - 1)))
python -c "
import numpy as np; r = np.random.default_rng(5); K = $K
e = [np.full(K, 0.25), np.full(K, 1.4), np.full(K, 0.8)] + [0.21 + 1.34 * r.random(K) for _ in range(24)]
np.save('$R/equil_eps.npy', np.array(e, dtype=np.float32))"
python $P/sim_ensembles.py --n-beads $N --eps-file $R/equil_eps.npy --n-rep 64 --no-subsample --seed 11 \
  $SIM --out $R/equil_ens.pt
python $P/check_equilibration.py $R/equil_ens.pt --out experiments/peptide/results/equilibration_N$N${TAG:-}.md
echo "done $(date)"
