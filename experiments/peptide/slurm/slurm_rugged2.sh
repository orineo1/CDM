#!/bin/bash
# Rugged peptide model with charges (frustrated electrostatics), ruggedness test before any method comparison.
# Design: N beads, LJ chain strengths fixed (eps_d = EPS), free charges q_1..q_N in [-1, 1], screened Coulomb
# U_ij = A q_i q_j exp(-r / debye) / r (simulate.Design, --free charge --elec A=..,debye=..).
#   STAGE=pilot sbatch --array=0-3 slurm_rugged2.sh    one configuration per task (CFGS below): pool of 256 prior
#        designs (8 replicas), 8 targets (64 replicas), equilibration check (8 designs incl. charge corners,
#        16 replicas), then reweight.py: MBAR landscape, multi-start descent on the true loss, direct-simulation
#        check of the barriers (--confirm). Outputs runs/rugged2/c<k>/.
# Simulation protocol per N as in the benchmark: N = 4 their 4M / 50k / 200; N >= 5 16M / 1M / 800.
#SBATCH --job-name=rugged2
#SBATCH --partition=salmon,catfish
#SBATCH --exclude=salmon-02,salmon-09
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/peptide/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
P=experiments/peptide/pipeline; k=${SLURM_ARRAY_TASK_ID:-0}
#      N  EPS  A   debye
CFGS=("4 0.5 1 2" "4 0.5 3 2" "4 0.5 3 1" "5 0.5 3 2")
read -r N EPS A DEB <<< "${CFG:-${CFGS[$k]}}"
R=${R:-runs/rugged2/c$k}; mkdir -p $R
D="--n-beads $N --free charge --fixed eps=$EPS --elec A=$A,debye=$DEB,rcut=6"
if [ "$N" = 4 ]; then STEPS=${STEPS:-4000000}; BURN=${BURN:-50000}; STRIDE=${STRIDE:-200}
else STEPS=${STEPS:-16000000}; BURN=${BURN:-1000000}; STRIDE=${STRIDE:-800}; fi
SIM="--n-steps $STEPS --burn $BURN --stride $STRIDE"
echo "config $k: N=$N eps=$EPS A=$A debye=$DEB, $SIM"; python -c "import torch; print(torch.cuda.get_device_name())"
case ${STAGE:?set STAGE=pilot} in
pilot)
  [ -f $R/pool.pt ] || python $P/sim_ensembles.py $D --n-eps ${POOL:-256} --seed 1 --n-rep 8 --keep 2000 $SIM --out $R/pool.pt
  [ -f $R/targets.pt ] || python $P/sim_ensembles.py $D --n-eps 8 --seed 2 --n-rep 64 --chunk 200 --keep 20000 $SIM --out $R/targets.pt
  if [ ! -f $R/equil.md ]; then
    python - "$N" "$R/equil_x.npy" <<'EOF'
import sys, numpy as np
n = int(sys.argv[1]); rng = np.random.default_rng(5)
alt = [(-1) ** i for i in range(n)]; blk = [1 if i < n // 2 else -1 for i in range(n)]
ends = [1] + [-1] * (n - 2) + [1]
X = np.array([alt, blk, ends, [1] * n] + list(rng.uniform(-1, 1, (4, n))), dtype=np.float32)
np.save(sys.argv[2], X)
EOF
    python $P/sim_ensembles.py $D --eps-file $R/equil_x.npy --seed 11 --n-rep 16 --no-subsample $SIM --out $R/equil_ens.pt
    python $P/check_equilibration.py $R/equil_ens.pt --out $R/equil.md
  fi
  python $P/reweight.py --pool $R/pool.pt --targets $R/targets.pt --out $R/landscape.json --confirm $SIM
  python $P/reweight.py --analyze $R/landscape.json
  ;;
esac
echo "done $(date)"
