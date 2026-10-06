#!/bin/bash
# Rugged (residue-design) peptide model, stages 0-2 (experiments/peptide/results/rugged_stage0-2.md).
# Design: N beads, X = (sigma_1..sigma_N, eta_1..eta_N), chain strengths fixed (eps_d = EPS), see simulate.Design.
#   STAGE=0 sbatch --array=0-3 slurm_rugged.sh   equilibration: 5 random sequences, 8 replicas x 16M steps, per config
#                                                  (task 0: T 0.09; 1: T 0.12; 2: T 0.09, eta in 0.5:1.5; 3: T 0.15)
#   STAGE=1 sbatch --array=0-3 slurm_rugged.sh   single-parameter sweeps (sigma4, eta4, sigma1, eta1; 3 base sequences)
#   STAGE=2 sbatch --array=0-9 slurm_rugged.sh   landscape: tasks 0-4 the residue model (targets 0-4), 5-9 the N = 4
#                                                  benchmark (eps1, eps2) as calibration
# TEMP, BOX, STEPS override the protocol chosen in stage 0. Outputs under runs/rugged/.
#SBATCH --job-name=rugged
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/peptide/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
P=experiments/peptide/pipeline; R=runs/rugged; mkdir -p $R
N=${N:-8}; EPS=${EPS:-0.5}; k=${SLURM_ARRAY_TASK_ID:-0}
D="--n-beads $N --free sigma,eta --fixed eps=$EPS"
STEPS=${STEPS:-16000000}; BURN=${BURN:-1000000}; STRIDE=${STRIDE:-800}
case ${STAGE:?set STAGE=0,1,2} in
0)
  TEMPS=(0.09 0.12 0.09 0.15); BOXES=("" "" "eta=0.5:1.5" "")
  T=${TEMPS[$k]}; B=${BOXES[$k]}
  OUT=$R/equil/cfg$k.pt; mkdir -p $R/equil
  python $P/sim_ensembles.py $D ${B:+--box $B} --temp $T --n-eps 5 --seed 3 --n-rep 8 --n-steps $STEPS --burn $BURN \
    --stride $STRIDE --no-subsample --out $OUT
  python $P/check_equilibration.py $OUT --out $R/equil/cfg${k}_T${T}${B:+_$B}.md
  ;;
1)
  PARAMS=(sigma4 eta4 sigma1 eta1); PR=${PARAMS[$k]}
  case $PR in sigma*) V=${VALUES_SIGMA:-0.7:1.4:15} ;; eta*) V=${VALUES_ETA:-0.3:2.0:15} ;; esac
  mkdir -p $R/probe
  python $P/probe_residue.py $D ${BOX:+--box $BOX} ${TEMP:+--temp $TEMP} --param $PR --values $V --n-base 3 --n-rep 64 \
    --n-steps ${PROBE_STEPS:-$STEPS} --burn $BURN --stride $STRIDE --out $R/probe/$PR.json
  ;;
2)
  if [ "$k" -lt 5 ]; then
    python $P/landscape.py $D ${BOX:+--box $BOX} ${TEMP:+--temp $TEMP} --target-id $k --n-steps ${LAND_STEPS:-$STEPS} \
      --burn $BURN --stride $STRIDE --out $R/landscape/t$(printf %03d $k).json
  else
    python $P/landscape.py --n-beads 4 --free 1,2 --fixed 3=0.5 --target-id $((k - 5)) --n-steps 4000000 --burn 50000 \
      --stride 200 --out $R/landscape_N4/t$(printf %03d $((k - 5))).json
  fi
  ;;
esac
echo "done $(date)"
