#!/bin/bash
# MNIST pilot of tangent zeroth-order guidance on the bimodal target of Table 2 (n_cond 1500, T 125, n_mc 3,
# original step rule, variance 300, no clamp). One array task per (arm, seed), seeds 0-4:
#   tasks 0-4    base           the Table 2 run (backpropagation through the sampler)
#   tasks 5-9    tangent30      --guide-grad zo --zo-dirs-mode tangent --zo-dirs 30, every step
#   tasks 10-14  tangent30_late the same, only at reverse steps t <= 500 (backpropagation before)
#   task 15      diag           diag_tweedie_spectrum.py: how many directions span the Tweedie Jacobian
#   sbatch --array=0-15 experiments/mnist/table_mlgdf/slurm_zo_pilot.sh      (from the repository root)
# Outputs runs/mnist_zo/<arm>/s<seed>/*.pkl (search_mnist.py) and runs/mnist_zo/spectrum.json;
# python experiments/mnist/table_mlgdf/make_table.py runs/mnist_zo/<arm>/s<seed> prints a run's numbers.
#SBATCH --job-name=mnist-zo
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh             # Python environment (CDM_ENV)
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
M=experiments/mnist
OUT=${OUT:-runs/mnist_zo}; mkdir -p "$OUT"
k=${SLURM_ARRAY_TASK_ID:-0}
if [ "$k" = 15 ]; then
  python $M/table_mlgdf/diag_tweedie_spectrum.py --steps 125 --at 0.9 0.6 0.3 0.1 0.03 --m 100 --n-traj 3 \
    --ckpt-dir $M/checkpoints --out "$OUT/spectrum.json"
  exit 0
fi
ARMS=(base tangent30 tangent30_late)
ARM=${ARMS[$((k / 5))]}
S=$((k % 5))
case $ARM in
  base)           ZO="" ;;
  tangent30)      ZO="--guide-grad zo --zo-dirs-mode tangent --zo-dirs 30" ;;
  tangent30_late) ZO="--guide-grad zo --zo-dirs-mode tangent --zo-dirs 30 --zo-from-t 500" ;;
esac
echo "mnist-zo $ARM seed $S: $ZO ($(hostname) $(date))"
python $M/table_mlgdf/search_mnist.py --target bimodal --steps 125 --n-cond 1500 --n-mc 3 \
  --step-size-mode original --bimodal-var 300 --seed-start $S --n-seeds 1 --device cuda --threshold 0.8 --no-clamp \
  --ckpt-dir $M/checkpoints --classifier-path ${CLASSIFIER:-$M/checkpoints/robust_classifier.pth} \
  --mnist-data-root ${MNIST_DATA_ROOT:-$M/train/outputs/data} $ZO --out "$OUT/$ARM/s$S"
echo "done $(date)"
