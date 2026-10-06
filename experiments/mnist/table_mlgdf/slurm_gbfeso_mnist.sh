#!/bin/bash
# GB-FESO (Adam on the design through the frozen conditional sampler, no prior) on the three MNIST angle
# targets of Table 2. One array task per target, 15 seeds (0-14) inside, n_cond and target variance from
# config/targets.tsv (the MLGD-F settings), GB-FESO's optimizer defaults (search_gbfeso_mnist.py):
#   task 0 unimodal, task 1 bimodal, task 2 uniform
#   sbatch --array=0-2 experiments/mnist/table_mlgdf/slurm_gbfeso_mnist.sh      (from the repository root)
# Extra flags for search_gbfeso_mnist.py go in GBFESO_ARGS (e.g. GBFESO_ARGS="--n-restarts 5" or "--init randn");
# OUT (default runs/mnist_gbfeso) should then differ per setting. Outputs one .pkl per target in $OUT;
#   python experiments/mnist/table_mlgdf/make_table.py runs/mnist_gbfeso      prints the Table 2 rows.
#SBATCH --job-name=mnist-gbfeso
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh             # Python environment (CDM_ENV)
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
M=experiments/mnist
OUT=${OUT:-runs/mnist_gbfeso}; mkdir -p "$OUT" logs
k=${SLURM_ARRAY_TASK_ID:-0}
TARGETS=(unimodal bimodal uniform)
T=${TARGETS[$k]}
read -r _ N_COND _ _ _ VAR < <(awk -v t="$T" '$1 == t' $M/table_mlgdf/config/targets.tsv)
EXTRA=()
[ "$T" = unimodal ] && EXTRA=(--unimodal-var "$VAR")
[ "$T" = bimodal ] && EXTRA=(--bimodal-var "$VAR")
echo "mnist-gbfeso $T: n_cond $N_COND ${EXTRA[*]} ${GBFESO_ARGS:-} ($(hostname) $(date), commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown))"
python $M/table_mlgdf/search_gbfeso_mnist.py --target "$T" --n-cond "$N_COND" "${EXTRA[@]}" \
  --n-seeds 15 --seed-start 0 --device cuda --threshold 0.8 \
  --ckpt-dir $M/checkpoints --classifier-path ${CLASSIFIER:-$M/checkpoints/robust_classifier.pth} \
  --mnist-data-root ${MNIST_DATA_ROOT:-$M/train/outputs/data} ${GBFESO_ARGS:-} --out "$OUT"
echo "done $(date)"
