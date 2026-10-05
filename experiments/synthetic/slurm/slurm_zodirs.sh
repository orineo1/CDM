#!/bin/bash
# Directions of the zeroth-order guidance at (10, 1), MLGD* recipe (d256 sampler, --guide-grad zo, 30 restarts,
# 100 Adam refinement steps with 3,000 samples keeping the best iterate, reselection of the 5 lowest-loss runs
# with 5,000 samples x 5). Five arms x five seeds, one array task each (task k: arm k / 5, seed 42 + k % 5):
#   coord          coordinate directions (the reference: TV 0.07 in Table 1)
#   orth10         10 random orthonormal directions (= a rotated coordinate basis)
#   tangent10/5/3  10, 5, 3 orthonormal directions in the range of the Tweedie Jacobian
#   sbatch --array=0-24 experiments/synthetic/slurm/slurm_zodirs.sh
# Outputs <OUTDIR>/zodirs_<arm>_s<seed>.json; summary:
#   python experiments/synthetic/pipeline/summarize_runs.py --runs-dir runs/zodirs/x10_y1 --data runs/table1v2/x10_y1/data.pt
#SBATCH --job-name=zodirs
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
D=${D:-runs/table1v2/x10_y1}
SAMPLER=${SAMPLER:-runs/night/x10_y1/models/d256/model_conditional.pt}
OUTDIR=${OUTDIR:-runs/zodirs/x10_y1}; mkdir -p "$OUTDIR"
ARMS=(coord orth10 tangent10 tangent5 tangent3)
k=${SLURM_ARRAY_TASK_ID:-0}
ARM=${ARMS[$((k / 5))]}
S=$((42 + k % 5))
case $ARM in
  coord)     DIRS="--zo-dirs-mode coord" ;;
  orth10)    DIRS="--zo-dirs-mode orth --zo-dirs 10" ;;
  tangent*)  DIRS="--zo-dirs-mode tangent --zo-dirs ${ARM#tangent}" ;;
esac
echo "zodirs $ARM seed $S: $DIRS ($(hostname) $(date))"
python experiments/synthetic/pipeline/search_mog.py --prior ${PRIOR:-$D/models/uncond/model_unconditional.pt} \
  --sampler "$SAMPLER" --target $D/data.pt --algo lgd --n-attempts 1 --n-cond 250 \
  --refine-lr 0.02 --guidance-optim adam --step-size 0.05 \
  --n-restarts 30 --guide-grad zo --refine-keep-best --refine-steps 100 --refine-n-cond 3000 \
  --select-top 5 --select-n 5000 --select-repeats 5 $DIRS \
  --seed $S --device cuda --out "$OUTDIR/zodirs_${ARM}_s$S.json" ${EXTRA:-}
echo "done $(date)"
