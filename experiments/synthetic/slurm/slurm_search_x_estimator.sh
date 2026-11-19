#!/bin/bash
# Search x gradient estimator, on the retrained (d256) samplers of the MLGD* / GB-FESO table (tab:sim_gbfeso).
# That table has two corners of the 2 x 2 design: MLGD* (diffusion search, finite differences in the guided
# steps) and GB-FESO (MMD) (direct search, backpropagation). This script runs the missing ones, same sampler,
# MMD loss, data, budget (30 starts, 250 samples per search step, 3,000 per refinement step) and selection
# (the 5 lowest re-scored 5 x 5,000). MLGD* refines with backpropagation, so the arm that mirrors it is zo_pw.
#   gbfeso_zo_pw   GB-FESO (MMD), finite differences in phase 0, backpropagation in phase 1
#   gbfeso_zo_zo   GB-FESO (MMD), finite differences in both phases
#   mlgd_pw        the MLGD* recipe with backpropagation in the guided steps (--guide-grad pathwise)
# One array task per cell, arm and seed (42-46): task k -> cell k / 15, arm (k / 5) % 3, seed 42 + k % 5.
#   sbatch --array=0-29 experiments/synthetic/slurm/slurm_search_x_estimator.sh                  # (10,1), (10,10)
#   CELLS="x1_y1 x1_y5 x1_y10 x5_y1 x5_y5 x5_y10 x10_y1 x10_y5 x10_y10" sbatch --array=0-134%30 ...   # all nine
# Outputs runs/search_x_estimator/<cell>/<arm>_s<seed>.json (layout of search_mog.py); score them with
#   python experiments/synthetic/pipeline/summarize_runs.py --runs-dir runs/search_x_estimator/<cell> --data runs/table1v2/<cell>/data.pt
#SBATCH --job-name=search_x_est
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh             # Python environment (CDM_ENV)
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CELLS=(${CELLS:-x10_y1 x10_y10})
ARMS=(gbfeso_zo_pw gbfeso_zo_zo mlgd_pw)
k=${SLURM_ARRAY_TASK_ID:-0}
CELL=${CELLS[$((k / 15))]}
ARM=${ARMS[$(((k / 5) % 3))]}
S=$((42 + k % 5))
D=runs/table1v2/$CELL
if [ "$CELL" = x10_y1 ]; then SAMPLER=runs/night/x10_y1/models/d256/model_conditional.pt
else SAMPLER=runs/night/grid/$CELL/d256/model_conditional.pt; fi
OUTDIR=${OUTDIR:-runs/search_x_estimator}/$CELL; mkdir -p "$OUTDIR"
OUT=$OUTDIR/${ARM}_s$S.json
echo "search x estimator: $CELL $ARM seed $S ($(hostname) $(date))"
case $ARM in
  gbfeso_zo_*)
    python experiments/synthetic/pipeline/search_gbfeso.py --sampler "$SAMPLER" --target $D/data.pt --loss mmd \
      --grad zo --grad-phase1 $([ $ARM = gbfeso_zo_zo ] && echo zo || echo pathwise) \
      --n-restarts 30 --n-attempts 1 --seed $S \
      --num-samples 250 --num-samples-phase1 3000 --final-n 1000 --select-top 5 --select-n 5000 --select-repeats 5 \
      --device cuda --out "$OUT" ${EXTRA:-} ;;
  mlgd_pw)
    python experiments/synthetic/pipeline/search_mog.py --prior $D/models/uncond/model_unconditional.pt \
      --sampler "$SAMPLER" --target $D/data.pt --algo lgd --n-attempts 1 --n-cond 250 \
      --refine-lr 0.02 --guidance-optim adam --step-size 0.05 \
      --n-restarts 30 --guide-grad pathwise --refine-keep-best --refine-steps 100 --refine-n-cond 3000 \
      --select-top 5 --select-n 5000 --select-repeats 5 \
      --seed $S --device cuda --out "$OUT" ${EXTRA:-} ;;
esac
echo "done $(date)"
