#!/bin/bash
# MLGD* recipe with the KDE-KL loss (GB-FESO's) instead of the MMD on the synthetic grid: d256 samplers,
# --guide-grad zo, 30 restarts, keep-best refinement (100 Adam steps, 3,000 samples), reselection of the 5
# lowest-loss runs with 5,000 samples x 5; the same loss is used for guidance, refinement and selection.
# Nine cells x three seeds, one array task each (task k: cell k / 3, seed 42 + k % 3):
#   sbatch --array=0-26 experiments/synthetic/slurm/slurm_lossgrid.sh
# The MMD reference runs of the same recipe: runs/night/grid/<cell>/G_zo_s*.json (Table 1, MLGD*) and, for
# (10, 1), runs/zodirs/x10_y1/zodirs_coord_s*.json. Summary per cell:
#   python experiments/synthetic/pipeline/summarize_runs.py --runs-dir runs/lossgrid/<cell> --data runs/table1v2/<cell>/data.pt
#SBATCH --job-name=lossgrid
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
CELLS=(x1_y1 x1_y5 x1_y10 x5_y1 x5_y5 x5_y10 x10_y1 x10_y5 x10_y10)
k=${SLURM_ARRAY_TASK_ID:-0}
C=${CELLS[$((k / 3))]}
S=$((42 + k % 3))
D=runs/table1v2/$C
[ "$C" = x10_y1 ] && SAMPLER=runs/night/x10_y1/models/d256/model_conditional.pt || SAMPLER=runs/night/grid/$C/d256/model_conditional.pt
LOSS=${LOSS:-"--loss kde-kl --kde-bandwidth 0.5"}
TAG=${TAG:-kdekl05}
OUTDIR=${OUTDIR:-runs/lossgrid/$C}; mkdir -p "$OUTDIR"
echo "lossgrid $C seed $S: $LOSS ($(hostname) $(date))"
python experiments/synthetic/pipeline/search_mog.py --prior $D/models/uncond/model_unconditional.pt \
  --sampler "$SAMPLER" --target $D/data.pt --algo lgd --n-attempts 1 --n-cond 250 \
  --refine-lr 0.02 --guidance-optim adam --step-size 0.05 \
  --n-restarts 30 --guide-grad zo --refine-keep-best --refine-steps 100 --refine-n-cond 3000 \
  --select-top 5 --select-n 5000 --select-repeats 5 $LOSS \
  --seed $S --device cuda --out "$OUTDIR/${TAG}_s$S.json" ${EXTRA:-}
echo "done $(date)"
