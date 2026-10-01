#!/bin/bash
# Step 3a: our method (find_xstar.py) on the targets, one array task per block of targets.
#   SAMPLER=diff sbatch --array=0-7 experiments/peptide/slurm/slurm_ours.sh    # MLGD   (conditional diffusion)
#   SAMPLER=cm   sbatch --array=0-7 experiments/peptide/slurm/slurm_ours.sh    # MLGD-F (consistency model)
# Task k handles targets k*PER .. k*PER+PER-1. Output runs/peptide/results/<MLGD|MLGD-F>_t<id>.json
#SBATCH --job-name=pep-ours
#SBATCH --partition=catfish        # all methods on the same GPU type (L4) so that wall times compare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/peptide/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=${R:-runs/peptide}; SAMPLER=${SAMPLER:-diff}; PER=${PER:-5}; OUT=${OUT:-$R/results}; mkdir -p $OUT
NAME=$([ "$SAMPLER" = cm ] && echo MLGD-F || echo MLGD)
python -c "import torch; print(torch.cuda.get_device_name())"
k=${SLURM_ARRAY_TASK_ID:-0}
for t in $(seq $((k * PER)) $((k * PER + PER - 1))); do
  tid=$(printf %03d $t); [ -f $R/targets/target_$tid.npy ] || continue
  [ -f $OUT/${NAME}${TAG:-}_t$tid.json ] && continue
  python experiments/synthetic/pipeline/find_xstar.py --prior $R/models/prior/model_unconditional.pt \
    --sampler $R/models/$SAMPLER/model_conditional.pt --target $R/targets/target_$tid.npy \
    --algo lgd --n-attempts 1 --n-cond 250 --guidance-optim adam --step-size 0.05 --guide-grad zo \
    --n-restarts ${RESTARTS:-10} --refine-steps 100 --refine-lr 0.02 --refine-n-cond 3000 --refine-keep-best \
    --select-top 5 --select-n 2000 --select-repeats 5 --seed $((42 + t)) --device cuda \
    --out $OUT/${NAME}${TAG:-}_t$tid.json ${EXTRA:-}
done
echo "done $(date)"
