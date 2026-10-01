#!/bin/bash
# One search_mog variant at (10, 1) over seeds 42, 43, ... (one array task per seed), Table 1 flags as defaults;
# EXTRA overrides them (argparse keeps the last value). Outputs <OUTDIR>/<NAME>_s<seed>.json.
#   NAME=best SAMPLER=runs/night/x10_y1/models/d256/model_conditional.pt \
#   EXTRA="--n-restarts 30 --guide-grad zo --refine-keep-best --refine-steps 100 --refine-n-cond 3000 --select-top 5" \
#   sbatch --array=0-4 --job-name=best --export=ALL experiments/synthetic/slurm/slurm_search.sh
# Summarize: python experiments/synthetic/pipeline/summarize_runs.py --runs-dir runs/night/x10_y1 --data runs/table1v2/x10_y1/data.pt
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
VENV=${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
D=${D:-runs/table1v2/x10_y1}
S=$((42 + ${SLURM_ARRAY_TASK_ID:-0}))
OUTDIR=${OUTDIR:-runs/night/x10_y1}; mkdir -p "$OUTDIR"
echo "$NAME seed $S sampler $SAMPLER extra: ${EXTRA:-} ($(hostname) $(date))"
python experiments/synthetic/pipeline/search_mog.py --prior ${PRIOR:-$D/models/uncond/model_unconditional.pt} \
  --sampler "$SAMPLER" --target $D/data.pt --algo lgd --n-attempts 1 --n-cond 250 \
  --refine-lr 0.02 --refine-n-cond 1000 --guidance-optim adam --step-size 0.05 \
  --seed $S --device cuda --out "$OUTDIR/${NAME}_s$S.json" ${EXTRA:-}
echo "done $(date)"
