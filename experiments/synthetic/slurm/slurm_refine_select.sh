#!/bin/bash
# (10,1): redo refinement and selection on Table 1's saved guided end points (refine_endpoints.py).
# Tasks: 0-2 MLGD (Table 1 sampler, diffusion_10k), 3-5 MLGD-F; variants per method:
#   a = pathwise refinement + reselection (top 3 re-estimated as the mean of 5 x 2,000 samples)
#   b = zeroth-order refinement, standard selection
#   c = zeroth-order refinement + reselection
#   sbatch --array=0-5 experiments/synthetic/slurm/slurm_refine_select.sh
#SBATCH --job-name=cdm-refsel
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/refsel_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh             # Python environment (CDM_ENV)
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
D=${D:-runs/table1v2/x10_y1}
M=("MLGD" "MLGD" "MLGD" "MLGD-F" "MLGD-F" "MLGD-F")
V=("a" "b" "c" "a" "b" "c")
m=${M[$SLURM_ARRAY_TASK_ID]}; v=${V[$SLURM_ARRAY_TASK_ID]}
# TASKS: optional subset, e.g. rerun only the reselection variants: sbatch --array=0,2,3,5 ...
S=""; [ "$m" = "MLGD" ] && S="--sampler $D/models/diffusion_10k/model_conditional.pt"
case "$v" in
  a) F="--refine-grad pathwise --select-top 3 --select-n 2000 --select-repeats 5" ;;
  b) F="--refine-grad zo" ;;
  c) F="--refine-grad zo --select-top 3 --select-n 2000 --select-repeats 5" ;;
esac
echo "=== $m variant $v on $(hostname) $(date)"
python experiments/synthetic/pipeline/refine_endpoints.py --result $D/${m}_adam_refine.json $S $F --device cuda --out $D/${m}_refsel_$v.json
echo "done $(date)"
