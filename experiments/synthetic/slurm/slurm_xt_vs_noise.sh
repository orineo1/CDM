#!/bin/bash
# (10,1): start x_T versus guidance noise (diag_xT_vs_noise.py), Table 1 flags.
#   task 0: exact-loss oracle, 40 starts x 4 guidance seeds;  task 1: Table 1 sampler (diffusion_10k), 20 x 3
#   sbatch --array=0-1 experiments/synthetic/slurm/slurm_xt_vs_noise.sh
#SBATCH --job-name=cdm-xt
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/xt_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
VENV=${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv$(grep -q "^13" /etc/debian_version 2>/dev/null && echo _deb13)}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
D=${D:-runs/table1v2/x10_y1}
COMMON="--prior $D/models/uncond/model_unconditional.pt --target $D/data.pt --algo lgd --n-cond 250
        --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 --guidance-optim adam --step-size 0.05 --device cuda"
case "$SLURM_ARRAY_TASK_ID" in
  0) python experiments/synthetic/pipeline/diag_xT_vs_noise.py --n-xt 40 --n-guide 4 --diag-out results/xt_vs_noise_exactloss.json $COMMON --sampler exactloss ;;
  1) python experiments/synthetic/pipeline/diag_xT_vs_noise.py --n-xt 20 --n-guide 3 --diag-out results/xt_vs_noise_mlgd.json $COMMON --sampler $D/models/diffusion_10k/model_conditional.pt ;;
esac
