#!/bin/bash
# Zeroth-order guidance (search_mog --guide-grad zo), Table 1 flags throughout.
#   task 0     : Stage 1 diagnostic, (10,1) with exact conditional samples, 10 trajectories x 5 attempts
#   tasks 1-18 : Stage 2, Table 1 grid (9 settings x {MLGD, MLGD-F}) with the Table 1 networks
#   RUNS=runs/screen sbatch --array=0-18 experiments/synthetic/slurm/slurm_zoguide.sh
# Collect afterwards:
#   python experiments/synthetic/pipeline/make_table1.py --collect --methods MLGD MLGD-F --tag _zoguide --runs-dir $RUNS --out results/table1_zoguide
#SBATCH --job-name=cdm-zoguide
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zoguide_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
[ -f experiments/synthetic/pipeline/make_table1.py ] || { echo "run sbatch from the repo root"; exit 1; }
. ./cluster_env.sh             # Python environment (VENV)
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RUNS=${RUNS:-runs/screen}
echo "=== task $SLURM_ARRAY_TASK_ID on $(hostname) $(date) ==="
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  D=runs/table1v2/x10_y1
  python experiments/synthetic/pipeline/search_mog.py --prior $D/models/uncond/model_unconditional.pt \
    --target $D/data.pt --sampler exact --algo lgd --n-cond 250 --guide-grad zo \
    --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 --guidance-optim adam --step-size 0.05 \
    --seed 42 --device cuda --n-restarts 10 --n-attempts 5 --out $D/MLGD-X_zoguide10.json
else
  SETTINGS=(1x1 1x5 1x10 5x1 5x5 5x10 10x1 10x5 10x10)
  METHODS=(MLGD MLGD-F)
  k=$((SLURM_ARRAY_TASK_ID - 1))
  S=${SETTINGS[$((k % 9))]}; M=${METHODS[$((k / 9))]}
  echo "setting $S method $M"
  python experiments/synthetic/pipeline/make_table1.py \
    --settings "$S" --methods $M \
    --budget 10 --attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05 \
    --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 \
    --seed 42 --device cuda --skip-check --no-collect \
    --runs-dir "$RUNS" --tag "_zoguide" --extra "--guide-grad zo"
fi
echo "=== done $(date) ==="
