#!/bin/bash
# Screened start (search_mog --init screen) on the Table 1 grid, with Table 1's networks and flags.
# One array task per (scenario, finish): 9 scenarios x {refine, guided} = 18 tasks.
#   RUNS=<runs dir with data.pt + models/ per scenario> sbatch --array=0-17 experiments/synthetic/slurm/slurm_screen.sh
# Collect afterwards:
#   python experiments/synthetic/pipeline/make_table1.py --collect --methods MLGD MLGD-F --tag _screen_refine --runs-dir $RUNS --out results/table1_screen_refine
#SBATCH --job-name=cdm-screen
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/screen_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
[ -f experiments/synthetic/pipeline/make_table1.py ] || { echo "run sbatch from the repo root"; exit 1; }
VENV=${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
SETTINGS=(1x1 1x5 1x10 5x1 5x5 5x10 10x1 10x5 10x10)
FINISHES=(refine guided)
S=${SETTINGS[$((SLURM_ARRAY_TASK_ID % 9))]}
F=${FINISHES[$((SLURM_ARRAY_TASK_ID / 9))]}
N_SCREEN=${N_SCREEN:-1000}
echo "=== task $SLURM_ARRAY_TASK_ID: $S finish=$F n_screen=$N_SCREEN on $(hostname) $(date) ==="
python experiments/synthetic/pipeline/make_table1.py \
  --settings "$S" --methods MLGD MLGD-F \
  --budget 10 --attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05 \
  --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 \
  --seed 42 --device cuda --skip-check --no-collect \
  --runs-dir "$RUNS" --tag "_screen_$F" \
  --extra "--init screen --n-screen $N_SCREEN --screen-finish $F"
echo "=== done $(date) ==="
