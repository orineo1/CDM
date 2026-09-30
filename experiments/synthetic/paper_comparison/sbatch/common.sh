# Sourced by every job in this directory. Runs from the repository root (sbatch is called from there).
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"
[ -f experiments/synthetic/pipeline/make_table1.py ] || { echo "submit from the repository root" >&2; exit 1; }
mkdir -p logs
# Python environment: set VENV (a venv dir) or CONDA_ENV before submitting; otherwise the submitting shell's python is used.
if [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ]; then . "$VENV/bin/activate"; fi
if [ -n "${CONDA_ENV:-}" ]; then eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"; fi
SYN=experiments/synthetic
GRID_RUNS=$SYN/main_grid/outputs/runs          # shared by Table 1, Table 4 and the Table 5 ablation arms
ALL_SETTINGS="1x1 1x5 1x10 5x1 5x5 5x10 10x1 10x5 10x10"
echo "=== $(date) | job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:--} | $(hostname)"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
