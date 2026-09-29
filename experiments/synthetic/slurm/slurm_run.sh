#!/usr/bin/env bash
# One synthetic experiment as one GPU job: runs experiments/synthetic/<experiment>/run.sh full,
# so every flag lives in that run.sh. Submit from the repository root:
#
#   sbatch experiments/synthetic/slurm/slurm_run.sh smc_correction
#   sbatch experiments/synthetic/slurm/slurm_run.sh toy_example          # after smc_correction (panel d)
#   sbatch experiments/synthetic/slurm/slurm_run.sh infeasible_target
#   sbatch experiments/synthetic/slurm/slurm_run.sh training_times
#   sbatch experiments/synthetic/slurm/slurm_run.sh model_quality        # after main_grid (reads its networks)
#   sbatch experiments/synthetic/slurm/slurm_run.sh ablations            # after main_grid (reads its networks)
#
# main_grid itself is the array job slurm_table1.sh. Logs: <experiment>-<jobid>.out in the
# submit directory. Override the partition or time with sbatch flags, e.g.
#   sbatch --time=24:00:00 experiments/synthetic/slurm/slurm_run.sh ablations
#
#SBATCH --job-name=cdm-synthetic
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
EXP="${1:?usage: sbatch slurm_run.sh <experiment>   (a directory under experiments/synthetic/)}"
[ -f "experiments/synthetic/$EXP/run.sh" ] || { echo "no experiments/synthetic/$EXP/run.sh; run sbatch from the repo root"; exit 1; }
VENV=${VENV:-/sci/labs/orzuk/ori_m/venvs/cdm}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

echo "=== $EXP full on $(hostname), $(date) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
bash "experiments/synthetic/$EXP/run.sh" full
echo "=== done $(date) ==="
