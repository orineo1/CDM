#!/bin/bash
# Step 3b: the GB-FESO baseline (their run_feso.optimize_epsilon_condition), 4 starts per target.
#   sbatch --array=0-7 experiments/peptide/slurm/slurm_gbfeso.sh      # task k: targets k*PER .. k*PER+PER-1
#SBATCH --job-name=pep-gbfeso
#SBATCH --partition=catfish        # all methods on the same GPU type (L4) so that wall times compare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/peptide/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv$(grep -q "^13" /etc/debian_version 2>/dev/null && echo _deb13)}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
N=${N:-4}; [ "$N" = 4 ] && R0=runs/peptide || R0=runs/peptide_N$N; R=${R:-$R0}; PER=${PER:-5}; OUT=${OUT:-$R/results}; mkdir -p $OUT
python -c "import torch; print(torch.cuda.get_device_name())"
k=${SLURM_ARRAY_TASK_ID:-0}
python experiments/peptide/pipeline/gbfeso.py opt --ckpt $R/models/gbfeso.pt --targets-ens $R/targets_ens.pt \
  --target-ids $((k * PER))-$((k * PER + PER - 1)) --plot-dir /tmp/gbfeso_$SLURM_JOB_ID --out $OUT/gbfeso${TAG:-}_$k.json ${EXTRA:-}
echo "done $(date)"
