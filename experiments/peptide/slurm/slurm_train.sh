#!/bin/bash
# Step 2: train the networks (one array task each).   [N=5] sbatch --array=0-3 experiments/peptide/slurm/slurm_train.sh
#   0: GB-FESO surrogate with their train.py (their TrainConfig: 80 epochs, width 256, depth 7, 1000 steps)
#   1: our conditional diffusion P(y | x) (MLGD)          2: our consistency model (MLGD-F)
#   3: our prior over x (uniform box draws)
#SBATCH --job-name=pep-train
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/peptide/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
N=${N:-4}; [ "$N" = 4 ] && R0=runs/peptide || R0=runs/peptide_N$N; R=${R:-$R0}; P=experiments/peptide/pipeline; T=experiments/synthetic/pipeline/train_models.py
python -c "import torch; print(torch.cuda.get_device_name())"
[ "$N" = 4 ] && FRAMES=${FRAMES:-4000} || FRAMES=${FRAMES:-2000}
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) python $P/gbfeso.py train --train-ens $R/train_ens.pt --frames-per-eps $FRAMES --epochs ${GB_EPOCHS:-80} --out $R/models/gbfeso.pt ;;
  1) python $T --data $R/data.pt --model conditional --architecture diffusion --nunits 256 --nblocks 4 \
        --epochs ${EPOCHS:-40} --batch-size 512 --seed 0 --output-dir $R/models/diff ;;
  2) python $T --data $R/data.pt --model conditional --architecture cm --nunits 256 --nblocks 4 \
        --epochs ${EPOCHS:-40} --batch-size 512 --seed 0 --output-dir $R/models/cm ;;
  3) python $T --data $R/prior_data.pt --model unconditional --nunits 128 --nblocks 3 \
        --epochs 50 --batch-size 512 --seed 0 --output-dir $R/models/prior ;;
esac
echo "done $(date)"
