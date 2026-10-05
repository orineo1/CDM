#!/bin/bash
# Train one network on fresh draws from a scenario's mixture.
#   ARCH=diffusion BATCHES=100000 UNITS=256 BLOCKS=4 OUTD=runs/night/x10_y1/models/d256 \
#   sbatch --job-name=tr_d256 --export=ALL experiments/synthetic/slurm/slurm_train_sampler.sh
# ARCH: diffusion | cm (conditional samplers) | unconditional (the prior). DATA defaults to the (10, 1) scenario.
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
VENV=${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv$(grep -q "^13" /etc/debian_version 2>/dev/null && echo _deb13)}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
MODEL=conditional; A=$ARCH
[ "$ARCH" = unconditional ] && { MODEL=unconditional; A=diffusion; }
mkdir -p "$OUTD"
echo "train $ARCH batches $BATCHES units $UNITS blocks $BLOCKS -> $OUTD ($(date))"
python experiments/synthetic/pipeline/train_models.py --data ${DATA:-runs/table1v2/x10_y1/data.pt} --model $MODEL \
  --architecture $A --epochs $BATCHES --batch-size ${BS:-1024} --nunits $UNITS --nblocks $BLOCKS --fresh-samples \
  --seed ${SEED:-7} --device cuda --output-dir "$OUTD"
echo "done $(date)"
