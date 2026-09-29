#!/usr/bin/env bash
# Train the two frozen MNIST networks: the unconditional DDPM prior P(X) and
# the conditional iCT angle sampler P(Y|X). Paper Tables 11 and 12.
#
#   bash run.sh smoke   CPU, ~2 min. 1 epoch over 64 images for the prior and
#                       2 epochs over 32 images for the conditional model.
#                       It produces throwaway weights, not the published models.
#   bash run.sh full    the published configuration (Table 12). GPU; hours.
#
# The published checkpoints are downloaded by the other experiments; nothing
# here has to be run to reproduce Table 2.
set -euo pipefail
export PYTHONUNBUFFERED=1   # print lines reach the log as they happen, also under sbatch
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
OUT="$HERE/outputs"
DATA="${MNIST_DATA_ROOT:-$OUT/data}"
TIER="${1:-smoke}"
mkdir -p "$OUT"

case "$TIER" in
  smoke)
    "$PY" "$HERE/train_uncond.py" --epochs 1 --subset 64 --batch-size 32 \
      --num-workers 0 --device cpu --data-root "$DATA" \
      --ckpt-dir "$OUT/smoke/uncond" --plots-dir "$OUT/smoke/uncond_plots" \
      --checkpoint-every 1
    "$PY" "$HERE/train_conditional.py" --clf-epochs 1 --epochs 2 --subset 32 \
      --batch-size 16 --clf-batch 16 --device cpu --data-root "$DATA" \
      --ckpt-dir "$OUT/smoke/cond" --plots-dir "$OUT/smoke/cond_plots"
    ;;

  full)
    # Table 12: AdamW 1e-3, batch 256, 100 epochs, squaredcos_cap_v2, 1000 steps, MSE.
    "$PY" "$HERE/train_uncond.py" --epochs 100 --batch-size 256 --lr 1e-3 \
      --data-root "$DATA" --ckpt-dir "$OUT/full/uncond" \
      --plots-dir "$OUT/full/uncond_plots"
    # Table 12: AdamW 1e-4, batch 256, 500 epochs, s0=10 s1=1280, eps=0.002,
    # smooth Huber; construction classifier 15 epochs, threshold 0.9999.
    "$PY" "$HERE/train_conditional.py" --clf-epochs 15 --epochs 500 \
      --batch-size 256 --clf-batch 128 --threshold 0.9999 \
      --data-root "$DATA" --ckpt-dir "$OUT/full/cond" \
      --plots-dir "$OUT/full/cond_plots"
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
