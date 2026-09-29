#!/usr/bin/env bash
# Table 5: network validation against the analytic mixture
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~1 min. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; minutes per scenario.
#
# Every flag is literal in this file; nothing is read from a hidden default.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-python}"
OUT="$HERE/outputs"
TIER="${1:-smoke}"
mkdir -p "$OUT"
cd "$REPO"

case "$TIER" in
  smoke)
    "$PY" experiments/synthetic/pipeline/generate_data.py \
      --random-mog --dim 2 --condition-on 1 --n-components 4 --standardize \
      --n-samples 500 --seed 42 --output "$OUT/data.pt"
    "$PY" experiments/synthetic/pipeline/train_models.py --data "$OUT/data.pt" \
      --model unconditional --architecture diffusion --fresh-samples \
      --epochs 30 --batch-size 256 --nunits 16 --nblocks 1 --seed 42 --device cpu \
      --output-dir "$OUT/models/uncond"
    "$PY" experiments/synthetic/pipeline/train_models.py --data "$OUT/data.pt" \
      --model conditional --architecture diffusion --fresh-samples \
      --epochs 30 --batch-size 256 --nunits 16 --nblocks 1 --seed 42 --device cpu \
      --output-dir "$OUT/models/diffusion"
    "$PY" experiments/synthetic/pipeline/train_models.py --data "$OUT/data.pt" \
      --model conditional --architecture cm --fresh-samples \
      --epochs 30 --batch-size 256 --nunits 16 --nblocks 1 --seed 42 --device cpu \
      --output-dir "$OUT/models/cm"
    "$PY" experiments/synthetic/pipeline/check_models.py \
      --prior "$OUT/models/uncond/model_unconditional.pt" \
      --sampler "$OUT/models/diffusion/model_conditional.pt" \
                "$OUT/models/cm/model_conditional.pt" \
      --data "$OUT/data.pt" \
      --n-samples 200 --n-random-x 2 --n-seeds 1 --n-t 3 --grid 5 \
      --device cpu --out "$OUT/check_x1_y1"
    ;;

  full)
    RUNS_DIR="${RUNS_DIR:-runs/table1v2}"
    for s in x1_y1 x1_y5 x1_y10 x5_y1 x5_y5 x5_y10 x10_y1 x10_y5 x10_y10; do
      [ -d "$RUNS_DIR/$s" ] || { echo "missing $RUNS_DIR/$s -- run main_grid first" >&2; exit 1; }
      "$PY" experiments/synthetic/pipeline/check_models.py \
        --prior "$RUNS_DIR/$s/models/uncond/model_unconditional.pt" \
        --sampler "$RUNS_DIR/$s/models/diffusion/model_conditional.pt" \
                  "$RUNS_DIR/$s/models/cm/model_conditional.pt" \
        --data "$RUNS_DIR/$s/data.pt" \
        --n-random-x 100 --device cuda --out "$OUT/checks__$s"
    done
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
