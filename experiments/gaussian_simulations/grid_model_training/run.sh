#!/usr/bin/env bash
# The 27 grid network trainings (App. B.4)
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~30 s. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; 27 trainings.
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
    "$PY" scripts/generate_data.py \
      --random-mog --dim 2 --condition-on 1 --n-components 4 --standardize \
      --n-samples 500 --seed 42 --output "$OUT/x1_y1/data.pt"
    for arch in "unconditional diffusion uncond" "conditional diffusion diffusion" "conditional cm cm"; do
      set -- $arch
      "$PY" scripts/train_models.py --data "$OUT/x1_y1/data.pt" \
        --model "$1" --architecture "$2" --fresh-samples \
        --epochs 30 --batch-size 256 --nunits 16 --nblocks 1 \
        --diffusion-steps 100 --guidance-dropout 0.2 \
        --seed 42 --device cpu --output-dir "$OUT/x1_y1/models/$3"
    done
    ;;

  full)
    for cell in 1:1 1:5 1:10 5:1 5:5 5:10 10:1 10:5 10:10; do
      dx="${cell%%:*}"; dy="${cell##*:}"
      s="x${dx}_y${dy}"
      "$PY" scripts/generate_data.py \
        --random-mog --dim $((dx + dy)) --condition-on "$dx" \
        --n-components 4 --standardize \
        --n-samples 20000 --seed 42 --output "$OUT/$s/data.pt"
      for arch in "unconditional diffusion uncond" "conditional diffusion diffusion" "conditional cm cm"; do
        set -- $arch
        "$PY" scripts/train_models.py --data "$OUT/$s/data.pt" \
          --model "$1" --architecture "$2" --fresh-samples \
          --epochs 10000 --batch-size 1024 --nunits 128 --nblocks 3 \
          --diffusion-steps 100 --guidance-dropout 0.2 \
          --seed 42 --device cuda --output-dir "$OUT/$s/models/$3"
      done
    done
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
