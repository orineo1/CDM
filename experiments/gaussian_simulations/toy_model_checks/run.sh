#!/usr/bin/env bash
# Toy-setting network-quality check (App. B.8)
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~1.5 min. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; minutes.
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
    "$PY" scripts/fig1_toy_example.py \
      --quick --setting hard --device cpu \
      --ckpt-dir "$OUT/ckpt" --out "$OUT/fig1_hard"
    "$PY" scripts/check_models.py \
      --setting hard --ckpt-dir "$OUT/ckpt" --seed 42 \
      --n-samples 200 --n-random-x 2 --n-seeds 1 --n-t 3 --grid 5 \
      --device cpu --out "$OUT/check_toy_hard"
    ;;

  full)
    for s in hard paper; do
      "$PY" scripts/fig1_toy_example.py \
        --setting "$s" --device cuda --ckpt-dir "$OUT/ckpt" --out "$OUT/fig1_$s"
      "$PY" scripts/check_models.py \
        --setting "$s" --ckpt-dir "$OUT/ckpt" --seed 42 \
        --n-samples 20000 --n-random-x 100 --n-t 12 \
        --device cuda --out "$OUT/check_toy_$s"
    done
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
