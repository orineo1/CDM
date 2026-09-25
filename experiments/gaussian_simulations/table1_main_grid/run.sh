#!/usr/bin/env bash
# Table 1 + Table 7: the 9 x 4 Gaussian grid
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~2 min. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; ~1 day for the full grid.
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
    "$PY" scripts/make_table1.py \
      --settings 1x1 \
      --methods MLGD MLGD-F PMLGD PMLGD-F \
      --n-components 4 --n-samples 500 --epochs 30 --batch-size 256 --nunits 16 --nblocks 1 \
      --budget 2 --attempts 1 --n-cond 16 --refine-steps 2 --refine-n-cond 32 \
      --guidance-optim adam --step-size 0.05 \
      --skip-check --seed 42 --device cpu \
      --runs-dir "$OUT/runs" --out "$OUT/table1"
    ;;

  full)
    "$PY" scripts/make_table1.py \
      --settings 1x1 1x5 1x10 5x1 5x5 5x10 10x1 10x5 10x10 \
      --methods MLGD MLGD-F PMLGD PMLGD-F \
      --n-components 4 --n-samples 20000 --epochs 10000 --batch-size 1024 \
      --nunits 128 --nblocks 3 \
      --budget 10 --attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05 --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 \
      --resample-beta 5.0 --resample-schedule constant \
      --seed 42 --device cuda \
      --runs-dir "$OUT/runs" --out "$OUT/table1"
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
