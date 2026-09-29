#!/usr/bin/env bash
# (10,1) at 100 restarts (App. B.7)
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~1 min. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; ~10x a normal cell.
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
    "$PY" experiments/synthetic/pipeline/make_table1.py --settings 10x1 --methods MLGD-F MLGD-X \
      --n-components 4 --n-samples 500 --epochs 30 --batch-size 256 --nunits 16 --nblocks 1 --budget 4 --attempts 1 --n-cond 16 --refine-steps 2 --refine-n-cond 32 \
      --guidance-optim adam --step-size 0.05 \
      --tag _r100 --skip-check --seed 42 --device cpu \
      --runs-dir "$OUT/runs" --out "$OUT/table1_10x1_r100"
    ;;

  full)
    "$PY" experiments/synthetic/pipeline/make_table1.py --settings 10x1 --methods MLGD MLGD-F MLGD-X \
      --n-components 4 --n-samples 20000 --epochs 10000 --batch-size 1024 \
      --nunits 128 --nblocks 3 \
      --budget 100 --attempts 3 --n-cond 250 \
      --guidance-optim adam --step-size 0.05 \
      --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 \
      --tag _r100 --seed 42 --device cuda \
      --runs-dir "$OUT/runs" --out "$OUT/table1_10x1_r100"
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
