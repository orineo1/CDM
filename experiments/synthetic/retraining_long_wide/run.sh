#!/usr/bin/env bash
# Longer / wider retraining study (App. B.6)
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~2 min. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; 24 trainings + 8 grids.
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
    "$PY" experiments/synthetic/pipeline/make_table1.py --settings 5x5 --methods MLGD-F \
      --n-components 4 --n-samples 500 --epochs 60 --batch-size 256 \
      --nunits 16 --nblocks 1 --budget 2 --attempts 1 --n-cond 16 --refine-steps 2 --refine-n-cond 32 \
      --guidance-optim adam --step-size 0.05 \
      --tag _long --skip-check --seed 42 --device cpu \
      --runs-dir "$OUT/runs_long" --out "$OUT/table1_long"
    "$PY" experiments/synthetic/pipeline/make_table1.py --settings 5x5 --methods MLGD-F \
      --n-components 4 --n-samples 500 --epochs 30 --batch-size 256 \
      --nunits 32 --nblocks 1 --budget 2 --attempts 1 --n-cond 16 --refine-steps 2 --refine-n-cond 32 \
      --guidance-optim adam --step-size 0.05 \
      --tag _wide --skip-check --seed 42 --device cpu \
      --runs-dir "$OUT/runs_wide" --out "$OUT/table1_wide"
    ;;

  full)
    CELLS="5x5 5x10 10x5 10x10"
    "$PY" experiments/synthetic/pipeline/make_table1.py --settings $CELLS --methods MLGD MLGD-F \
      --n-components 4 --n-samples 20000 --epochs 50000 --batch-size 1024 \
      --nunits 128 --nblocks 3 --budget 10 --attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05 --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 \
      --tag _long --seed 42 --device cuda \
      --runs-dir "$OUT/runs_long" --out "$OUT/table1_long"
    "$PY" experiments/synthetic/pipeline/make_table1.py --settings $CELLS --methods MLGD MLGD-F \
      --n-components 4 --n-samples 20000 --epochs 10000 --batch-size 1024 \
      --nunits 256 --nblocks 3 --budget 10 --attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05 --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 \
      --tag _wide --seed 42 --device cuda \
      --runs-dir "$OUT/runs_wide" --out "$OUT/table1_wide"
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
