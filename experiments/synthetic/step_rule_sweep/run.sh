#!/usr/bin/env bash
# Constant-zeta vs Adam step-rule sweep (App. B.6)
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~2 min. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; ~20 grid cells.
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
    "$PY" experiments/synthetic/pipeline/make_table1.py --settings 1x1 --methods MLGD-F \
      --n-components 4 --n-samples 500 --epochs 30 --batch-size 256 --nunits 16 --nblocks 1 --budget 2 --attempts 1 --n-cond 16 --refine-steps 2 --refine-n-cond 32 --guidance-optim sgd --step-size 0.3 \
      --tag _const0.3 --skip-check --seed 42 --device cpu \
      --runs-dir "$OUT/runs" --out "$OUT/sweep_const_0.3"
    "$PY" experiments/synthetic/pipeline/make_table1.py --settings 1x1 --methods MLGD-F \
      --n-components 4 --n-samples 500 --epochs 30 --batch-size 256 --nunits 16 --nblocks 1 --budget 2 --attempts 1 --n-cond 16 --refine-steps 2 --refine-n-cond 32 --guidance-optim adam --step-size 0.05 \
      --tag _adam0.05 --skip-check --seed 42 --device cpu \
      --runs-dir "$OUT/runs" --out "$OUT/sweep_adam_0.05"
    ;;

  full)
    CFG="$HERE/config/zeta_grid.txt"
    for cell in 1x1 5x5 10x10; do
      while read -r rule zeta; do
        case "$rule" in ''|'#'*) continue ;; esac
        "$PY" experiments/synthetic/pipeline/make_table1.py --settings "$cell" --methods MLGD-F \
          --n-components 4 --n-samples 20000 --epochs 10000 --batch-size 1024 \
          --nunits 128 --nblocks 3 \
          --budget 10 --attempts 3 --n-cond 250 \
          --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 \
          --guidance-optim "$rule" --step-size "$zeta" \
          --tag "_${rule}${zeta}" --skip-check --seed 42 --device cuda \
          --runs-dir "$OUT/runs" --out "$OUT/sweep_${rule}_${zeta}"
      done < "$CFG"
    done
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
