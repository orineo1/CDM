#!/usr/bin/env bash
# Figure 1: the 1-D toy example
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~1 min. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; minutes per setting.
#
# Every flag is literal in this file; nothing is read from a hidden default.
set -euo pipefail
export PYTHONUNBUFFERED=1   # print lines reach the log as they happen, also under sbatch
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-python}"
OUT="$HERE/outputs"
TIER="${1:-smoke}"
mkdir -p "$OUT"
cd "$REPO"

case "$TIER" in
  smoke)
    "$PY" experiments/synthetic/pipeline/fig1_toy_example.py \
      --quick --setting hard --device cpu \
      --ckpt-dir "$OUT/ckpt" --out "$OUT/fig1_hard"
    ;;

  full)
    for s in hard bimodal paper; do
      "$PY" experiments/synthetic/pipeline/fig1_toy_example.py \
        --setting "$s" --n-restarts 25 --n-cond 250 --n-mc 3 \
        --methods MLGD MLGD-F PMLGD PMLGD-F \
        --betas 0 2 8 16 --n-beta-samples 2000 \
        --rerun-sim --rerun-sweep --device cuda \
        --ckpt-dir "$OUT/ckpt" --out "$OUT/fig1_$s"
    done
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
