#!/usr/bin/env bash
# MNIST figures: Figure 5 (top-5 digits + polar histograms) and Figures 6-8
# (all 15 seeds per target).
#
#   bash run.sh smoke   CPU, ~1 min. Renders every panel from whatever run
#                       records the smoke tier of ../table_mlgdf left behind.
#                       It does NOT reproduce a published figure.
#   bash run.sh full    the published configuration, over the full-tier records.
#
# Both tiers read .pkl run records; neither re-runs the optimiser.
set -euo pipefail
export PYTHONUNBUFFERED=1   # print lines reach the log as they happen, also under sbatch
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-python}"
OUT="$HERE/outputs"
TIER="${1:-smoke}"
mkdir -p "$OUT"

case "$TIER" in
  smoke)
    SRC="${2:-$HERE/../table_mlgdf/outputs/smoke}"
    [ -d "$SRC" ] || { echo "no run records at $SRC -- run ../table_mlgdf/run.sh smoke first" >&2; exit 65; }
    "$PY" "$HERE/make_figures.py" --results-dir "$SRC" --out "$OUT/smoke" \
      --device cpu --top-k 1 --n-cond-plot 32 --dpi 60
    ;;

  full)
    SRC="${2:-$HERE/../table_mlgdf/outputs/full}"
    [ -d "$SRC" ] || { echo "no run records at $SRC -- run ../table_mlgdf/run.sh full first" >&2; exit 65; }
    "$PY" "$HERE/make_figures.py" --results-dir "$SRC" --out "$OUT/full" \
      --device cuda --top-k 5 --n-cond-plot 500 --dpi 150 \
      --ckpt-dir "$REPO/experiments/mnist/checkpoints"
    ;;

  *) echo "usage: bash run.sh [smoke|full] [results-dir]" >&2; exit 64 ;;
esac
