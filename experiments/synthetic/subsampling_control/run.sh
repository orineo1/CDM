#!/usr/bin/env bash
# Appendix G control: gradient subsampling vs full backprop
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~1.5 min. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; ~1 h.
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
    SIM_DIR="${CMP_SIM_DIR:-$HOME/github/conditional-matching-paper/simulations}"
    [ -d "$SIM_DIR/src" ] || { echo "set CMP_SIM_DIR to conditional-matching-paper/simulations" >&2; exit 1; }
    "$PY" experiments/synthetic/pipeline/control_subsample.py \
      --sim-dir "$SIM_DIR" --device cpu \
      --n-restarts 1 --configs all_250 uniform_detach --init zeros \
      --out "$OUT/control_subsample_10d.json"
    ;;

  full)
    SIM_DIR="${CMP_SIM_DIR:-$HOME/github/conditional-matching-paper/simulations}"
    [ -d "$SIM_DIR/src" ] || { echo "set CMP_SIM_DIR to conditional-matching-paper/simulations" >&2; exit 1; }
    "$PY" experiments/synthetic/pipeline/control_subsample.py \
      --sim-dir "$SIM_DIR" --device cuda \
      --n-restarts 25 \
      --configs all_250 uniform_detach uniform_detach_rs uniform_subset all_125 all_250_halfzeta \
      --init zeros \
      --out "$OUT/control_subsample_10d.json"
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
