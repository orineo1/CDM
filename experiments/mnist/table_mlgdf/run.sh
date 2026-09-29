#!/usr/bin/env bash
# MLGD-F on rotated MNIST -- the three angle targets of paper Table 2.
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~1 min. Proves the code path runs end to end on a
#                       4-step schedule with an untrained prior. It does NOT
#                       reproduce any published number.
#   bash run.sh full    the configuration in config/targets.tsv. GPU.
#                       CLASSIFIER=<path.pth> picks the digit classifier (default: the
#                       paper's checkpoints/robust_classifier.pth; a missing file is
#                       trained by the code and saved there).
#
# Every flag is literal in this file or in config/targets.tsv.
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
    for target in uniform bimodal unimodal; do
      "$PY" "$HERE/run_mlgdf.py" --target "$target" \
        --steps 4 --n-cond 32 --n-mc 2 --step-size-mode original \
        --n-seeds 1 --device cpu --no-classify --allow-untrained-prior \
        --out "$OUT/smoke"
    done
    ;;

  full)
    while read -r target n_cond steps n_mc mode variance; do
      case "$target" in ''|'#'*) continue ;; esac
      extra=()
      [ "$target" = "unimodal" ] && extra=(--unimodal-var "$variance")
      [ "$target" = "bimodal" ]  && extra=(--bimodal-var  "$variance")
      "$PY" "$HERE/run_mlgdf.py" --target "$target" \
        --steps "$steps" --n-cond "$n_cond" --n-mc "$n_mc" \
        --step-size-mode "$mode" "${extra[@]}" \
        --n-seeds 15 --device cuda --threshold 0.75 --no-clamp \
        --ckpt-dir "$REPO/experiments/mnist/checkpoints" \
        --classifier-path "${CLASSIFIER:-$REPO/experiments/mnist/checkpoints/robust_classifier.pth}" \
        --out "$OUT/full"
    done < "$HERE/config/targets.tsv"
    "$PY" "$HERE/make_table.py" "$OUT/full"      # prints Table 2, writes outputs/full/table2.md
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
