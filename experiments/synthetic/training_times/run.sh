#!/usr/bin/env bash
# Table 10: measured training wall-clock, legacy settings
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~30 s. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; 9 trainings, up to 4 h each.
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
    echo "params,network,model,architecture,seconds" > "$OUT/timings.csv"
    "$PY" experiments/synthetic/pipeline/generate_data.py \
      --params-file experiments/synthetic/configs/mog/mog_2d_full.txt --condition-on 1 \
      --n-samples 500 --seed 42 --output "$OUT/mog_2d/data.pt"
    for arch in "unconditional diffusion uncond" "conditional diffusion diffusion" "conditional cm cm"; do
      set -- $arch
      t0=$SECONDS
      "$PY" experiments/synthetic/pipeline/train_models.py --data "$OUT/mog_2d/data.pt" \
        --model "$1" --architecture "$2" \
        --epochs 2 --batch-size 256 --nunits 16 --nblocks 1 \
        --seed 42 --device cpu --output-dir "$OUT/mog_2d/models/$3"
      echo "TIMING mog_2d $1/$2 $((SECONDS - t0))s" | tee -a "$OUT/timings.txt"
      echo "mog_2d,$3,$1,$2,$((SECONDS - t0))" >> "$OUT/timings.csv"
    done
    ;;

  full)
    echo "params,network,model,architecture,seconds" > "$OUT/timings.csv"     # Table 10 (minutes = seconds / 60)
    # NOTE: no 5D parameter file exists in experiments/synthetic/configs/mog/; the 5D row of
    # Table 10 cannot be reproduced from this repository.
    for pf in mog_2d_full mog_10d_full ldm_10D_full; do
      "$PY" experiments/synthetic/pipeline/generate_data.py \
        --params-file "experiments/synthetic/configs/mog/$pf.txt" --condition-on 1 \
        --n-samples 20000 --seed 42 --output "$OUT/$pf/data.pt"
      for arch in "unconditional diffusion uncond" "conditional diffusion diffusion" "conditional cm cm"; do
        set -- $arch
        t0=$SECONDS
        "$PY" experiments/synthetic/pipeline/train_models.py --data "$OUT/$pf/data.pt" \
          --model "$1" --architecture "$2" \
          --epochs 10000 --batch-size 1024 --nunits 128 --nblocks 3 \
          --seed 42 --device cuda --output-dir "$OUT/$pf/models/$3"
        echo "TIMING $pf $1/$2 $((SECONDS - t0))s" | tee -a "$OUT/timings.txt"
        echo "$pf,$3,$1,$2,$((SECONDS - t0))" >> "$OUT/timings.csv"
      done
    done
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
