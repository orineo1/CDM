#!/usr/bin/env bash
# Figure 3: twisted-SMC correction of the beta-sweep
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~1 min. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; tens of minutes. Reads the toy
#                       networks of ../toy_example (CKPT_DIR), training them if missing.
#                       ../toy_example/run.sh full calls this script, so one job does both.
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
    # the toy networks Figure 3 reads are trained by the Figure 1 script
    "$PY" experiments/synthetic/pipeline/fig1_toy_example.py \
      --quick --setting hard --device cpu \
      --ckpt-dir "$OUT/ckpt" --out "$OUT/fig1_hard"
    # NOTE: four betas, not one -- a single --betas value crashes the script (see README)
    "$PY" experiments/synthetic/pipeline/fig_smc_correction.py \
      --setting hard --ckpt-dir "$OUT/ckpt" \
      --betas 0 2 8 16 --n-particles 100 --n-repeats 1 \
      --max-shift 0.25 --ess-frac 1.0 --bins 80 \
      --device cpu --out "$OUT/fig_smc_hard"
    ;;

  full)
    # The toy networks of the hard setting: those trained by ../toy_example/run.sh full
    # (which calls this script), or trained here on first use with the same configuration.
    CKPT="${CKPT_DIR:-$HERE/../toy_example/outputs/ckpt}"
    if [ ! -f "$CKPT/hard/Diffusion_uncond_seed42.pt" ] || [ ! -f "$CKPT/hard/Diffusion_cond_seed42.pt" ] || [ ! -f "$CKPT/hard/CM_seed42.pt" ]; then
      "$PY" experiments/synthetic/pipeline/fig1_toy_example.py \
        --setting hard --n-restarts 10 --n-cond 250 --n-mc 3 \
        --methods MLGD MLGD-F PMLGD PMLGD-F \
        --betas 0 2 8 16 --n-beta-samples 2000 \
        --rerun-sim --rerun-sweep --device cuda \
        --ckpt-dir "$CKPT" --out "$OUT/fig1_hard"
    fi
    "$PY" experiments/synthetic/pipeline/fig_smc_correction.py \
      --setting hard --ckpt-dir "$CKPT" \
      --betas 0 2 8 16 --n-particles 2000 --n-repeats 10 \
      --max-shift 0.25 --ess-frac 1.0 --bins 80 \
      --device cuda --out "$OUT/fig_smc_hard"
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
