#!/usr/bin/env bash
# Figure 1: the 1-D toy example
#
# Two tiers, and they are not interchangeable:
#   bash run.sh smoke   CPU, ~1 min. Proves the code path runs end to end.
#                       It does NOT reproduce any published number.
#   bash run.sh full    the published configuration. GPU; minutes per setting, plus the
#                       twisted-SMC correction (../smc_correction, tens of minutes) that
#                       panel (d) of the hard setting overlays.
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
    # 1. the three settings: train the toy networks (cached under outputs/ckpt/<setting>/),
    #    ten trajectories per method, the beta sweep, and the figure
    for s in hard bimodal paper; do
      "$PY" experiments/synthetic/pipeline/fig1_toy_example.py \
        --setting "$s" --n-restarts 10 --n-cond 250 --n-mc 3 \
        --methods MLGD MLGD-F PMLGD PMLGD-F \
        --betas 0 2 8 16 --n-beta-samples 2000 \
        --rerun-sim --rerun-sweep --device cuda \
        --ckpt-dir "$OUT/ckpt" --out "$OUT/fig1_$s"
    done
    # 2. the twisted-SMC correction of the hard setting's beta sweep, on the same networks
    CKPT_DIR="$OUT/ckpt" bash "$HERE/../smc_correction/run.sh" full
    # 3. redraw the hard figure with the SMC populations in panel (d) (the paper's
    #    fig1_toy_hard_smc.png); the simulation is read from the cache written in step 1
    "$PY" experiments/synthetic/pipeline/fig1_toy_example.py \
      --setting hard --device cuda --ckpt-dir "$OUT/ckpt" --out "$OUT/fig1_hard" \
      --smc-npz "$HERE/../smc_correction/outputs/fig_smc_hard.npz"
    ;;

  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac
