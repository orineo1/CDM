#!/usr/bin/env bash
# Table 9 + Figure 4: infeasible target (no x*).
#
#   bash run.sh smoke   CPU, ~1 min. Proves the path runs. NOT a reproduction.
#   bash run.sh full    the published configuration (see config/nonstar_params.txt
#                       and PROVENANCE.md for where every number comes from).
#
# Every flag is literal in this file.
set -euo pipefail
export PYTHONUNBUFFERED=1   # print lines reach the log as they happen, also under sbatch
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-python}"
OUT="$HERE/outputs"
TIER="${1:-smoke}"
mkdir -p "$OUT"
cd "$REPO"

PARAMS="$HERE/config/nonstar_params.txt"

case "$TIER" in
  smoke)
    EPOCHS=40; NUNITS=16; NBLOCKS=1; BATCH=256; NSAMP=400; ATTEMPTS=2; NBIG=16
    DEV=cpu
    ;;
  full)
    EPOCHS=20000; NUNITS=128; NBLOCKS=3; BATCH=1024; NSAMP=5000; ATTEMPTS=25; NBIG=250
    DEV=cuda
    ;;
  *) echo "usage: bash run.sh [smoke|full]" >&2; exit 64 ;;
esac

"$PY" experiments/synthetic/pipeline/generate_data.py \
  --params-file "$PARAMS" --condition-on 1 \
  --n-samples "$NSAMP" --seed 42 --output "$OUT/data.pt"

"$PY" experiments/synthetic/pipeline/train_models.py --data "$OUT/data.pt" \
  --model unconditional --architecture diffusion --fresh-samples \
  --epochs "$EPOCHS" --batch-size "$BATCH" --nunits "$NUNITS" --nblocks "$NBLOCKS" \
  --diffusion-steps 100 --seed 42 --device "$DEV" --output-dir "$OUT/models/uncond"

"$PY" experiments/synthetic/pipeline/train_models.py --data "$OUT/data.pt" \
  --model conditional --architecture cm --fresh-samples \
  --epochs "$EPOCHS" --batch-size "$BATCH" --nunits "$NUNITS" --nblocks "$NBLOCKS" \
  --diffusion-steps 100 --seed 42 --device "$DEV" --output-dir "$OUT/models/cm"

# The three arms of Table 9. Upstream runs one trajectory per restart with the Adam-form
# push at step 0.05 from x_T = 0, one Monte-Carlo perturbation, no refinement.
for ARM in "MMD_n$NBIG mmd $NBIG" "L2_n$NBIG l2 $NBIG" "L2_n1 l2 1"; do
  set -- $ARM
  echo "=== arm $1 (loss=$2, n_cond=$3) ==="
  "$PY" experiments/synthetic/pipeline/find_xstar.py \
    --prior "$OUT/models/uncond/model_unconditional.pt" \
    --sampler "$OUT/models/cm/model_conditional.pt" \
    --target "$OUT/data.pt" \
    --algo lgd --n-restarts 1 --n-attempts "$ATTEMPTS" \
    --n-mc 1 --init zeros --guidance-optim adam --step-size 0.05 \
    --refine-steps 0 --n-cond "$3" --loss "$2" \
    --seed 42 --device "$DEV" --out "$OUT/arm_$1.json"
done

echo "NOTE: the '% at x_bi' column of Table 9 is not emitted by any script in this repo;"
echo "      the per-attempt x_hat needed for it is in $OUT/arm_*.json. See README.md."
