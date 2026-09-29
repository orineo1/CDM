#!/usr/bin/env bash
# Submit the whole grid for one target as a SLURM array, then a small job that ranks
# the finished configurations once every array task has ended.
#   bash experiments/mnist/grid_search/submit.sh uniform
#   bash experiments/mnist/grid_search/submit.sh bimodal
# Edit grids/<target>.args to change the grid. WANDB_MODE=offline|disabled is passed on.
set -euo pipefail
TARGET="${1:?usage: bash submit.sh <uniform|bimodal>}"
cd "$(dirname "$0")/../../.."
VENV=${VENV:-/sci/labs/orzuk/ori_m/venvs/cdm}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
mkdir -p logs
# shellcheck disable=SC2046
N=$(python experiments/mnist/grid_search/grid_search.py --target "$TARGET" \
      $(cat "experiments/mnist/grid_search/grids/$TARGET.args") --count)
echo "$TARGET: $N configurations"
A=$(sbatch --parsable --job-name="grid-$TARGET" --array="0-$((N - 1))" \
      --export=ALL experiments/mnist/grid_search/slurm.sh "$TARGET")
echo "array job $A"
S=$(sbatch --parsable --dependency="afterany:$A" --job-name="rank-$TARGET" --partition=catfish \
      --gres=gpu:1 --cpus-per-task=1 --mem=2G --time=00:10:00 \
      --output="logs/rank-$TARGET-%j.out" --error="logs/rank-$TARGET-%j.err" \
      --wrap "bash -c '. $VENV/bin/activate && python experiments/mnist/grid_search/summarize.py --target $TARGET'")
echo "ranking job $S (runs when every array task has ended)"
