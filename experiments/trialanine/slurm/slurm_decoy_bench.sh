#!/bin/bash
# Deceptive-target benchmark, step 2: the comparison on the exact loss, one task per selected target:
#   sbatch --array=0-<n-1> experiments/trialanine/slurm/slurm_decoy_bench.sh
# Methods: MLGD-exact, MLGD-exact+polish, GBFESO-exact-x0, GBFESO-exact-rand4, GBFESO-exact-randEq (random starts
# matching MLGD's loss evaluations). Output runs/trialanine/decoy/results/<method>_tXXX.json
#SBATCH --job-name=tri-dbench
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --exclude=salmon-02,salmon-09
#SBATCH --output=logs/trialanine/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
D=${D:-runs/trialanine/decoy}; O=$D/results; mkdir -p $O
WP=$(python -c "import json; print(json.load(open('$D/targets.json'))['wP'])")
k=${SLURM_ARRAY_TASK_ID:-0}; tid=$(printf %03d $k)
X="python -W ignore experiments/trialanine/pipeline/search_exact_tri.py"
C="--meta $D/targets.json --target $D/dummy_target.npy --target-id $k --wP $WP --seed $((42 + k))"
[ -f $O/MLGD-exact_t$tid.json ] || $X mlgd $C --out $O/MLGD-exact_t$tid.json --method-name MLGD-exact \
  --prior runs/trialanine/cdm/models/prior/model_unconditional.pt --algo lgd --n-attempts 1 --n-cond 1 --guidance-optim adam \
  --step-size 0.05 --guide-grad pathwise --n-restarts 10 --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1 --refine-keep-best \
  --select-top 5 --select-n 1 --select-repeats 1 --x-box -2 2
[ -f $O/MLGD-exact+polish_t$tid.json ] || $X gbfeso $C --init json --init-json $O/MLGD-exact_t$tid.json --init-top 3 \
  --method-name MLGD-exact+polish --out $O/MLGD-exact+polish_t$tid.json
[ -f $O/GBFESO-exact-x0_t$tid.json ] || $X gbfeso $C --init zero --method-name GBFESO-exact-x0 --out $O/GBFESO-exact-x0_t$tid.json
[ -f $O/GBFESO-exact-rand4_t$tid.json ] || $X gbfeso $C --init randn --n-starts 4 --method-name GBFESO-exact-rand4 --out $O/GBFESO-exact-rand4_t$tid.json
[ -f $O/GBFESO-exact-randEq_t$tid.json ] || $X gbfeso $C --init randn --match-evals $O/MLGD-exact_t$tid.json \
  --method-name GBFESO-exact-randEq --out $O/GBFESO-exact-randEq_t$tid.json
echo "done $(date)"
