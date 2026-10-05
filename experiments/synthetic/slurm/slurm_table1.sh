#!/usr/bin/env bash
# Table 1 grid on the cluster: one array task per (dim X, dim Y) scenario, then one collect.
#
#   sbatch experiments/synthetic/slurm/slurm_table1.sh                                   # Table 1 as published
#   SETTINGS="5x5 10x10" sbatch --array=0-1 experiments/synthetic/slurm/slurm_table1.sh  # a subset (array size = number of settings)
#   TAG=_detach ARGS="" sbatch experiments/synthetic/slurm/slurm_table1.sh               # a variant: see ../README.md
#
# Each task generates its scenario's data, trains the three networks, runs check_models.py
# and the search algorithms under $RUNS/<scenario>/ (steps whose outputs exist are skipped,
# so a task can be resubmitted and variants reuse the networks). The last array task
# collects $OUT.{json,md,tex} plus _compact.tex, _models.tex and _appendix.md.
#
# Environment:
#   SETTINGS  scenarios (default: all nine)
#   RUNS      runs directory (default runs/table1v2)
#   TAG       suffix of the method JSONs (default _adam_refine, the published run)
#   ARGS      extra make_table1.py flags, appended after the Table 1 flags so they override
#             them (default --extra=--bandwidth-grad: the published run predates the detached
#             RBF bandwidth that is now the library default)
#   OUT       table stem (default results/table1$TAG)
#
#SBATCH --job-name=cdm-table1
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/table1-%A_%a.out
#SBATCH --error=logs/table1-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
[ -f experiments/synthetic/pipeline/make_table1.py ] || { echo "run sbatch from the repo root"; exit 1; }
mkdir -p logs
. ./cluster_env.sh             # Python environment (CDM_ENV)
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

ALL_SETTINGS=${SETTINGS:-"1x1 1x5 1x10 5x1 5x5 5x10 10x1 10x5 10x10"}
RUNS=${RUNS:-runs/table1v2}
TAG=${TAG-_adam_refine}
OUT=${OUT:-results/table1$TAG}
[ -n "${ARGS+set}" ] || ARGS="--extra=--bandwidth-grad"     # ARGS="" runs with the detached default
# Table 1: ten trajectories per run, five runs, n_cond 250, Adam push 0.05, 100 refinement steps.
T1=(--budget 10 --attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05
    --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 --seed 42 --device cuda)
eval "EXTRA=($ARGS)"          # eval, so that a quoted --extra '...' in ARGS stays one argument

read -r -a ARR <<< "$ALL_SETTINGS"
S=${ARR[$SLURM_ARRAY_TASK_ID]}
echo "=== task $SLURM_ARRAY_TASK_ID: scenario $S tag '$TAG' on $(hostname) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python experiments/synthetic/pipeline/make_table1.py --settings "$S" "${T1[@]}" \
    --runs-dir "$RUNS" --tag "$TAG" --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"} --no-collect

# the last task collects (cheap; a re-run of --collect is always safe)
if [ "$SLURM_ARRAY_TASK_ID" = "$((${#ARR[@]} - 1))" ]; then
    echo "=== collect ==="
    # shellcheck disable=SC2086
    python experiments/synthetic/pipeline/make_table1.py --settings $ALL_SETTINGS "${T1[@]}" \
        --runs-dir "$RUNS" --tag "$TAG" --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"} --collect || true
fi
echo "done"
