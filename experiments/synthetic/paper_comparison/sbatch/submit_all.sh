#!/usr/bin/env bash
# Submit every synthetic experiment of the paper, chained with dependencies, then the comparison.
# Run from the repository root:
#
#   PARTITION=gpu VENV=~/venvs/cdm bash experiments/synthetic/paper_comparison/sbatch/submit_all.sh
#
# Optional: ACCOUNT=<slurm account>, CPU_PARTITION=<partition for the collect/compare jobs> (default PARTITION),
#           CONDA_ENV=<name> instead of VENV, N_RESTARTS=<toy trajectories, default 10>.
# Everything a job needs is exported to it (sbatch --export=ALL, the default).
set -euo pipefail
[ -f experiments/synthetic/pipeline/make_table1.py ] || { echo "run from the repository root" >&2; exit 1; }
D=experiments/synthetic/paper_comparison/sbatch
mkdir -p logs
OPT=()
[ -n "${PARTITION:-}" ] && OPT+=(--partition="$PARTITION")
[ -n "${ACCOUNT:-}" ] && OPT+=(--account="$ACCOUNT")
CPU_OPT=("${OPT[@]}")
[ -n "${CPU_PARTITION:-}" ] && CPU_OPT=(--partition="$CPU_PARTITION" ${ACCOUNT:+--account="$ACCOUNT"})
sub() { sbatch --parsable "$@" | cut -d';' -f1; }

GRID=$(sub "${OPT[@]}" $D/01_grid.sh)
GCOL=$(sub "${CPU_OPT[@]}" --dependency=afterany:$GRID $D/02_grid_collect.sh)
ABL=$(sub "${OPT[@]}" --dependency=aftercorr:$GRID $D/03_ablations.sh)
ACOL=$(sub "${CPU_OPT[@]}" --dependency=afterany:$ABL $D/04_ablations_collect.sh)
MQ=$(sub "${OPT[@]}" --dependency=afterany:$GRID $D/05_model_quality.sh)
INF=$(sub "${OPT[@]}" $D/06_infeasible.sh)
TOY=$(sub "${OPT[@]}" $D/07_toy.sh)
SMC=$(sub "${OPT[@]}" --dependency=afterok:$TOY $D/08_smc.sh)
CMP=$(sub "${CPU_OPT[@]}" --dependency=afterany:$GCOL:$ACOL:$MQ:$INF:$SMC $D/09_compare.sh)

cat <<MSG
submitted:
  01 grid (Table 1)          $GRID  (array 0-8)
  02 grid collect            $GCOL
  03 ablations (Table 5)     $ABL   (array 0-8, task i after grid task i)
  04 ablations collect       $ACOL
  05 model quality (Tab. 4)  $MQ
  06 infeasible (Table 6)    $INF
  07 toy (Figure 1)          $TOY
  08 SMC correction          $SMC
  09 compare to paper        $CMP
watch:   squeue -u \$USER
logs:    logs/cdm-*.out
result:  experiments/synthetic/paper_comparison/outputs/comparison.md
MSG
