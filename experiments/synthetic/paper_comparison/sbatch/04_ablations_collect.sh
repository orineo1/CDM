#!/usr/bin/env bash
# Collect the three ablation arms into experiments/synthetic/ablations/outputs/table8_{norefine,annealed,oracle}.*
#SBATCH --job-name=cdm-ablations-collect
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
source experiments/synthetic/paper_comparison/sbatch/common.sh
OUT=$SYN/ablations/outputs; mkdir -p $OUT
C="--settings $ALL_SETTINGS --runs-dir $GRID_RUNS --collect"
python $SYN/pipeline/make_table1.py $C --methods MLGD MLGD-F PMLGD PMLGD-F --tag _norefine --out $OUT/table8_norefine
python $SYN/pipeline/make_table1.py $C --methods PMLGD PMLGD-F --tag _annealed --out $OUT/table8_annealed
python $SYN/pipeline/make_table1.py $C --methods MLGD-X PMLGD-X --tag _oracle --out $OUT/table8_oracle
