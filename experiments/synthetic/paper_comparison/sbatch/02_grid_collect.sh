#!/usr/bin/env bash
# Collect Table 1 into experiments/synthetic/main_grid/outputs/table1.{json,md,tex}. No GPU needed.
#SBATCH --job-name=cdm-grid-collect
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
source experiments/synthetic/paper_comparison/sbatch/common.sh
SETTINGS="$ALL_SETTINGS" bash $SYN/main_grid/run.sh full --collect
