#!/usr/bin/env bash
# Table 1 (+ Table 4 check.json, + Table 5 distances): one array task per (dim X, dim Y) scenario.
# Data, the three networks, check_models.py and the four searches (MLGD, MLGD-F, PMLGD, PMLGD-F; 5 runs each)
# with the published flags of experiments/synthetic/main_grid/run.sh full.
#SBATCH --job-name=cdm-grid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x-%A_%a.out
source experiments/synthetic/paper_comparison/sbatch/common.sh
read -r -a ARR <<< "$ALL_SETTINGS"
S=${ARR[$SLURM_ARRAY_TASK_ID]}
echo "scenario $S"
SETTINGS="$S" bash $SYN/main_grid/run.sh full --no-collect
