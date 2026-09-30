#!/usr/bin/env bash
# Table 4: check_models.py on the grid's trained networks (experiments/synthetic/model_quality/run.sh full).
# (01_grid already writes a check.json per scenario with the same defaults; this is the documented Table 4 path.)
#SBATCH --job-name=cdm-model-quality
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
source experiments/synthetic/paper_comparison/sbatch/common.sh
RUNS_DIR=$GRID_RUNS bash $SYN/model_quality/run.sh full
