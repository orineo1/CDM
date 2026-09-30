#!/usr/bin/env bash
# Paper vs this run: writes experiments/synthetic/paper_comparison/outputs/comparison.{md,csv}.
#SBATCH --job-name=cdm-compare
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
source experiments/synthetic/paper_comparison/sbatch/common.sh
python $SYN/paper_comparison/compare_to_paper.py
