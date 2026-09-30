#!/usr/bin/env bash
# Table 6 (no-star target): experiments/synthetic/infeasible_target/run.sh full (2 trainings + 3 arms x 25 restarts).
#SBATCH --job-name=cdm-infeasible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
source experiments/synthetic/paper_comparison/sbatch/common.sh
bash $SYN/infeasible_target/run.sh full
