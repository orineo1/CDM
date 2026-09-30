#!/usr/bin/env bash
# Twisted-SMC correction of the Figure 1(d) beta sweep (experiments/synthetic/smc_correction/run.sh full),
# on the toy networks trained by 07_toy.sh.
#SBATCH --job-name=cdm-smc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
source experiments/synthetic/paper_comparison/sbatch/common.sh
OUT=$SYN/smc_correction/outputs; mkdir -p $OUT
python $SYN/pipeline/fig_smc_correction.py \
    --setting hard --ckpt-dir $SYN/toy_example/outputs/ckpt \
    --betas 0 2 8 16 --n-particles 2000 --n-repeats 10 \
    --max-shift 0.25 --ess-frac 1.0 --bins 80 \
    --device cuda --out $OUT/fig_smc_hard
