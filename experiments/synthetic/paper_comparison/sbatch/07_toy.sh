#!/usr/bin/env bash
# Figure 1 (toy, setting 'hard' = the published figure): trains the three toy networks, runs the four methods
# and the beta sweep. Flags of experiments/synthetic/toy_example/run.sh full, except --n-restarts 10:
# the paper reports ten trajectories per method (run.sh uses 25).
#SBATCH --job-name=cdm-toy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
source experiments/synthetic/paper_comparison/sbatch/common.sh
OUT=$SYN/toy_example/outputs; mkdir -p $OUT
python $SYN/pipeline/fig1_toy_example.py \
    --setting hard --n-restarts ${N_RESTARTS:-10} --n-cond 250 --n-mc 3 \
    --methods MLGD MLGD-F PMLGD PMLGD-F \
    --betas 0 2 8 16 --n-beta-samples 2000 \
    --rerun-sim --rerun-sweep --device cuda \
    --ckpt-dir $OUT/ckpt --out $OUT/fig1_hard
