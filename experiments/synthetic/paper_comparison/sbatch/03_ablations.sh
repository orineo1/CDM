#!/usr/bin/env bash
# Table 5 ablation columns, one array task per scenario, on the SAME trained networks as Table 1
# (runs-dir = the grid's, so nothing is retrained; each arm writes <METHOD><tag>.json next to Table 1's).
# The flags are those of experiments/synthetic/ablations/run.sh full, restricted to one scenario:
#   plain    : all four methods, no refinement             (--tag _norefine)
#   annealed : PMLGD, PMLGD-F, linear resampling schedule   (--tag _annealed; TV is read before refinement)
#   oracle   : exact conditional as sampler, no refinement  (--tag _oracle)
# Submit with --dependency=aftercorr:<01_grid job id> so task i starts when grid task i is done.
#SBATCH --job-name=cdm-ablations
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
COMMON="--settings $S --n-components 4 --n-samples 20000 --epochs 10000 --batch-size 1024 --nunits 128 --nblocks 3
        --budget 10 --attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05
        --seed 42 --device cuda --runs-dir $GRID_RUNS --no-collect"
python $SYN/pipeline/make_table1.py $COMMON --methods MLGD MLGD-F PMLGD PMLGD-F \
    --refine-steps 0 --tag _norefine
python $SYN/pipeline/make_table1.py $COMMON --methods PMLGD PMLGD-F \
    --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 \
    --resample-beta 5.0 --resample-schedule linear --tag _annealed
python $SYN/pipeline/make_table1.py $COMMON --methods MLGD-X PMLGD-X \
    --refine-steps 0 --tag _oracle
