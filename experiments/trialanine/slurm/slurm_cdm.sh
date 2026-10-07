#!/bin/bash
# Trialanine CDM benchmark (label-free count targets), from the repo root. STAGE selects the step:
#   STAGE=data   sbatch -p glacier --gres=none experiments/trialanine/slurm/slurm_cdm.sh   (CPU, reweighting)
#   STAGE=train  sbatch --array=0-1 ...      0: conditional diffusion P(y | x)   1: prior over x
#   STAGE=search sbatch --array=0-K ...      task k = target k: MLGD-KL, MLGD-MMD, GB-FESO (x0 start; 4 random
#                                            starts), MLGD-KL + GB-FESO polish, MLGD-MMD + GB-FESO polish
# Outputs: runs/trialanine/cdm/{data.pt, prior_data.pt, targets/, meta.json, models/, results/<method>_tXXX.json}
#SBATCH --job-name=tri-cdm
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=salmon-02,salmon-09
#SBATCH --output=logs/trialanine/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=${R:-runs/trialanine/cdm}; P=experiments/trialanine/pipeline; T=experiments/synthetic/pipeline/train_models.py
mkdir -p $R/results
k=${SLURM_ARRAY_TASK_ID:-0}
case "${STAGE:?set STAGE}" in
  data)
    python $P/make_data_tri.py --frames "runs/trialanine/prod_x0_w*.npz" --out $R ${EXTRA:-} ;;
  train)
    python -c "import torch; print(torch.cuda.get_device_name())"
    if [ $k = 0 ]; then
      python $T --data $R/data.pt --model conditional --architecture diffusion --nunits 256 --nblocks 4 \
        --epochs ${EPOCHS:-40} --batch-size 512 --seed 0 --output-dir $R/models/diff
    else
      python $T --data $R/prior_data.pt --model unconditional --nunits 128 --nblocks 3 \
        --epochs 50 --batch-size 512 --seed 0 --output-dir $R/models/prior
    fi ;;
  search)
    python -c "import torch; print(torch.cuda.get_device_name())"
    tid=$(printf %03d $k); TG=$R/targets/target_$tid.npy; [ -f $TG ] || { echo "no $TG"; exit 0; }
    S=$R/models/diff/model_conditional.pt; PR=$R/models/prior/model_unconditional.pt; O=$R/results
    MLGD="--algo lgd --n-attempts 1 --n-cond 250 --guidance-optim adam --step-size 0.05 --guide-grad pathwise
          --n-restarts 10 --refine-steps 100 --refine-lr 0.02 --refine-n-cond 3000 --refine-keep-best
          --select-top 5 --select-n 2000 --select-repeats 5 --x-box -2 2 --seed $((42 + k)) --device cuda"
    for L in kde-kl mmd; do
      N=MLGD-$([ $L = kde-kl ] && echo KL || echo MMD)
      [ -f $O/${N}_t$tid.json ] || python experiments/synthetic/pipeline/search_mog.py --prior $PR --sampler $S \
        --target $TG --loss $L --kde-bandwidth 0.5 $MLGD --out $O/${N}_t$tid.json
    done
    G="python $P/search_gbfeso_tri.py --sampler $S --target $TG --loss kl --seed $((42 + k)) --device cuda"
    [ -f $O/GBFESO-x0_t$tid.json ] || $G --init zero --method-name GBFESO-x0 --out $O/GBFESO-x0_t$tid.json
    [ -f $O/GBFESO-rand4_t$tid.json ] || $G --init randn --n-starts 4 --method-name GBFESO-rand4 --out $O/GBFESO-rand4_t$tid.json
    for N in MLGD-KL MLGD-MMD; do
      [ -f $O/${N}+polish_t$tid.json ] || $G --init json --init-json $O/${N}_t$tid.json --init-top 3 \
        --method-name ${N}+polish --out $O/${N}+polish_t$tid.json
    done ;;
esac
echo "done $(date)"
