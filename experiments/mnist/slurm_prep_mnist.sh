#!/bin/bash
# One-time preparation of the MNIST experiments on the cluster (run before the MNIST arrays):
# the two pretrained networks from Hugging Face into experiments/mnist/checkpoints/ (cache on the lab disk, not
# $HOME), MNIST into experiments/mnist/train/outputs/data, and the rotation-robust classifier trained once into
# experiments/mnist/checkpoints/robust_classifier.pth (otherwise every array task would train and write it).
#   sbatch experiments/mnist/slurm_prep_mnist.sh      (from the repository root)
#SBATCH --job-name=mnist-prep
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
. ./cluster_env.sh
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-${CDM_LAB:-.}/hf_cache}
M=experiments/mnist
python - <<PY
import os, shutil, torch
from cdm.models.mnist.checkpoints import resolve_checkpoint, COND_FILENAME, UNCOND_FILENAME
from cdm.models.mnist.classifier import load_or_train_classifier
ck = "$M/checkpoints"
for fn in (COND_FILENAME, UNCOND_FILENAME):
    dst = os.path.join(ck, fn)
    if not os.path.exists(dst):
        shutil.copy(resolve_checkpoint(fn, None), dst)
    print("checkpoint", dst, os.path.getsize(dst))
load_or_train_classifier(save_path=os.path.join(ck, "robust_classifier.pth"),
                         data_root="$M/train/outputs/data", device="cuda" if torch.cuda.is_available() else "cpu")
print("classifier ready")
PY
echo "done $(date)"
