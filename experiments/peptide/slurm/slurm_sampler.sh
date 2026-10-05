#!/bin/bash
# Accuracy of the trained surrogates at the 40 held-out eps* (ours: diffusion, consistency model; GB-FESO's surrogate)
# against the simulated target ensembles.   N=5 sbatch experiments/peptide/slurm/slurm_sampler.sh -> results/surrogate_accuracy_N5.md
#SBATCH --job-name=pep-sampler
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/peptide/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv$(grep -q "^13" /etc/debian_version 2>/dev/null && echo _deb13)}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
N=${N:-4}; [ "$N" = 4 ] && R0=runs/peptide || R0=runs/peptide_N$N; R=${R:-$R0}; P=experiments/peptide/pipeline
python $P/evaluate.py sampler --prior $R/models/prior/model_unconditional.pt \
  --sampler $R/models/diff/model_conditional.pt $R/models/cm/model_conditional.pt --gbfeso-ckpt $R/models/gbfeso.pt \
  --targets-ens $R/targets_ens.pt --meta $R/meta.json --target-ids ${IDS:-0-39} --n 4096 \
  --out experiments/peptide/results/surrogate_accuracy_N$N.md
echo "done $(date)"
