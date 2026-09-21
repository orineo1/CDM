#!/usr/bin/env bash
# Figure 1 (toy example) on the cluster: all settings, with restarts.
#
#   sbatch --partition=<gpu partition> scripts/slurm_fig1.sh      # everything below
#   sbatch scripts/slurm_fig1.sh hard            # one setting
#   N_RESTARTS=50 sbatch scripts/slurm_fig1.sh   # more restarts
#
# Each setting trains its three small networks once (cached under
# checkpoints/toy/<setting>/), runs N_RESTARTS restarts of MLGD and MLGD-F,
# the batched beta-sweep, and writes results/fig1_<setting>.png plus the
# simulation cache (_sim.json, _sweep.npz) next to it. Copy results/ back
# to redraw locally without recomputing (scp through the jump host fails;
# stream the files instead):
#   for f in fig1_hard.png fig1_hard_sim.json fig1_hard_sweep.npz; do
#     ssh huji-work "cat /sci/labs/orzuk/orzuk/github/CDM/results/$f" > results/$f; done
#
#SBATCH --job-name=cdm-fig1
#SBATCH --partition=gpu            # override: sbatch --partition=<name> scripts/slurm_fig1.sh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig1-%j.out
#SBATCH --error=logs/fig1-%j.err
set -euo pipefail

# sbatch runs a copy of this script from the spool dir, so locate the repo via the submit dir
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -f scripts/fig1_toy_example.py ] || { echo "run sbatch from the repo root"; exit 1; }
mkdir -p logs results
# environment: a venv with CUDA torch (no modules needed when the torch wheel bundles CUDA)
VENV=${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi

N_RESTARTS=${N_RESTARTS:-25}
SETTINGS=${1:-"hard bimodal paper"}

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
for s in $SETTINGS; do
    echo "=== setting $s, $N_RESTARTS restarts ==="
    python scripts/fig1_toy_example.py --setting "$s" --n-restarts "$N_RESTARTS" --rerun-sim --device cuda
done
echo "done: $(ls results/fig1_*.png)"
