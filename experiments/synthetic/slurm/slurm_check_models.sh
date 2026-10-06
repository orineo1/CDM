#!/bin/bash
# Re-run the network check (check_models.py) for the nine Table 1 scenarios, one array task per scenario:
#   1. the Table 1 networks (prior, conditional DDPM, conditional iCT)  -> runs/table1v2/<cell>/check.json
#   2. MLGD*'s retrained conditional sampler (diffusion, 256 units, 4 blocks) -> runs/table1v2/<cell>/check_d256.json
# The previous check.json is kept once as check_before_norm1d.json (it predates the normalised 1D error at x*).
#   sbatch --array=0-8 --job-name=check experiments/synthetic/slurm/slurm_check_models.sh
# SRC: repository whose runs/ hold the networks and data (default: this one). OUT: where the checks go
# (default: next to the networks). To read someone else's runs and write in your own directory:
#   SRC=/sci/labs/orzuk/orzuk/github/CDM OUT=$PWD/runs/checks sbatch --array=0-8 --job-name=check --export=ALL \
#     experiments/synthetic/slurm/slurm_check_models.sh
# then rebuild the tables (the _models.tex file gets the two d256 columns):
#   python experiments/synthetic/pipeline/make_table1.py --collect --runs-dir runs/table1v2 --tag _main --out <out>
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh             # Python environment (CDM_ENV)
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CELLS=(x1_y1 x1_y5 x1_y10 x5_y1 x5_y5 x5_y10 x10_y1 x10_y5 x10_y10)
CELL=${CELL:-${CELLS[${SLURM_ARRAY_TASK_ID:-0}]}}
SRC=${SRC:-.}
D=$SRC/runs/table1v2/$CELL
if [ "$CELL" = x10_y1 ]; then D256=$SRC/runs/night/x10_y1/models/d256; else D256=$SRC/runs/night/grid/$CELL/d256; fi
O=${OUT:+$OUT/$CELL}; O=${O:-$D}
mkdir -p "$O"
echo "check $CELL: networks from $D, d256 from $D256, output to $O ($(hostname) $(date))"

if [ -f "$O/check.json" ] && [ ! -f "$O/check_before_norm1d.json" ]; then cp "$O/check.json" "$O/check_before_norm1d.json"; fi
python experiments/synthetic/pipeline/check_models.py --prior $D/models/uncond/model_unconditional.pt \
  --sampler $D/models/diffusion/model_conditional.pt $D/models/cm/model_conditional.pt \
  --data $D/data.pt --out $O/check --device cuda

python experiments/synthetic/pipeline/check_models.py --sampler $D256/model_conditional.pt \
  --data $D/data.pt --out $O/check_d256 --device cuda
echo "done $(date)"
