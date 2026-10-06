#!/bin/bash
# Parity runs on GB-FESO's own benchmark: MLGD on GB-FESO's surrogate (models/gbfeso.pt, the same sampler GB-FESO
# optimizes on), MLGD* settings (FiniteDiff = zeroth-order coordinate-direction guidance, keep-best refinement,
# re-estimated top-5 selection), one array task per target. Variants by environment:
#   V      variant name (file tag), e.g. r10_kl_box
#   R      restarts (10)          SELN  samples per re-scoring (2000)
#   LOSS   "mmd" or "kl" (GB-FESO's KDE-KL, bandwidth 0.5)
#   BOX    1: --x-box (projection onto the eps box [0.21, 1.55]^K, in standardized units +-sqrt(3), as GB-FESO clamps)
#   REFG   pathwise | zo  (refinement gradient)
#   N=5 V=r10_kl_box R=10 LOSS=kl BOX=1 sbatch --array=0-39 --exclude=salmon-02,salmon-09 experiments/peptide/slurm/slurm_parity.sh
# Output: runs/peptide_parity/N<N>/results/<V>/MLGD_<V>_theirs_t<id>.json (diagnostics: per-step traces, refinement
# traces, wall time, in the JSON). Scoring: slurm_eval.sh with RES/EV pointing there (see results/parity.md).
#SBATCH --job-name=pep-parity
#SBATCH --partition=salmon,catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/peptide/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ./cluster_env.sh             # Python environment (CDM_ENV)
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
N=${N:-5}; [ "$N" = 4 ] && R0=runs/peptide || R0=runs/peptide_N$N; D=${D:-$R0}
V=${V:?variant name}; OUT=${OUT:-runs/peptide_parity/N$N/results/$V}; mkdir -p "$OUT"
case "${LOSS:-mmd}" in mmd) LF="";; kl) LF="--loss kde-kl --kde-bandwidth 0.5";; *) echo "LOSS?"; exit 1;; esac
[ "${BOX:-0}" = 1 ] && BF="--x-box -1.7320508 1.7320508" || BF=""
t=${T:-${SLURM_ARRAY_TASK_ID:-0}}; tid=$(printf %03d $t)
F=$OUT/MLGD_${V}_theirs_t$tid.json
[ -f "$F" ] && { echo "exists $F"; exit 0; }
echo "parity N=$N $V target $tid on $(hostname): $(python -c 'import torch; print(torch.cuda.get_device_name())')"
python experiments/peptide/pipeline/search_peptide.py --prior $D/models/prior/model_unconditional.pt \
  --sampler $D/models/gbfeso.pt --target $D/targets/target_$tid.npy \
  --algo lgd --n-attempts 1 --n-cond 250 --guidance-optim adam --step-size 0.05 --guide-grad zo \
  --n-restarts ${R:-10} --refine-steps 100 --refine-lr 0.02 --refine-n-cond 3000 --refine-keep-best \
  --refine-grad ${REFG:-pathwise} --select-top 5 --select-n ${SELN:-2000} --select-repeats 5 \
  $LF $BF --seed $((42 + t)) --device cuda --out "$F" ${EXTRA:-}
echo "done $(date)"
