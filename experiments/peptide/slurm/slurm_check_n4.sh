#!/bin/bash
# N = 4 regression check of the N-bead code against the stored N = 4 run (runs/peptide):
#  1. the 40 targets re-simulated with the generalized simulator (same seeds) vs runs/peptide/targets_ens.pt
#  2. GB-FESO (4 corner starts) and MLGD with 2 trajectories on targets 0-1 vs the stored result files
#   sbatch experiments/peptide/slurm/slurm_check_n4.sh       -> experiments/peptide/results/check_N4.md
#SBATCH --job-name=pep-check4
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/peptide/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
. ${VENV:-/sci/labs/orzuk/orzuk/venvs/CDM_venv}/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
R=runs/peptide; C=$R/check_generalized; P=experiments/peptide/pipeline; mkdir -p $C/results
python -c "import torch; print(torch.cuda.get_device_name())"
python $P/sim_ensembles.py --n-eps 40 --seed 2 --n-rep 64 --chunk 200 --keep 20000 --out $C/targets_ens.pt
python $P/gbfeso.py opt --ckpt $R/models/gbfeso.pt --targets-ens $R/targets_ens.pt --target-ids 0-1 \
  --plot-dir /tmp/gbfeso_$SLURM_JOB_ID --out $C/results/gbfeso_0.json
for t in 0 1; do tid=$(printf %03d $t)
  python $P/search_peptide.py --prior $R/models/prior/model_unconditional.pt --sampler $R/models/diff/model_conditional.pt \
    --target $R/targets/target_$tid.npy --algo lgd --n-attempts 1 --n-cond 250 --guidance-optim adam --step-size 0.05 --guide-grad zo \
    --n-restarts 2 --refine-steps 100 --refine-lr 0.02 --refine-n-cond 3000 --refine-keep-best \
    --select-top 5 --select-n 2000 --select-repeats 5 --seed $((42 + t)) --device cuda --out $C/results/MLGD_r2_t$tid.json
done
python - <<'PY' | tee experiments/peptide/results/check_N4.md
import json, torch, numpy as np
R, C = "runs/peptide", "runs/peptide/check_generalized"
a, b = torch.load(f"{R}/targets_ens.pt", weights_only=False), torch.load(f"{C}/targets_ens.pt", weights_only=False)
d = (a["Y"] - b["Y"]).abs()
print("# N = 4 check of the generalized code\n")
print(f"targets re-simulated: eps identical {torch.equal(a['eps'], b['eps'])}, Y bit-identical {torch.equal(a['Y'], b['Y'])}, "
      f"max |dY| {float(d.max()):.3g}, fraction of differing entries {float((d > 0).float().mean()):.3g}, kinetic T {a['temperature']:.5f} vs {b['temperature']:.5f}\n")
old = {(r["target_id"], r["start"]): r for r in json.load(open(f"{R}/results/gbfeso_0.json"))["runs"]}
print("| run | stored eps_hat | regenerated eps_hat | stored KL | regenerated KL |\n|---|---|---|---|---|")
for r in json.load(open(f"{C}/results/gbfeso_0.json"))["runs"]:
    o = old[(r["target_id"], r["start"])]
    print(f"| GB-FESO t{r['target_id']} {r['start']} | {np.round(o['eps_hat'], 4).tolist()} | {np.round(r['eps_hat'], 4).tolist()} | {o['surrogate_loss']:.4f} | {r['surrogate_loss']:.4f} |")
for t in (0, 1):
    o, n = json.load(open(f"{R}/results/MLGD_r2_t{t:03d}.json"))["best"], json.load(open(f"{C}/results/MLGD_r2_t{t:03d}.json"))["best"]
    print(f"| MLGD_r2 t{t} | {np.round(o['x_hat'], 4).tolist()} (std. x) | {np.round(n['x_hat'], 4).tolist()} | {o.get('select_loss', o['final_loss']):.5f} | {n.get('select_loss', n['final_loss']):.5f} |")
PY
echo "done $(date)"
