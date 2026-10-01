#!/usr/bin/env bash
# Reruns the synthetic experiments whose outputs are in this folder, with the exact flags used.
# Run from the repository root, on a GPU node:
#
#   bash experiments/synthetic/results/run.sh all          # everything below, in order
#   bash experiments/synthetic/results/run.sh grid         # one step: grid | checks | ablations | nonstar | fig1 | smc
#   SETTINGS="5x5" bash experiments/synthetic/results/run.sh grid    # one scenario
#
# Detached RBF bandwidth throughout (the library default; no --bandwidth-grad). Outputs go to
# runs/reproduction/ (networks, per-method JSONs) and results/reproduction/ (tables, figures),
# both git-ignored. Every step whose output exists is skipped, so a killed run can be resumed.
# On SLURM: sbatch --gres=gpu:1 --time=3-00:00:00 --wrap="bash experiments/synthetic/results/run.sh all"
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
[ -f experiments/synthetic/pipeline/make_table1.py ] || { echo "run from the repository root" >&2; exit 1; }

P=experiments/synthetic/pipeline
RUNS=runs/reproduction
OUT=results/reproduction
DEV=${DEVICE:-cuda}
SETTINGS=${SETTINGS:-"1x1 1x5 1x10 5x1 5x5 5x10 10x1 10x5 10x10"}
mkdir -p "$RUNS" "$OUT"

# Table 1 search flags (paper App. B.4-B.6); data and training are the make_table1.py defaults:
# 4 components, 20,000 samples, MLP 3 x 128, 10,000 fresh batches of 1,024, resampling exp(-5 loss).
T1=(--budget 10 --attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05
    --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 --seed 42 --device "$DEV"
    --runs-dir "$RUNS/grid")

grid() {        # Table 1 and the refined columns of tab:sim_full; also writes check.json per scenario
  python $P/make_table1.py --settings $SETTINGS "${T1[@]}" --out "$OUT/table1"
}

checks() {      # tab:model_quality: the networks of each scenario against the analytic mixture
  mkdir -p "$OUT/checks"
  for s in $SETTINGS; do
    d="$RUNS/grid/x${s%x*}_y${s#*x}"
    python $P/check_models.py --prior "$d/models/uncond/model_unconditional.pt" \
      --sampler "$d/models/diffusion/model_conditional.pt" "$d/models/cm/model_conditional.pt" \
      --data "$d/data.pt" --n-random-x 100 --device "$DEV" --out "$OUT/checks/table1_x${s%x*}_y${s#*x}"
  done
}

ablations() {   # tab:sim_full ablation columns, TV without refinement, on the grid's networks
  python $P/make_table1.py --settings $SETTINGS "${T1[@]}" --refine-steps 0 \
    --tag _plain --out "$OUT/table1_plain"
  python $P/make_table1.py --settings $SETTINGS "${T1[@]}" --refine-steps 0 \
    --methods PMLGD PMLGD-F --resample-schedule linear --tag _anneal --out "$OUT/table1_anneal"
  python $P/make_table1.py --settings $SETTINGS "${T1[@]}" --refine-steps 0 \
    --methods MLGD-X PMLGD-X --tag _oracle --out "$OUT/table1_oracle"
}

nonstar() {     # tab:nonstar_results: point-mass target, MLGD-F, 25 restarts per arm
  local d="$RUNS/nonstar"; mkdir -p "$d"
  [ -f "$d/data.pt" ] || python $P/generate_data.py --params-file experiments/synthetic/configs/mog/nonstar_2d.txt \
    --condition-on 1 --n-samples 5000 --seed 42 --output "$d/data.pt"
  for m in uncond cm; do
    local f="$d/models/$m/model_$([ $m = uncond ] && echo unconditional || echo conditional).pt"
    [ -f "$f" ] || python $P/train_models.py --data "$d/data.pt" \
      --model $([ $m = uncond ] && echo unconditional || echo conditional) \
      --architecture $([ $m = uncond ] && echo diffusion || echo cm) --fresh-samples \
      --epochs 20000 --batch-size 1024 --nunits 128 --nblocks 3 --diffusion-steps 100 --seed 42 \
      --device "$DEV" --output-dir "$d/models/$m"
  done
  mkdir -p "$OUT/nonstar"
  for arm in "MMD_n250 mmd 250" "L2_n250 l2 250" "L2_n1 l2 1"; do
    set -- $arm
    python $P/find_xstar.py --prior "$d/models/uncond/model_unconditional.pt" \
      --sampler "$d/models/cm/model_conditional.pt" --target "$d/data.pt" \
      --algo lgd --n-restarts 1 --n-attempts 25 --n-mc 1 --init zeros \
      --guidance-optim adam --step-size 0.05 --refine-steps 0 --loss "$2" --n-cond "$3" \
      --seed 42 --device "$DEV" --out "$OUT/nonstar/arm_$1.json"
  done
  # "% at x_bi" = fraction of the 25 x_hat closer to -2 than to +2 (no script computes it)
}

fig1() {        # Figure 1, setting 'hard': toy networks, the four methods (25 trajectories each), beta sweep
  python $P/fig1_toy_example.py --setting hard --n-restarts 25 --n-cond 250 --n-mc 3 \
    --methods MLGD MLGD-F PMLGD PMLGD-F --betas 0 2 8 16 --n-beta-samples 2000 \
    --rerun-sim --device "$DEV" --ckpt-dir "$RUNS/toy" --out "$OUT/fig1/fig1_hard"
}

smc() {         # twisted-SMC correction of the Figure 1(d) sweep, on the networks fig1 trained
  python $P/fig_smc_correction.py --setting hard --ckpt-dir "$RUNS/toy" \
    --betas 0 2 8 16 --n-particles 2000 --n-repeats 10 --max-shift 0.25 --ess-frac 1.0 --bins 80 \
    --device "$DEV" --out "$OUT/smc/fig_smc_hard"
  # Figure 1 with the corrected populations overlaid on panel (d):
  python $P/fig1_toy_example.py --setting hard --ckpt-dir "$RUNS/toy" --out "$OUT/fig1/fig1_hard" \
    --smc-npz "$OUT/smc/fig_smc_hard.npz"
}

case "${1:-}" in
  all)                     grid; checks; ablations; nonstar; fig1; smc ;;
  grid|checks|ablations|nonstar|fig1|smc) "$1" ;;
  *) echo "usage: bash $0 {all|grid|checks|ablations|nonstar|fig1|smc}" >&2; exit 64 ;;
esac
