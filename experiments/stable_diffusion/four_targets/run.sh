#!/usr/bin/env bash
# MLGD-F on four distributional targets (paper Table 3 / Table 15).
#
#   bash run.sh check                              CPU, seconds, no downloads, no GPU.
#                                                  Static integrity of the ported package +
#                                                  proof that the SD MMD is the shared compute_mmd.
#   bash run.sh smoke <scenario> [flags...]        GPU. 6 guided steps, 6 variations. Proves the
#                                                  pipeline runs end to end; reproduces NO
#                                                  published number.
#   bash run.sh full  <scenario> [flags...]        GPU (paper: L40S 48 GB). The published
#                                                  configuration, from config/<scenario>.env.
#   bash run.sh help                               every flag run_mlgd_f accepts.
#
# <scenario>: balanced | skewed | gender_interpolation | age_interpolation | all
#
# Defaults come from config/<scenario>.env (tracked in git). Any run_mlgd_f flag given
# after the scenario overrides them -- it is passed last and the last value wins:
#   bash run.sh full balanced --base_zeta 5 --seed 2
#   bash run.sh full skewed   --num_variations 50 --backsel_k 25 --run_unguided
#
# On SLURM, submit this file with --wrap from the repo root (sbatch passes the active
# venv and exported variables such as HF_HOME to the job):
#   mkdir -p logs
#   sbatch --partition=salmon --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
#          --output=logs/sd-%j.out --wrap "bash experiments/stable_diffusion/four_targets/run.sh full balanced"
#   sbatch ... --array=0-3 --output=logs/sd-%A_%a.out --wrap "bash experiments/stable_diffusion/four_targets/run.sh full all"
# Under an array, `all` runs only scenario SLURM_ARRAY_TASK_ID (0 balanced, 1 skewed,
# 2 gender_interpolation, 3 age_interpolation); otherwise the four run one after another.
# Nodes without internet: WANDB_MODE=offline, then `wandb sync experiments/stable_diffusion/four_targets/outputs/wandb/offline-run-*`.
#
# Logging: every run goes to Weights & Biases (per-step MMD, gradient norm, zeta,
# correction norm, the per-step visualisation, input and final scribbles, final MMD;
# console output under the run's Logs tab). Everything except the per-step visualisation
# is also saved to outputs/<tier>_<Scenario>/: steps.csv (written step by step),
# metrics.json, run_info.txt (commit, GPU, exact command) and the images.
# python experiments/stable_diffusion/export_results.py then copies finished full runs
# into the committed results/, as for MNIST.
#
# Environment (optional):
#   PYTHON=/path/to/python   interpreter (default: python on PATH)
#   WANDB_PROJECT=<name>     default sdxl-smoke for smoke, sdxl-mlgdf for full
#   WANDB_ENTITY=<team>      default: the logged-in user
#   WANDB_NAME=<run name>    default <tier>_<scenario>[_<slurm job id>]
#   CACHE_DIR=/path          where model weights and all caches go (default <repo>/.cache;
#                            HF_HOME, MPLCONFIGDIR, WANDB_DIR, ... override one by one)
#   DRY_RUN=1                print the command instead of running it
set -euo pipefail
export PYTHONUNBUFFERED=1   # print lines reach the log as they happen, also under sbatch
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-python}"
OUT="$HERE/outputs"
SCENARIOS=(balanced skewed gender_interpolation age_interpolation)
TIER="${1:-check}"
cd "$REPO"

# Keep every cache next to the repo (e.g. lab storage), not in $HOME: model weights,
# matplotlib, wandb, torch, pip. Anything already set in the environment is kept.
CACHE_DIR="${CACHE_DIR:-$REPO/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_DIR/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_DIR/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_DIR/pip}"
export WANDB_DIR="${WANDB_DIR:-$OUT}"                  # wandb/ run files (and offline runs)
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$CACHE_DIR/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$CACHE_DIR/wandb-data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$CACHE_DIR/wandb-artifacts}"
mkdir -p "$CACHE_DIR" "$MPLCONFIGDIR" "$OUT"

usage() { echo "usage: bash run.sh [check | help | smoke|full <balanced|skewed|gender_interpolation|age_interpolation|all> [run_mlgd_f flags...]]" >&2; exit 64; }

run_one() {   # run_one <tier> <scenario> [extra flags...]
  local tier="$1" name="$2"; shift 2
  local cfg="$HERE/config/${name}.env"
  [ -f "$cfg" ] || { echo "no such scenario: $name (config/$name.env)" >&2; usage; }
  # the config sets SCENARIO, MODE, N_STEPS, ...; a subshell keeps scenarios from leaking into each other
  (
    set -a; source "$cfg"; set +a

    local outdir="$OUT/${tier}_${SCENARIO}" project
    local a
    for a in "$@"; do [ "${prev:-}" = --output_dir ] && outdir="$a"; prev="$a"; done

    local args=( --mode "$MODE" --base_zeta "$BASE_ZETA" --guidance_scale "$GUIDANCE_SCALE"
                 --controlnet_scale "$CONTROLNET_SCALE" --backsel_rule "$BACKSEL_RULE"
                 --loss_fn "$LOSS_FN" --seed "$SEED" )
    if [ "$tier" = smoke ]; then
      project="${WANDB_PROJECT:-sdxl-smoke}"
      args+=( --n_steps 12 --start_step 6 --num_variations 6 --backsel_k 3 )
      [ "$MODE" = age ] && args+=( --age_min "$AGE_MIN" --age_max "$AGE_MAX" --age_step 10
                                   --n_per_age 1 --age_gender "$AGE_GENDER" )
    else
      project="${WANDB_PROJECT:-sdxl-mlgdf}"
      args+=( --n_steps "$N_STEPS" --start_step "$START_STEP"
              --num_variations "$NUM_VARIATIONS" --backsel_k "$BACKSEL_K" )
      [ "$MODE" = age ] && args+=( --age_min "$AGE_MIN" --age_max "$AGE_MAX" --age_step "$AGE_STEP"
                                   --n_per_age "$N_PER_AGE" --age_gender "$AGE_GENDER" )
    fi
    if [ "$MODE" != age ]; then
      local tp; IFS='|' read -r -a tp <<< "$TARGET_PROMPTS"
      args+=( --target_prompts "${tp[@]}" )
    fi
    args+=( --output_dir "$outdir" --wandb_project "$project" )
    [ -n "${WANDB_ENTITY:-}" ] && args+=( --wandb_entity "$WANDB_ENTITY" )
    args+=( "$@" )   # CLI overrides, last so they win

    export WANDB_NAME="${WANDB_NAME:-${tier}_${name}${SLURM_JOB_ID:+_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}}"
    export WANDB_TAGS="${WANDB_TAGS:-$tier,$name}"
    local cmd=( "$PY" -m cdm.models.sd.run_mlgd_f "${args[@]}" )

    echo "=== $tier | $name | wandb $project/$WANDB_NAME ==="
    printf '%q ' "${cmd[@]}"; echo
    [ -n "${DRY_RUN:-}" ] && exit 0

    mkdir -p "$outdir"
    {   # what produced this run
      echo "date:       $(date -Iseconds)"
      echo "git commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$REPO" diff --quiet 2>/dev/null || echo ' (dirty)')"
      echo "host:       $(hostname)  slurm job: ${SLURM_JOB_ID:-none}"
      "$PY" -c "import torch; print('gpu:       ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none', '| torch', torch.__version__)"
      echo "config:     config/$name.env  overrides: ${*:-none}"
      printf 'command:    '; printf '%q ' "${cmd[@]}"; echo
    } > "$outdir/run_info.txt"
    "${cmd[@]}"
  )
}

case "$TIER" in
  check)
    "$PY" -m cdm.models.sd.check_imports
    "$PY" -m cdm.models.sd.check_mmd_equivalence
    ;;

  help)
    "$PY" -m cdm.models.sd.run_mlgd_f --help
    ;;

  smoke|full)
    SCEN="${2:-}"; [ -n "$SCEN" ] || usage
    shift 2
    if [ -z "${DRY_RUN:-}" ]; then   # fail now, not after loading the models
      "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA GPU visible (sbatch: add --gres=gpu:1)')"
      if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != offline ] && ! grep -q api.wandb.ai ~/.netrc 2>/dev/null; then
        echo "warning: no wandb login found; run 'wandb login' once, or export WANDB_API_KEY / WANDB_MODE=offline" >&2
      fi
    fi
    if [ "$SCEN" = all ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
      SCEN="${SCENARIOS[$SLURM_ARRAY_TASK_ID]:?array index must be 0-3}"
    fi
    if [ "$SCEN" = all ]; then
      for s in "${SCENARIOS[@]}"; do run_one "$TIER" "$s" "$@"; done
    else
      run_one "$TIER" "$SCEN" "$@"
    fi
    if [ -z "${DRY_RUN:-}" ]; then
      if [ "$TIER" = full ]; then echo "save into results/: python experiments/stable_diffusion/export_results.py"
      else echo "export (not into results/): python experiments/stable_diffusion/export_results.py --tier $TIER --dest <dir>"; fi
    fi
    ;;

  *) usage ;;
esac
