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
#   bash run.sh eval  <scenario> [flags...]        GPU, ~1 h. The paper's final evaluation
#                                                  (Table 3 / 15): N = 2,000 images per method
#                                                  vs a 2,000-image target, MMD / SWD / p(male).
#                                                  Reads the scribbles of outputs/full_<Scenario>
#                                                  (--run_dir overrides) and writes
#                                                  outputs/eval_<Scenario>. No photos are saved.
#   bash run.sh help                               every flag run_mlgd_f accepts.
#
# <scenario>: balanced | skewed | gender_interpolation | age_interpolation | all
#
# Defaults come from config/<scenario>.env (tracked in git). Any run_mlgd_f flag given
# after the scenario overrides them -- it is passed last and the last value wins:
#   bash run.sh full balanced --base_zeta 5 --seed 2
#   bash run.sh full skewed   --num_variations 50 --backsel_k 25 --run_unguided
#   bash run.sh eval balanced --n_eval 200 --run_dir experiments/stable_diffusion/four_targets/outputs/smoke_BalancedTarget
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
# no venv active (e.g. sbatch from a plain login shell): try VENV, then the repo's .venv
if ! command -v "$PY" >/dev/null 2>&1; then
  for cand in ${VENV:+"$VENV/bin/python"} "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.venv/bin/python"; do
    [ -x "$cand" ] && { PY="$cand"; break; }
  done
  command -v "$PY" >/dev/null 2>&1 || { echo "run.sh: no python on PATH -- activate the venv before sbatch" \
    "(source /path/to/venv/bin/activate), or set VENV=/path/to/venv or PYTHON=/path/to/python" >&2; exit 69; }
fi
OUT="$HERE/outputs"
SCENARIOS=(balanced skewed gender_interpolation age_interpolation)
TIER="${1:-check}"
cd "$REPO"

# Keep every cache next to the repo (e.g. lab storage), not in $HOME: model weights,
# matplotlib, wandb, torch, pip, CUDA/Triton kernels. Anything already set in the
# environment is kept. The wandb key: `wandb login` (writes ~/.netrc) or WANDB_API_KEY.
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
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$CACHE_DIR/wandb-config}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_DIR/nv}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_DIR/triton}"
mkdir -p "$CACHE_DIR" "$MPLCONFIGDIR" "$OUT"
# deterministic cuBLAS (run_mlgd_f.make_deterministic also sets it; here it is set
# before the interpreter starts, which is what cuBLAS requires)
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

usage() { echo "usage: bash run.sh [check | help | smoke|full|eval <balanced|skewed|gender_interpolation|age_interpolation|all> [flags...]]" >&2; exit 64; }

revision_args() {   # the pinned weights of config/models.env, as --*_revision flags
  local rev v
  for rev in ARCHITECT SPRINTER CONTROLNET HED CLIP; do
    v="${rev}_REVISION"
    [ -n "${!v:-}" ] && echo "--$(tr 'A-Z' 'a-z' <<< "$rev")_revision" "${!v}"
  done
}

eval_one() {   # eval_one <scenario> [extra flags...]: the N=2000 final evaluation
  local name="$1"; shift
  local cfg="$HERE/config/${name}.env"
  [ -f "$cfg" ] || { echo "no such scenario: $name (config/$name.env)" >&2; usage; }
  (
    set -a
    source "$HERE/config/models.env"
    source "$cfg"
    set +a
    local outdir="$OUT/eval_${SCENARIO}" a
    for a in "$@"; do [ "${prev:-}" = --output_dir ] && outdir="$a"; prev="$a"; done
    local args=( --mode "$MODE" --controlnet_scale "$CONTROLNET_SCALE"
                 --run_dir "$OUT/full_${SCENARIO}" --n_eval 2000 --seed 42 )
    if [ "$MODE" = age ]; then
      args+=( --age_min "$AGE_MIN" --age_max "$AGE_MAX" --age_step "$AGE_STEP" --age_gender "$AGE_GENDER" )
    else
      local tp; IFS='|' read -r -a tp <<< "$TARGET_PROMPTS"
      args+=( --target_prompts "${tp[@]}" )
    fi
    args+=( $(revision_args) --output_dir "$outdir" --wandb_project "${WANDB_PROJECT:-sdxl-eval}" )
    [ -n "${WANDB_ENTITY:-}" ] && args+=( --wandb_entity "$WANDB_ENTITY" )
    args+=( "$@" )
    export WANDB_NAME="${WANDB_NAME:-eval_${name}${SLURM_JOB_ID:+_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}}"
    export WANDB_TAGS="${WANDB_TAGS:-eval,$name}"
    local cmd=( "$PY" -m cdm.models.sd.eval_all_experiments "${args[@]}" )
    echo "=== eval | $name | wandb ${WANDB_PROJECT:-sdxl-eval}/$WANDB_NAME ==="
    printf '%q ' "${cmd[@]}"; echo
    [ -n "${DRY_RUN:-}" ] && exit 0
    mkdir -p "$outdir"
    {
      echo "date:       $(date -Iseconds)"
      echo "git commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$REPO" diff --quiet 2>/dev/null || echo ' (dirty)')"
      echo "host:       $(hostname)  slurm job: ${SLURM_JOB_ID:-none}"
      "$PY" -c "import torch; print('gpu:       ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none', '| torch', torch.__version__)"
      echo "config:     config/$name.env  overrides: ${*:-none}"
      printf 'command:    '; printf '%q ' "${cmd[@]}"; echo
      "$PY" -c "from cdm.models.sd.run_mlgd_f import library_versions as v; print('versions:   ', v())"
    } > "$outdir/run_info.txt"
    "$PY" -m pip freeze > "$outdir/pip_freeze.txt" 2>/dev/null || true
    "${cmd[@]}"
  )
}

run_one() {   # run_one <tier> <scenario> [extra flags...]
  local tier="$1" name="$2"; shift 2
  local cfg="$HERE/config/${name}.env"
  [ -f "$cfg" ] || { echo "no such scenario: $name (config/$name.env)" >&2; usage; }
  # the config sets SCENARIO, MODE, N_STEPS, ...; a subshell keeps scenarios from leaking into each other
  (
    set -a
    source "$HERE/config/models.env"   # pinned model revisions, shared by every scenario
    source "$cfg"
    set +a

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
    args+=( $(revision_args) --output_dir "$outdir" --wandb_project "$project" )   # pinned weights (config/models.env)
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
      "$PY" -c "from cdm.models.sd.run_mlgd_f import library_versions as v; print('versions:   ', v())"
    } > "$outdir/run_info.txt"
    "$PY" -m pip freeze > "$outdir/pip_freeze.txt" 2>/dev/null || true   # the exact environment
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

  eval)
    SCEN="${2:-}"; [ -n "$SCEN" ] || usage
    shift 2
    if [ -z "${DRY_RUN:-}" ]; then
      "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA GPU visible (sbatch: add --gres=gpu:1)')"
    fi
    if [ "$SCEN" = all ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
      SCEN="${SCENARIOS[$SLURM_ARRAY_TASK_ID]:?array index must be 0-3}"
    fi
    if [ "$SCEN" = all ]; then
      for s in "${SCENARIOS[@]}"; do eval_one "$s" "$@"; done
    else
      eval_one "$SCEN" "$@"
    fi
    [ -n "${DRY_RUN:-}" ] || echo "save into results/: python experiments/stable_diffusion/export_results.py"
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
