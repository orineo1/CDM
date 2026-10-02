# Shared by the run.sh of every Stable Diffusion experiment: `source "$HERE/../env.sh"`
# after setting HERE (the experiment directory) and OUT (its outputs/). Sets REPO, PY,
# FT_OUT (four_targets/outputs, where the MLGD-F runs live), keeps every cache next to the
# repo, and defines revision_args, flag_value, latest_run, resolve_run_dir, write_run_info.
export PYTHONUNBUFFERED=1   # print lines reach the log as they happen, also under sbatch
SD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SD_DIR/../.." && pwd)"
FT_OUT="$SD_DIR/four_targets/outputs"
PY="${PYTHON:-python}"
# no venv active (e.g. sbatch from a plain login shell): try VENV, then the repo's .venv
if ! command -v "$PY" >/dev/null 2>&1; then
  for cand in ${VENV:+"$VENV/bin/python"} "$REPO/.venv/bin/python"; do
    [ -x "$cand" ] && { PY="$cand"; break; }
  done
  command -v "$PY" >/dev/null 2>&1 || { echo "run.sh: no python on PATH -- activate the venv before sbatch" \
    "(source /path/to/venv/bin/activate), or set VENV=/path/to/venv or PYTHON=/path/to/python" >&2; exit 69; }
fi
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
# deterministic cuBLAS (models.make_deterministic also sets it; here it is set before the
# interpreter starts, which is what cuBLAS requires)
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

revision_args() {   # the pinned weights of four_targets/config/models.env, as --*_revision flags
  local rev v
  for rev in ARCHITECT SPRINTER CONTROLNET HED CLIP; do
    v="${rev}_REVISION"
    [ -n "${!v:-}" ] && echo "--$(tr 'A-Z' 'a-z' <<< "$rev")_revision" "${!v}"
  done
}

flag_value() {   # flag_value <flag> <default> [args...]: the last value given to <flag>
  local flag="$1" val="$2" prev="" a; shift 2
  for a in "$@"; do [ "$prev" = "$flag" ] && val="$a"; prev="$a"; done
  printf '%s' "$val"
}

latest_run() {   # latest_run <Scenario>: the newest finished full run directory, or nothing
  local d
  for d in $(ls -dt "$FT_OUT/full_${1}_seed"* "$FT_OUT/full_${1}" 2>/dev/null); do
    [ -f "$d/metrics.json" ] && { printf '%s' "$d"; return; }
  done
}

resolve_run_dir() {   # resolve_run_dir <Scenario> [args...]: --run_dir, else the latest run
  local d
  d="$(flag_value --run_dir "" "${@:2}")"
  [ -n "$d" ] || d="$(latest_run "$1")"
  if [ -z "$d" ]; then
    [ -n "${DRY_RUN:-}" ] && { printf '%s' "$FT_OUT/full_${1}_seed<S>_<id>"; return; }
    echo "no finished full run of $1 in $FT_OUT: run 'four_targets/run.sh full' first, or pass --run_dir" >&2; exit 66
  fi
  if [ -z "${DRY_RUN:-}" ] && [ ! -d "$d" ]; then echo "no such run directory: $d" >&2; exit 66; fi
  [ -d "$d" ] && (cd "$d" && pwd) || printf '%s' "$d"
}

require_gpu() {   # fail now, not after loading the models
  [ -n "${DRY_RUN:-}" ] && return 0
  "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA GPU visible (sbatch: add --gres=gpu:1)')"
}

write_run_info() {   # write_run_info <dir> <config label> <cmd...>: what produced this output
  local dir="$1" label="$2"; shift 2
  mkdir -p "$dir"
  {
    echo "date:       $(date -Iseconds)"
    echo "git commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$REPO" diff --quiet 2>/dev/null || echo ' (dirty)')"
    echo "host:       $(hostname)  slurm job: ${SLURM_JOB_ID:-none}"
    "$PY" -c "import torch; print('gpu:       ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none', '| torch', torch.__version__)"
    echo "config:     $label"
    printf 'command:    '; printf '%q ' "$@"; echo
    "$PY" -c "from cdm.models.sd.run_mlgd_f import library_versions as v; print('versions:   ', v())" 2>/dev/null || true
  } > "$dir/run_info.txt"
  "$PY" -m pip freeze > "$dir/pip_freeze.txt" 2>/dev/null || true
}
