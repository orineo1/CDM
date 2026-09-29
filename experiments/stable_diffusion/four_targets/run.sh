#!/usr/bin/env bash
# MLGD-F on four distributional targets, plus the baseline scribbles.
#
# Three tiers, and they are NOT interchangeable:
#
#   bash run.sh check              CPU, seconds, no downloads, no GPU.
#                                  Static integrity of the ported package + proof that
#                                  the SD MMD is the shared compute_mmd.
#                                  This is the ONLY tier that runs without a GPU.
#
#   bash run.sh smoke <scenario>   GPU REQUIRED. Tiny: 6 guided steps, 6 variations.
#                                  Proves the pipeline runs end to end. It reproduces
#                                  NO published number and must never be reported as if
#                                  it did.
#
#   bash run.sh full <scenario>    GPU REQUIRED (L40S 48 GB for the optimisation;
#                                  the paper's 2,000-image evaluation used an A100 80 GB).
#                                  The published configuration, from config/<scenario>.env.
#
# <scenario> is one of: balanced | skewed | gender_interpolation | age_interpolation
# Every flag is literal in this file or in config/; nothing is read from a hidden default.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-python}"
OUT="$HERE/outputs"
TIER="${1:-check}"
mkdir -p "$OUT"
cd "$REPO"

load_cfg() {
  local name="${1:?usage: bash run.sh $TIER <scenario>}"
  local f="$HERE/config/${name}.env"
  [ -f "$f" ] || { echo "no such scenario config: $f" >&2; exit 64; }
  # shellcheck disable=SC1090
  set -a; source "$f"; set +a
}

common_args() {
  echo --mode "$MODE" \
       --base_zeta "$BASE_ZETA" \
       --guidance_scale "$GUIDANCE_SCALE" \
       --controlnet_scale "$CONTROLNET_SCALE" \
       --backsel_rule "$BACKSEL_RULE" \
       --loss_fn "$LOSS_FN" \
       --seed "$SEED"
}

case "$TIER" in
  check)
    "$PY" -m cdm.models.sd.check_imports
    "$PY" -m cdm.models.sd.check_mmd_equivalence
    ;;

  smoke)
    load_cfg "${2:-}"
    ARGS=( $(common_args)
           --n_steps 12 --start_step 6
           --num_variations 6 --backsel_k 3
           --output_dir "$OUT/smoke_${SCENARIO}"
           --wandb_project sdxl-smoke )
    if [ "$MODE" = age ]; then
      ARGS+=( --age_min "$AGE_MIN" --age_max "$AGE_MAX" --age_step 10 --n_per_age 1 --age_gender "$AGE_GENDER" )
    else
      IFS='|' read -r -a TP <<< "$TARGET_PROMPTS"
      ARGS+=( --target_prompts "${TP[@]}" )
    fi
    "$PY" -m cdm.models.sd.run_mlgd_f "${ARGS[@]}"
    ;;

  full)
    load_cfg "${2:-}"
    ARGS=( $(common_args)
           --n_steps "$N_STEPS" --start_step "$START_STEP"
           --num_variations "$NUM_VARIATIONS" --backsel_k "$BACKSEL_K"
           --output_dir "$OUT/full_${SCENARIO}"
           --wandb_project sdxl-mlgdf )
    if [ "$MODE" = age ]; then
      ARGS+=( --age_min "$AGE_MIN" --age_max "$AGE_MAX" --age_step "$AGE_STEP" \
              --n_per_age "$N_PER_AGE" --age_gender "$AGE_GENDER" )
    else
      IFS='|' read -r -a TP <<< "$TARGET_PROMPTS"
      ARGS+=( --target_prompts "${TP[@]}" )
    fi
    "$PY" -m cdm.models.sd.run_mlgd_f "${ARGS[@]}"

    # Baseline scribbles (Average prompt, SDEdit Best). Needs the MLGD-F wall-clock
    # in minutes, which sets the SDEdit candidate budget, and an experiments/<SCENARIO>/
    # directory holding scribble_source.png and scribble_mlgdd.png.
    # "$PY" -m cdm.models.sd.eval_baselines --experiment "$SCENARIO" --lgd_cm_minutes <minutes>
    ;;

  *) echo "usage: bash run.sh [check | smoke <scenario> | full <scenario>]" >&2; exit 64 ;;
esac
