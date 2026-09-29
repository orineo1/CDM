#!/usr/bin/env bash
# Random-prior-draw null baseline (App. B.7)
#
# THERE IS NO PRODUCER. No script in this repository, and none in
# orineo1/GlobalConditional or shaulTol/conditional-matching-paper, runs this
# experiment. See README.md. This script deliberately exits 2 rather than
# succeeding with nothing done.
set -euo pipefail
export PYTHONUNBUFFERED=1   # print lines reach the log as they happen, also under sbatch
cat <<'EOF'
STATUS: NO PRODUCER
The prior-draw null baseline has no producer in either repo.
See README.md. Nothing to run.
EOF
exit 2
