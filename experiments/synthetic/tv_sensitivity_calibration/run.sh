#!/usr/bin/env bash
# TV sensitivity calibration (App. B.7)
#
# THERE IS NO PRODUCER. No script in this repository, and none in
# orineo1/GlobalConditional or shaulTol/conditional-matching-paper, runs this
# experiment. See README.md. This script deliberately exits 2 rather than
# succeeding with nothing done.
set -euo pipefail
export PYTHONUNBUFFERED=1   # print lines reach the log as they happen, also under sbatch
cat <<'EOF'
STATUS: NO PRODUCER
The delta-displacement TV calibration has no producer in either repo.
Table 1's caption depends on it. See README.md. Nothing to run.
EOF
exit 2
