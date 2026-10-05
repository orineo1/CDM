#!/bin/bash
# Submit the whole peptide pipeline for N beads as one dependency chain (run from the repo root on the cluster):
#   N=5 bash experiments/peptide/slurm/submit_all.sh            # data -> networks -> searches -> evaluation
#   N=6 bash experiments/peptide/slurm/submit_all.sh
#   N=5 SKIP_DATA=1 bash ...    (data exist)     N=5 SKIP_TRAIN=1 bash ...   (networks exist too)
#   N=5 AFTER=<jobid> bash ...  (the data job waits for that job, e.g. the equilibration check, to succeed)
# Searches (each on the same GPU type, so that wall times compare), 40 targets:
#   MLGD (10 trajectories), MLGD_r2 (2 trajectories), MLGD-F, GB-FESO (4 starts: all-low, all-high, rand1, rand2),
#   random-1500 and random-8000 (random search on our diffusion sampler's loss, ~ the time of one GB-FESO run /
#   of MLGD), then sampler accuracy at held-out eps* and the evaluation (results_table_N<N>.md).
set -euo pipefail
N=${N:-5}; export N
S=experiments/peptide/slurm
mkdir -p logs/peptide
sb() { sbatch --parsable --export=ALL "$@"; }
dep() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }
D=""; T=""
if [ -z "${SKIP_DATA:-}" ]; then D=$(sb $(dep "${AFTER:-}") $S/slurm_data.sh); echo "data      $D"; fi
if [ -z "${SKIP_TRAIN:-}" ]; then T=$(sb $(dep "$D") --array=0-3 $S/slurm_train.sh); echo "train     $T"; fi
W=${T:-$D}
J1=$(SAMPLER=diff sb $(dep "$W") --array=0-7 $S/slurm_ours.sh);                     echo "MLGD      $J1"
J2=$(SAMPLER=diff RESTARTS=2 TAG=_r2 sb $(dep "$W") --array=0-7 $S/slurm_ours.sh);  echo "MLGD_r2   $J2"
J3=$(SAMPLER=cm sb $(dep "$W") --array=0-7 $S/slurm_ours.sh);                       echo "MLGD-F    $J3"
J4=$(sb $(dep "$W") --array=0-7 $S/slurm_gbfeso.sh);                                echo "GB-FESO   $J4"
J5=$(RANDOM_M=1500 sb $(dep "$W") --array=0-3 $S/slurm_grid.sh);                    echo "rand-1500 $J5"
J6=$(PER=5 RANDOM_M=8000 sb $(dep "$W") --array=0-7 $S/slurm_grid.sh);           echo "rand-8000 $J6"
J7=$(sb $(dep "$W") $S/slurm_sampler.sh);                                            echo "sampler   $J7"
E=$(sb --dependency=afterok:$J1:$J2:$J3:$J4:$J5:$J6 $S/slurm_eval.sh);              echo "eval      $E"
