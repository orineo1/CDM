# Peptide benchmark: CDM vs GB-FESO on the four-bead toy chain

Head-to-head comparison of our method (MLGD / MLGD-F, `find_xstar.py`) with GB-FESO (Zick & Mendels 2026,
"Gradient-Based Inverse Design of Free-Energy Landscapes with Diffusion Models", arXiv:2607.06421) on
their toy-peptide benchmark: X = (eps1, eps2), the LJ strengths of the 4-bead 2D chain; Y = (r12, r23,
r34, theta123, theta234); the target G is the simulated equilibrium ensemble at a held-out eps*.

GB-FESO runs from THEIR code (github GB-FESO, MIT), cloned next to this repository (`../GB-FESO`, or
`$GBFESO_DIR`): their `train.py` trains the surrogate, their `run_feso.optimize_epsilon_condition` optimizes,
their `utils.py` / `loss.py` score every method. Our method runs from `experiments/synthetic/pipeline/
train_models.py` and `find_xstar.py` unchanged. New code: the simulator, the data conversion, the
GB-FESO / grid-search drivers and the evaluation.

## Pipeline (cluster, from the repo root)

| step | command | output |
|---|---|---|
| 1 data | `sbatch experiments/peptide/slurm/slurm_data.sh` (`slurm_targets.sh` redoes the targets only) | `runs/peptide/{train_ens,targets_ens}.pt`, `data.pt`, `prior_data.pt`, `targets/target_XXX.npy`, `meta.json` |
| 2 networks | `sbatch --array=0-3 experiments/peptide/slurm/slurm_train.sh` | `runs/peptide/models/{gbfeso.pt, diff/, cm/, prior/}` |
| 3a MLGD | `SAMPLER=diff sbatch --array=0-7 experiments/peptide/slurm/slurm_ours.sh` | `runs/peptide/results/MLGD_tXXX.json` |
| 3b MLGD-F | `SAMPLER=cm sbatch --array=0-7 experiments/peptide/slurm/slurm_ours.sh` | `runs/peptide/results/MLGD-F_tXXX.json` |
| 3c GB-FESO | `sbatch --array=0-7 experiments/peptide/slurm/slurm_gbfeso.sh` | `runs/peptide/results/gbfeso_K.json` |
| 3d grid | `sbatch --array=0-3 experiments/peptide/slurm/slurm_grid.sh` | `runs/peptide/results/grid_K.json` |
| 4 evaluation | `sbatch experiments/peptide/slurm/slurm_eval.sh` | `experiments/peptide/results/{results_table.md, per_run.json}` |

## Setup

**Simulator** (`pipeline/simulate.py`): a batched PyTorch port of their LAMMPS input (`lammps_scripts/gener.py`):
harmonic bonds K = 0.15, r0 = 1.5 (LAMMPS convention, no 1/2); plain `lj/cut 4.0` on all six pairs
(`special_bonds lj 1 1 1`): bonded pairs eps1, (1,3),(2,4) eps2, (1,4) 0.5; `fix langevin` T = 0.09,
damp 1 + `fix nve`, dt = 0.005, straight start with spacing 1. LAMMPS integrator order (half kick, drift,
force + friction on the half-step velocity + random force, half kick); Gaussian instead of LAMMPS's uniform
random force (same variance). All chains in one tensor, 200-step blocks replayed as a CUDA graph:
4M steps for any number of chains take ~210 s on an L4/L40S (47 us/step). Kinetic temperature 0.0900.
Not cross-checked against LAMMPS (not installed on the cluster).

**Protocol of P(Y | eps)** (the same for training data, targets and evaluation): independent replicas
of 4,000,000 steps each (their run length) from the straight chain, frames every 200 steps after a
50,000-step burn-in, a random subset kept. Training ensembles use 8 replicas per eps; targets and the
evaluation ensembles use 64 (more replicas are free on the GPU, and with 8 the basin populations of
strongly attractive eps, where basins interconvert slowly, varied enough between two independent
estimates to give a median KDE-KL of 0.06 at eps_hat = eps*). Their data set is one 4M-step trajectory
per eps. A first round against 8-replica targets is kept in `runs/peptide/v1_8rep_targets/` (not reported).

**Data**: 300 eps ~ U[0.21, 1.55]^2, 4,000 frames each (1.2M rows) for every surrogate. 40 held-out
targets eps* ~ U[0.21, 1.55]^2 (seed 2), 20,000 frames each. (GB-FESO's script uses the training
trajectories themselves as targets; we use held-out ones.) 27/40 targets populate >= 2 basins.

**Representations**: GB-FESO keeps its own (raw 5D, distances min-max scaled, angles / pi). Our networks
use x = standardized eps and y = standardized (r12, r23, r34, cos th123, sin th123, cos th234, sin th234),
so the Euclidean MMD treats the angles periodically (as the MNIST rotation task does).

**Networks**: GB-FESO surrogate = their FiLM-MLP DDPM with their TrainConfig (width 256, depth 7, 1000
steps, AdamW 1.5e-4, batch 256, 80 epochs, best epoch). Ours: conditional diffusion (256 units, 4 blocks,
100 steps, 40 epochs, batch 512), consistency model (same size and epochs), prior = unconditional
diffusion on 200k uniform draws of the box.

**Hardware**: all searches ran on NVIDIA L4 GPUs (partition `catfish`), one run at a time per GPU, so the
wall times compare.

**Methods**
- GB-FESO: their `optimize_epsilon_condition` with their CLI defaults (201 Adam steps, lr 5e-2, phase 1
  at lr_refine 5e-2 when the phase-0 best KL > 0.2, 3500 samples, KDE bandwidth 0.5, kl_lambda 1, clamp
  to the box, perturbation std 0.05, 0.005 for high-high starts), 40 DDIM steps (their paper; the code
  default is 100). Four starts per target: low = 0.25, high = 1.4 (their README example).
- MLGD / MLGD-F: `find_xstar.py --guide-grad zo --n-restarts 10 --n-cond 250 --guidance-optim adam
  --step-size 0.05 --refine-steps 100 --refine-lr 0.02 --refine-n-cond 3000 --refine-keep-best
  --select-top 5 --select-n 2000 --select-repeats 5`.
- Grid: 30 x 30 grid of the box, MMD with 1000 samples of our diffusion surrogate per point, top-5
  reselection as in `find_xstar`.
- Oracle: eps_hat = eps* (an independent re-simulation), the noise floor of the metrics.

**Evaluation** (`pipeline/evaluate.py`): every eps_hat (clamped to the box) is re-simulated with the
protocol above and compared with the target ensemble by their `utils.compare_basin_metrics` (basin boxes
in (theta123, theta234)) and their `loss.kde_kl_target_to_generated` (bandwidth 0.5, symmetric, 4096 vs
4096 frames). Success = no target basin with mass > 5% is below 0.5% at eps_hat. dF error = their per-run
median |dF_hat - dF| (k_BT, cap 6) over the non-reference active basins, multi-basin targets only.

## Results (40 held-out targets, 2026-10-01)

`results/results_table.md` (written by step 4; `evaluate.py table --out experiments/peptide/results`
rewrites it from `results/per_run.json`). All metrics on fresh 64-replica simulations at eps_hat.

| method | runs | success | success, mean dF err <= 1 k_BT | median dF err (k_BT) | mean dF err | dF err <= 0.3 | median KDE-KL | mean KDE-KL | median basin TV | median abs(eps_hat - eps*) | time / run (s), median (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **MLGD-F** (all) | 40 | 100.0% | 100.0% | 0.163 (n=26) | 0.189 | 73% | 0.031 | 0.042 | 0.018 | 0.190 | 116.5 (116.7) |
| **MLGD** (all) | 40 | 100.0% | 100.0% | 0.053 (n=26) | 0.077 | 100% | 0.017 | 0.029 | 0.011 | 0.080 | 552.3 (552.8) |
| **GB-FESO** (all) | 160 | 98.8% | 96.9% | 0.095 (n=104) | 0.357 | 85% | 0.022 | 0.130 | 0.013 | 0.200 | 96.8 (100.9) |
| &nbsp;&nbsp;GB-FESO [low-low] | 40 | 100.0% | 100.0% | 0.089 (n=26) | 0.121 | 92% | 0.017 | 0.026 | 0.014 | 0.108 | 96.9 (103.5) |
| &nbsp;&nbsp;GB-FESO [low-high] | 40 | 95.0% | 87.5% | 0.166 (n=26) | 1.058 | 69% | 0.033 | 0.433 | 0.018 | 0.286 | 96.7 (106.4) |
| &nbsp;&nbsp;GB-FESO [high-low] | 40 | 100.0% | 100.0% | 0.084 (n=26) | 0.124 | 92% | 0.022 | 0.031 | 0.011 | 0.156 | 96.7 (96.8) |
| &nbsp;&nbsp;GB-FESO [high-high] | 40 | 100.0% | 100.0% | 0.072 (n=26) | 0.126 | 85% | 0.024 | 0.029 | 0.012 | 0.219 | 96.9 (96.9) |
| **grid** (all) | 40 | 100.0% | 100.0% | 0.106 (n=26) | 0.127 | 92% | 0.019 | 0.028 | 0.012 | 0.182 | 62.2 (62.8) |
| **oracle** (all) | 40 | 100.0% | 100.0% | 0.041 (n=26) | 0.052 | 100% | 0.013 | 0.027 | 0.006 | 0.000 | 0.0 (0.0) |
| **GB-FESO, best of 4 starts** (all) | 40 | 100.0% | 100.0% | 0.124 (n=26) | 0.127 | 92% | 0.018 | 0.025 | 0.011 | 0.136 | 390.4 (403.7) |

Reading the table:
- The oracle row is the noise floor: two independent 64-replica ensembles at the same eps differ by a
  median dF of 0.041 k_BT and a median KDE-KL of 0.013.
- MLGD is the most accurate: median dF error 0.053 k_BT (mean 0.077, every multi-basin target within
  0.3 k_BT), median |eps_hat - eps*| 0.08, against 0.072-0.166 k_BT (mean 0.12-1.06) and 0.11-0.29 for
  GB-FESO per start. Paired per target, MLGD has the smaller dF error than GB-FESO on 16-18 of 26
  multi-basin targets (per start) and the smaller eps error on 26-31 of 40.
- GB-FESO fails only from the low-high start (eps = (0.25, 1.4)): on 5 of 40 targets it ends at the
  eps2 ceiling 1.54 with the target's second basin almost empty (dF errors 2.7-6 k_BT); 2 of these count
  as failures under their 0.5% rule, all 5 under the dF <= 1 k_BT rule. Phase 1 ran on 4 of 160 runs.
- The success rate under their definition is 99-100% for every method: on our simulated data the
  benchmark does not separate the methods by success, only by dF / eps accuracy.
- Cost: one MLGD run (10 restarts + refinement + reselection) takes 552 s on an L4, 5.7 times one
  GB-FESO run (97 s) and 1.4 times GB-FESO over its four starts (390 s, best start by surrogate KL).
  MLGD-F (consistency model) is fast (117 s) but less accurate than the 30 x 30 grid on our diffusion
  surrogate (62 s): its few-step sampler is the limiting factor here.
