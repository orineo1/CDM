# Peptide: comparison with GB-FESO

The four-bead peptide benchmark of GB-FESO (Zick & Mendels 2026, arXiv:2607.06421): `X = (eps1, eps2)`, the
Lennard-Jones strengths of a 2-D four-bead chain in `[0.21, 1.55]²`; `Y = (r12, r23, r34, theta123, theta234)`; the
target `G` is the simulated equilibrium ensemble at a held-out `eps*`. Every returned `eps_hat` is re-simulated
and scored with their basin metrics.

**GB-FESO's code** runs unchanged from their repository, cloned next to this one (`$GBFESO_DIR` overrides):

```bash
git clone https://github.com/MendelsResearchGroup/Gradient-Based-Free-Energy-Surface-Optimization ../GB-FESO
```

## Setup

- **Simulator** (`pipeline/simulate.py`): a batched PyTorch port of their LAMMPS input (harmonic bonds, LJ on all
  pairs, Langevin at T = 0.09, dt = 0.005). Every ensemble uses independent replicas of 4M steps from the straight
  chain, frames every 200 steps after a 50,000-step burn-in: 8 replicas per training `eps`, 64 for targets and for
  the evaluation.
- **Data**: 300 random `eps` with 4,000 frames each; 40 held-out targets (seed 2), 20,000 frames each.
- **Networks**: GB-FESO's surrogate (their FiLM-MLP DDPM and training defaults); ours: conditional diffusion and
  consistency samplers and a diffusion prior on `eps`, on standardized `(r, cos theta, sin theta)`.
- **Methods**: GB-FESO with their defaults (201 Adam steps, 40 DDIM steps, KDE-KL) from four corner starts; MLGD /
  MLGD-F with zeroth-order guidance, 10 restarts, refinement with 3,000 samples, re-selection of the best 5; a
  30 × 30 grid search on our sampler's loss; the oracle `eps_hat = eps*` (the noise floor).
- **Evaluation** (`pipeline/evaluate.py`): their `compare_basin_metrics` and KDE-KL. Success: every target basin
  with mass > 5% keeps ≥ 0.5% at `eps_hat`; strict success also needs a mean ΔF error ≤ 1 k_BT; ΔF error over the
  26 multi-basin targets.

## Run (SLURM, from the repository root)

| step | command | output |
|---|---|---|
| 1 data | `sbatch experiments/peptide/slurm/slurm_data.sh` | `runs/peptide/` |
| 2 networks | `sbatch --array=0-3 experiments/peptide/slurm/slurm_train.sh` | `runs/peptide/models/` |
| 3 MLGD / MLGD-F | `SAMPLER=diff sbatch --array=0-7 experiments/peptide/slurm/slurm_ours.sh` (`SAMPLER=cm` for MLGD-F; `RESTARTS=2 TAG=_r2` for 2 trajectories) | `runs/peptide/results/` |
| 3 GB-FESO | `sbatch --array=0-7 experiments/peptide/slurm/slurm_gbfeso.sh`; 2 more starts: `TAG=_rand EXTRA='--starts rand1,rand2'`; 5.7× longer: `PER=2 TAG=_long EXTRA='--steps 1150 --method-name GB-FESO-long'` with `--array=0-19` | `runs/peptide/results/` |
| 3 grid | `sbatch --array=0-3 experiments/peptide/slurm/slurm_grid.sh` | `runs/peptide/results/` |
| 4 evaluation | `sbatch experiments/peptide/slurm/slurm_eval.sh` | `experiments/peptide/results/{results_table.md, per_run.json}` |

Add `--partition=<gpu partition>` to `sbatch`; all methods should run on one GPU type (the paper: NVIDIA L4) so
that times compare. `python experiments/peptide/pipeline/evaluate.py table --out experiments/peptide/results`
rebuilds the table from `per_run.json`. `pipeline/plot_fes.py` draws free-energy maps (`results/fes_examples.png`).

## Results (`results/results_table.md`, 40 targets)

| method | time / target (s) | mean ΔF error (k_BT) | ΔF error ≤ 0.3 | strict success | median abs(eps_hat − eps*) |
|---|---|---|---|---|---|
| MLGD, 2 trajectories | 113.0 | 0.105 | 96% | 100.0% | 0.067 |
| GB-FESO, one start | 96.6 | 0.457 | 81% | 95.4% | 0.194 |
| MLGD-F, 10 trajectories | 116.5 | 0.189 | 73% | 100.0% | 0.190 |
| grid search, 30 × 30 | 62.2 | 0.130 | 88% | 100.0% | 0.182 |
| MLGD, 10 trajectories | 552.3 | 0.101 | 100% | 100.0% | 0.080 |
| GB-FESO, best of 6 starts | 583.6 | 0.136 | 88% | 100.0% | 0.101 |
| GB-FESO, 5.7× longer, one start | 551.4 | 0.162 | 89% | 99.4% | 0.146 |
| GB-FESO, 5.7× longer, best of 4 starts | 2219.1 | 0.144 | 85% | 100.0% | 0.071 |
| oracle (`eps_hat = eps*`) | – | 0.068 | 96% | 100.0% | 0.000 |

Times are medians. Re-simulating the same `eps` changes a target's ΔF error by a median of 0.04 k_BT, so the
robust differences are in the tail (the share within 0.3 k_BT, the mean error), not in the medians.

## N beads

The scripts also run N = 5, 6 beads (`N=5 bash experiments/peptide/slurm/submit_all.sh`, outputs in
`runs/peptide_N<N>/`; GB-FESO generalized in `pipeline/gbfeso.py`, random search in place of the grid). These runs
are not in the paper. `results/equilibration_N*_4M_cpu.md` are the equilibration checks (`check_equilibration.py`)
that set their longer protocol (16M steps per replica).
