# Rugged (residue-design) peptide model — stages 0-2

Model: 2D bead chain, N = 8, pair (i, j) at sequence distance d: eps_ij = eps_d * sqrt(eta_i eta_j),
sigma_ij = (sigma_i + sigma_j)/2, chain strengths fixed eps_d = 0.5, design X = (sigma_1..8, eta_1..8)
(p = 16, continuous, uniform prior on sigma in [0.7, 1.4], eta in [0.3, 2.0]). The benchmark of Zick & Mendels
is the special case of identical residues (`simulate.Design`, bit-identical for the old N = 4/5/6 runs).
Code: `simulate.py`, `probe_residue.py`, `landscape.py`, `slurm/slurm_rugged.sh` (commit aa325d8 + this one).
Runs on the cluster under `runs/rugged/` (jobs 46315621-3, 46316584-5).

## Stage 0 — equilibration (5 random sequences, 8 replicas x 16M steps)

| configuration | seeds TV median / max | autocorrelation time median / max (steps) |
|---|---|---|
| T 0.09 (benchmark) | 0.24 / 0.98 | 132k / 21.8M (longer than the run) |
| T 0.09, eta in 0.5:1.5 | 0.25 / 0.83 | 87k / 3.1M |
| T 0.12 | 0.14 / 0.48 | 18k / 786k |
| **T 0.15** | **0.10 / 0.24** | **5.6k / 43k** |

At the benchmark temperature N = 8 residue chains do not equilibrate. **Chosen: T = 0.15**, default ranges; max
tau 43k << 4M steps. The planned "basin TV < 0.05" is not reachable at N = 8 with 2,048 conformation classes
(the residual ~0.1 is the histogram noise floor), so mixing is judged by tau.

## Stage 1 — one-parameter sweeps (3 base sequences, 15 values, 64 replicas)

Contacts of the swept bead (r < 1.3 sigma_ij) vs sigma_4, eta_4 (middle bead), sigma_1, eta_1 (end bead):
- T 0.09: 4 interior bumps in 66 contact curves (mostly monotone; noisy, not equilibrated).
- **T 0.15: 0 bumps; 47 monotone, 19 flat.** Stickiness raises contacts monotonically; size barely matters.
- The chain is a fluctuating compact globule (basin entropy 4.4-6.6 nats; non-local contacts 10-90 %), not
  a set of distinct folds; 2,048-class basin populations are too fine to be informative at N = 8.

The predicted steric-fit bumps (non-monotone response to size) do not appear at an equilibrating temperature.
Plot: `rugged_stage1_sweeps.png`.

## Stage 2 — true-loss landscape (5 hidden targets; energy distance on standardized features)

Per target: 500 uniform random designs, 10 lines x 20 points from random designs to x*, 20 perturbations of x*
at radii 0.02-0.4 (std of X); 16 replicas x 4M steps per design (64 for the target). Noise of one design's loss:
the mean energy distance of designs at radius 0.02 (~ x*). Barrier = a rise of > 3 noise levels after an
earlier minimum along a line (3-point median); calibrated on the N = 4 benchmark.

| | N = 4 benchmark (calibration) | N = 8 residue model (T 0.15) |
|---|---|---|
| random designs within 5x noise of target | 1-36 % (median 25 %) | **0 / 500 for every target** |
| best random design / noise | ~0.1-1.5x | **10-16x** |
| lines with a barrier | 0-2 / 10 (false-positive level) | **0 / 10 for every target** |
| loss vs radius around x* | smooth | smooth (0.008 at 0.02 -> 0.06-0.09 at 0.4) |
| Spearman(distance to x*, loss) over random designs | 0.75 | 0.48 |

Tables: `rugged_stage2_table_N8.md`, `rugged_stage2_table_N4.md`; plots: `rugged_stage2_landscape_N8.png`,
`rugged_stage2_landscape_N4.png` (line scans of target 0; mean loss vs perturbation radius).

## Conclusion

- **Random search becomes hopeless** (p = 16: no random design of 500 near any target), as intended.
- **The landscape is not rugged**: no barriers on any of 50 straight paths to x*, a smooth basin around x*,
  and monotone single-parameter responses. Interactions here are all attractive and additive
  (eps_ij = eps_d sqrt(eta_i eta_j)), so the loss is close to a single funnel; gradient descent from a random
  start (GB-FESO) should reach x* when the surrogate is accurate.
- **Funnel**: yes in the sense of no barriers; the weaker distance-loss correlation (0.48) points to flat or
  degenerate directions (designs far from x* with similar loss), not to separate minima.

**Recommendation:** do not proceed to stage 3 with this model — it would likely favour local descent, not MLGD.
To create competing minima the interactions must frustrate: e.g. sign-changing pair terms
(eps_ij = eps0 + c q_i q_j with a continuous "charge" q_i in [-1, 1], or HP-like attraction between like
residues and repulsion between unlike ones), which make the best placement of one residue depend on the others.
Such models are also glassier, so they need either a higher temperature or replica exchange to equilibrate;
re-run stages 1-2 (cheap: ~1 GPU-hour) on the new model before any training.
