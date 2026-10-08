# Trialanine in vacuum: feasibility of a CDM benchmark (force-field calibration)

Capped trialanine (Ace-Ala3-Nme, 42 atoms), Amber ff14SB (`openmm` `amber14-all.xml`), vacuum (NoCutoff),
HBonds constraints, 2 fs Langevin-middle at 300 K, friction 1/ps, OpenMM Reference platform (fastest for 42 atoms).
States (Tang, Chen, Roux & Chipot, arXiv:2609.04449, Fig. 2B): each phi_i is '+' (the phi ~ +60 basin) if
0 < phi_i < 120 deg, else '-' (the phi ~ -70 basin); 8 states labeled by (phi1, phi2, phi3).

## Design x (12 dims, energy linear in x)

U(conf; x) = U_ff14SB(conf) + sum_i [x_{i,1} cos phi_i + x_{i,2} sin phi_i + x_{i,3} cos 2phi_i + x_{i,4} sin 2phi_i],
i = 1, 2, 3 (kJ/mol; `system.py`, CustomTorsionForce with global parameters). x0 = 0 is ff14SB. Any x is evaluated
from saved frames by reweighting with B(conf) = the 12 Fourier terms (computed from phi).
Prior: x ~ N(0, (sigma kT)^2 I); sigma = 1 kT per coefficient (|x| ~ 3.5 kT) is the default used below.

## Pipeline (`pipeline/`)

| step | script | output |
|---|---|---|
| build + system | `system.py` | `trialanine.pdb` |
| metadynamics (6 walkers x 12 ns, WT gamma 8, on phi1..3) | `sample.py metad` | `runs/trialanine/metad_bias.npz` |
| frozen-bias production (16 walkers x 50 ns at x0, 800k frames) | `sample.py prod`, `slurm/slurm_prod.sh` | `runs/trialanine/prod_x0_w*.npz` |
| reweighting core | `rw.py` | |
| ESS vs prior width | `prior_ess.py` | `results/prior_ess.{json,png}` |
| reweighted vs direct runs | `validate.py` | `results/validate.{json,png}` |
| landscape (multi-start L-BFGS, components, elastic bands) | `landscape.py`, `slurm/slurm_landscape.sh` | `results/landscape_*.{json,png}` |
| 2-D slice through 3 minimizers (labeled-state loss) | `slice_plot.py` | `results/slice_kl.{json,png}` |
| label-free (count) target modes | `mode_check.py` | `results/mode_check_1.{json,png}` |

Python env: `pip install openmm pymbar numpy scipy matplotlib PeptideBuilder biopython`
(cluster: `$CDM_LAB/venvs/trialanine-venv`; locally a separate venv, not the cdm venv).

## Results (2026-10-07)

- Unbiased MD at x0 is not ergodic: 20 ns, 5 phi flips, 2 of 8 states visited.
- WT metadynamics (6 walkers x 12 ns): every walker visits all 8 states (~1600 phi flips each).
- Frozen-bias production at x0 (16 x 50 ns, 800k frames): all 8 states, per-state Kish ESS 406 (+++) to 12,486;
  total ESS 26,576. x0 populations: --- 0.606, +-- 0.222, -+- 0.156, --+ 0.0098, -++ 0.0034, +-+ 0.0015,
  ++- 0.0008, +++ 0.0001.
- ESS vs prior sd sigma (kT per coefficient; median ESS(x)/ESS(x0)): 0.25: 1.01, 0.5: 0.92, 1: 0.70, 1.5: 0.48,
  2: 0.31, 3: 0.20; median of the smallest per-state ESS stays ~400 for every sigma, >= 100 in all 1200 draws.
- Validation (populations of the 8 states, reweighted from x0 vs independent walkers simulated at x):
  |x| = 2.7 kT TV 0.003 (max |z| 0.65); 7.3 kT TV 0.0005 (0.97); 10.7 kT TV 0.021 (1.63) against walkers with their
  own metadynamics bias at x. With the x0 bias, the 10.7 kT walkers were trapped (7 of 8 never visited phi1 '+').
- Landscape, labeled 8-state target G = P(.|x*) (KL(G || P_x)): every L-BFGS start (24-48 per target, 9 targets)
  reaches L ~ 1e-15; the minimizers spread 3-9 kT and form one connected zero-loss set (straight lines between them
  rise up to 0.36 nats, relaxed elastic bands <= 2e-5). Non-identifiable, not multimodal.
- Dihedral MMD (6 dihedrals, von Mises kernel): one minimizer (= x*), all 16 starts.
- Label-free target (distribution of the number of '+' residues): separated minima, one per residue carrying the '+'
  (3 components; band barriers 0.03-0.28 nats vs a random-x median loss ~0.5), both for engineered and for
  prior-drawn x*.

## CDM benchmark on label-free count targets: MLGD vs GB-FESO (2026-10-08)

Setup (`pipeline/make_data_tri.py`, `slurm/slurm_cdm.sh`; data in `runs/trialanine/cdm/`):
- Design prior for training, for MLGD's prior and for GB-FESO's random starts: x = 2 kT * x_std, with
  x_std ~ N(0, I) truncated to [-2, 2]^12 (|x_j| <= 4 kT). MLGD's prior is the trained unconditional diffusion
  model on 200k draws, as in the other experiments; the exact Gaussian score was not used.
- Conditional sampler, shared by all methods: conditional diffusion (256 units, 4 blocks, 40 epochs) of
  y | x. Here y = the label-free dihedral features (the residues' (phi, psi) sorted by phi, as cos/sin, 12
  dims, standardized). Training data: 1,000 designs x 1,500 frames resampled from the x0 reference frames with
  the importance weights. Per-design ESS: median 10,019, minimum 765.
- Targets: 30 in total, 24 random x* ~ N(0, (1.25 kT)^2) and 6 engineered: -3.5 kT on sin(phi_i) for one
  or two residues. The target ensemble is 20k resampled frames. ESS at x* >= 2,104, so none was dropped.
- Loss feature: the full label-free dihedral ensemble identifies x* exactly (`results/cdm/canon_check.json`:
  all starts converge to x*), so it has one mode. Every search loss is therefore computed on the soft count of
  '+' residues c(y) = sum_i sigmoid(8 sin phi_i) (`pipeline/count_feature.py`). This depends only on the count
  distribution, so the three residue choices are equally good.
- Methods: all ran on L40S GPUs (salmon).
  - GB-FESO: `search_gbfeso_tri.py`, their optimizer loop with their defaults (201 Adam steps, lr 0.05,
    3,500 samples, KDE-KL bandwidth 0.5, phase 1 only if KL > 0.2, which never happened). Run from
    x0 = ff14SB, and as the best of 4 starts from the prior.
  - MLGD: `search_mlgd_tri.py` = search_mog with the LGD preset, pathwise gradient, Adam guidance step 0.05,
    10 restarts, n_cond 250, refinement of 100 steps x 3,000 samples with keep-best, and top-5 re-selection.
    Run with the count KDE-KL and with the count MMD.
  - Hybrid: GB-FESO run from MLGD's top-3 restarts.
- Evaluation (`pipeline/eval_tri.py`): exact, by reweighting the reference frames. Solved = count TV <= 0.05.

Results: `results/cdm/results_table.md`, `results/cdm/per_run.json`. Summary:

| method | solved (TV <= 0.05) | solved (TV <= 0.10) | median count TV | x*'s mode | median time (s) | samples (M) |
|---|---|---|---|---|---|---|
| GB-FESO, x0 start | 43% | 90% | 0.059 | 43% | 30 | 0.7 |
| GB-FESO, best of 4 | 50% | 67% | 0.051 | 57% | 111 | 2.8 |
| MLGD-KL | 33% | 60% | 0.084 | 30% | 595 | 3.3 |
| MLGD-MMD | 33% | 73% | 0.073 | 40% | 614 | 3.3 |
| MLGD-KL + polish | 40% | 60% | 0.088 | 37% | 678 | 5.4 |
| MLGD-MMD + polish | 37% | 67% | 0.064 | 30% | 701 | 5.4 |

**The sampler limits every method.** Its count distribution at x* is off by a median TV of 0.062 (max 0.30;
`results/cdm/sampler_check.json`), which is larger than the solve threshold. Every method's result correlates
with the sampler's error at the target (r = 0.12-0.62). On the 14 targets where the sampler is accurate
(TV <= 0.05 at x*), the methods solve 36-64%; on the other 16 they solve 19-44%.

**The methods do not separate.** Paired on the count TV, MLGD-MMD beats GB-FESO best-of-4 on 19 of 30 targets
but GB-FESO-x0 on only 13.

**The count loss is multimodal.** Solved runs often land on a residue other than x*'s: 10-80% depending on the
method and the target's mode.

## Exact-sampler comparison on the same 30 count targets (2026-10-08)

`pipeline/search_exact_tri.py` and `slurm/slurm_exact.sh` (CPU, job 46333084); results in `results/cdm_exact/`.

**Setup.** The trained sampler is replaced by the exact conditional, obtained by reweighting the 800k reference
frames. Every method minimizes the same deterministic loss, L(x) = KL(G_count || P_count(x)), with gradients
from autograd through the weights.
- MLGD: search_mog's loss-oracle mode (IdentitySampler). L is evaluated at the Tweedie estimate inside the
  guidance steps, the refinement and the re-selection; no frames are resampled. Prior, box (|x_j| <= 4 kT),
  10 restarts, keep-best refinement and top-5 re-selection are as before.
- GB-FESO: their Adam settings on L, started from ff14SB, as the best of 4 prior starts, and with 20 prior
  starts. The 20-start row (randEq) matches MLGD's number of loss evaluations (about 4,000).
- Hybrid: Adam polish from MLGD's top 3 restarts.

| method | solved (TV <= 0.05) | solved (TV <= 0.02) | median count TV | x*'s mode | median abs(x - x*) (kT) | loss evals | median time (s, CPU) |
|---|---|---|---|---|---|---|---|
| GB-FESO, x0 start | 100% | 100% | 0.002 | 47% | 4.9 | 202 | 7 |
| GB-FESO, best of 4 | 100% | 100% | 0.002 | 60% | 6.4 | 808 | 28 |
| GB-FESO, 20 starts (equal evals) | 100% | 100% | 0.002 | 50% | 7.3 | 4,040 | 138 |
| MLGD | 100% | 100% | 0.001 | 47% | 7.7 | 3,995 | 157 |
| MLGD + polish | 100% | 100% | 0.001 | 47% | 7.7 | 4,601 | 178 |

**Finding.** With an exact sampler every method solves every target, including a single local Adam run from
ff14SB (202 evaluations). The count loss is multimodal: across methods, 24 of 30 targets were answered with at
least two different residue modes, and about half of all answers use a residue other than x*'s. But every mode
reaches zero loss, and there are essentially no bad local minima:
- 94% of the 600 individual GB-FESO random starts end below KL 1e-3, and the worst ends at 0.02.
- 99% of MLGD's 300 restarts end below 1e-3.

MLGD's global search therefore brings no benefit here and costs 20 times the evaluations of a single local run.
The earlier gaps came from sampler error. ESS stays above 500 at every returned design except MLGD on the two
double-'+' targets 27 and 28 (ESS 487 and 376); both are still solved.
