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
(cluster: `/sci/labs/orzuk/orzuk/venvs/trialanine-venv`; locally a separate venv, not the cdm venv).

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
