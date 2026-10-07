# Rugged peptide benchmark with charged residues (2026-10-07)

Track 2 (experiments only; nothing here is in the paper). Goal: a design landscape with separated local minima,
where single-start gradient descent on the design (GB-FESO) gets trapped and a guided search from many starts
(MLGD) should not. Earlier attempt (`../rugged_stage0-2.md`: residue sizes and stickiness) was not rugged.

## Model (stage 1)

`simulate.Design(N, free="charge", fixed="eps=0.5", elec={A, debye, rcut})`: the N-bead 2D LJ chain of the
benchmark (bonds K = 0.15, r0 = 1.5; LJ eps_d = 0.5 on every pair; T = 0.09) plus a charge q_i in [-1, 1] on
every bead with screened Coulomb U_ij = A q_i q_j exp(-r / 2) / r (r < 6) on every pair. Like charges repel,
unlike ones attract, so the best charge of a bead depends on the others. Design X = (q_1..q_N); q and -q have
identical energies (the evaluation measures the design error up to this flip). Designs without charges are
bit-identical to before (checked on N = 4, 5 legacy and N = 8 sigma/eta designs).

## Ruggedness test (stage 2) — `pipeline/reweight.py`, `slurm/slurm_rugged2.sh`

With only charges free the energy is linear in q_i q_j, so the ensemble of ANY design is a reweighting (MBAR) of a
pool of simulated ensembles (256 prior designs x 2,000 frames). The loss (MMD^2 on the standardized features, as
MLGD) is then exact up to MBAR noise, cheap and differentiable. Per configuration, 8 hidden targets q* (64
replicas, the benchmark protocol): MBAR check at q* (loss and basin TV vs the simulated target), 1,000 random
designs, projected Adam on the true loss from 128 random starts (= ideal single-start GB-FESO with an exact
surrogate), clustering of the endpoints, line scans spurious minimum -> q*, and a **direct re-simulation** (32
replicas, no reweighting) of those lines for the two best spurious minima per target.
'Solved' = the endpoint's loss is within 5% of (median random loss - loss at q*) of the loss at q*.

| config | N | A | fraction of single starts reaching q* (mean over 8 targets; min) | targets with spurious minima | direct-simulation lines with a barrier | equilibration (8 designs incl. 4 charge corners): tau median / random designs |
|---|---|---|---|---|---|---|
| c0 | 4 | 1 | 0.99 | 1/8 (a plateau) | - | fine |
| c1 | 4 | 3 | 0.98 (0.85) | 1/8 (a plateau, rise 1%) | 0/2 | tau <= 0.03M |
| c2 | 4 | 3 (debye 1) | 0.99 | 1/8 | - | fine |
| c3 | 5 | 3 | 0.91 (0.65) | 5/8 | 8/9 | 0.3M / 26k-344k |
| c4 | 4 | 3, + sizes sigma_i free | 0.92 | 4/8 | (MBAR weak: ESS ~ 10^3, basin TV at q* 0.01-0.04) | - |
| c5 | 4 | sizes only, eps 1.0 | 0.93 | 6/8 | (MBAR weak: basin TV at q* 0.02-0.17) | - |
| c6 | 6 | 3 | 0.68 (0.55) | 8/8 | 14/16 | 2.7M; random designs up to > 9M (not equilibrated) |
| c7 | 6 | 5 | 0.58 (0.45) | 8/8 | 13/16 | 3.4M; worse |
| **c8** | **5** | **5** | **0.77 (0.45)** | **8/8** | **14/14** | 1.0M / 50k-1.3M (seeds TV 0.02-0.08 at 16 replicas) |

Reading:
- **N = 4: not rugged** (98-99% of single starts reach q*). Reason: the energy is linear in the pair couplings
  q_i q_j, and the relative entropy (forward KL) of an exponential family is convex in its natural parameters,
  so ruggedness can come only from the map q -> (q_i q_j). With all 6 pairs of N = 4 effectively observed (bond
  lengths too), that map adds only the exact q ~ -q symmetry. The same argument explains the earlier
  sigma/eta failure (eps_ij = eps_d sqrt(eta_i eta_j) is a monotone reparametrization).
- **N = 5, 6 with strong charges are rugged**: separated local minima, mostly at faces of the charge box
  (|q_i| = 1, wrong sign pattern), with loss 10-300x the loss at q*, basin TV 0.25-0.9 to the target. Direct
  simulation confirms the barriers: along the straight line from the spurious minimum to q* the true MMD^2
  first RISES by 0.02-0.38 (noise ~1e-3) and then falls to ~1e-3 at q* (c8: 14/14 lines; table in
  `c8_landscape.md`, plot `c8_landscape_lines.png`).
- Ruggedness grows with N and A, equilibration gets worse with both. **Chosen: c8 (N = 5, A = 5)**: 23% of
  ideal single starts are trapped on average (up to 55% on 2 of 8 targets), random designs equilibrate within
  the 16M-step protocol. c3 (N = 5, A = 3; 9% trapped) is run as a milder second benchmark.

Files: `c<k>_landscape.md` (per-target table + direct-simulation lines), `c<k>_landscape_lines.png`,
`c<k>_equil.md`. Raw JSON (endpoints, loss trajectories, MBAR diagnostics) on the cluster in
`runs/rugged2/c<k>/landscape.json`.

## The ruggedness belongs to the MMD, not to the design problem (forward-KL check)

The same MBAR landscapes scored with the forward KL(target || design) of Gaussian KDEs (bandwidth 0.5 on the
standardized features; `reweight.py --loss fkl`; the convex direction of GB-FESO's symmetric KDE-KL):

| config | MMD^2: single starts reaching q* | forward KL: single starts reaching q* | forward KL: targets with spurious minima |
|---|---|---|---|
| c1 (N = 4, A = 3) | 0.98 | 1.00 | 0/8 |
| c3 (N = 5, A = 3) | 0.91 | 1.00 (8/8 targets) | 0/8 |
| c8 (N = 5, A = 5) | 0.77 | 1.00 (8/8 targets) | 0/8 |

(`c<k>_landscape_fkl.md`; one target per config first returned NaN from a constant term, rerun after the fix:
solved 1.00.) This is what the exponential-family argument predicts: the relative entropy is convex in the pair
couplings q_i q_j, and the rank-1 map q -> q_i q_j adds no spurious minima here. The MMD is not convex in the
couplings, and its kernel saturates between well-separated conformations, which creates the face minima. GB-FESO
optimizes a symmetric KDE-KL, so its landscape on this benchmark is essentially single-basin.

## Comparison on 40 held-out targets (stage 3-4)

Same pipeline as the benchmark (`slurm_data.sh` with `FREE=charge FIXED=eps=0.5 ELEC=A=..,debye=2,rcut=6`,
`slurm_train.sh`, `slurm_ours.sh`, `slurm_gbfeso.sh`, `slurm_eval.sh`; N = 5 protocol 16M / 1M / 800; 600
designs x 2,000 frames; 40 targets x 64 replicas). GB-FESO: their optimizer on their surrogate from 10 uniform
random starts per target (`--starts rand1..rand10`): "GB-FESO (all)" = a single start (400 runs), "best of 10
starts" = the start with the lowest surrogate KL. MLGD: 10 trajectories + refinement (MMD), MLGD_r2 = 2
trajectories. Searches on H200 (goldfish) and, for A = 5, partly L40S (salmon), so times are indicative only.
Scores: re-simulation (64 replicas) of every returned design. Design error = |x_hat - x*| up to the charge flip.

**c3 model (N = 5, A = 3)** — `results_table_data_N5.md`:

| method | mean dF err (k_BT) | median dF err | dF err <= 0.3 | success, mean dF <= 1 | median design err | time / run |
|---|---|---|---|---|---|---|
| GB-FESO, single start (400 runs) | **0.179** | 0.145 | **86%** | 99.5% | **0.164** | 30 s |
| GB-FESO, best of 10 starts | **0.166** | 0.145 | 85% | 100% | 0.148 | 305 s |
| MLGD_r2 (2 trajectories) | 0.243 | 0.148 | 78% | 95.0% | 0.222 | 58 s |
| MLGD (10 trajectories) | 0.224 | 0.185 | 75% | 97.5% | 0.194 | 283 s |
| oracle (x* re-simulated) | 0.056 | 0.045 | 100% | 100% | 0 | - |

**c8 model (N = 5, A = 5)** — full table `results_table_data_N5_A5.md` was still being written by the evaluation job
(46327626) when the cluster became unreachable. Design error from the result files (no re-simulation needed):

| method | runs | median design err | err < 0.3 | err < 0.5 | err > 1 (trapped) |
|---|---|---|---|---|---|
| GB-FESO, single start | 254 of 400 (at the time) | **0.183** | **84%** | 97% | 3% (7 runs; all had phase 1, surrogate KL 1.8-6.5) |
| MLGD (10 trajectories, MMD) | 40 | 0.270 | 55% | 90% | 0% |
| MLGD_r2 | 40 | 0.293 | 55% | 85% | 0% |
| MLGD, KDE-KL loss (re-simulated) | 40 | 0.308 | | | mean dF err 0.385, 68% <= 0.3 k_BT |

On targets 1 and 3 (the same designs as c8 targets 1, 3, where 52-55% of ideal MMD descents were trapped) all 10
GB-FESO starts reached x* (design error 0.05-0.25).

## Conclusion

- The charged model is rugged **for the MMD** (separated minima, confirmed by direct simulation), but **not for
  KL-type losses**: forward-KL descent reaches x* from 100% of starts on the same landscapes, and GB-FESO
  (symmetric KDE-KL on its surrogate) is trapped in ~3% of single starts. MLGD's own loss is the rugged one.
- On both charged benchmarks GB-FESO from one random start is **more accurate** than MLGD (c3: mean dF error
  0.18 vs 0.22 k_BT, 86% vs 75% within 0.3 k_BT; c8: design error 0.18 vs 0.27). The benchmark does not show
  the hoped-for MLGD advantage.
- To get a design problem that is multimodal for KL too, the energy must depend on the design non-linearly
  in a way that makes the reachable set of natural parameters strongly non-convex (relative entropy is
  convex in linear couplings, so pair strengths, stickiness and charges all give near-convex landscapes).
