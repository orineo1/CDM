# Search × gradient estimator (the 2 × 2 design), on the retrained samplers

Asked for by Reviewer Qk8S (W1), the Area Chair (point 1) and the AI review of AISTATS 2027 submission 1135.
`tab:sim_gbfeso` compares MLGD\* (diffusion search, finite differences in the guided steps) with GB-FESO (MMD)
(direct search, backpropagation), so the search and the gradient estimator change together. This table adds the
two missing corners.

TV(P(Y | x̂), G), mean ± std (population std, as in `tab:sim_gbfeso`) over 5 runs (seeds 42–46, 30 starts each);
lower is better. All arms: the same retrained conditional sampler per cell (diffusion, 256 units, 4 blocks), the
Table 1 data and target, the MMD loss, 250 samples per search step, 3,000 per refinement / phase-1 step, and the
same selection (the 5 lowest re-scored 5 × 5,000).

| dim X × dim Y | MLGD\*, FiniteDiff | MLGD\*, Backprop | GB-FESO, FiniteDiff | GB-FESO, Backprop | GB-FESO, FiniteDiff both phases |
|---|---|---|---|---|---|
| 1×1 | 0.025 ± 0.015 | 0.032 ± 0.019 | 0.017 ± 0.012 | 0.030 ± 0.018 | – |
| 1×5 | 0.017 ± 0.008 | 0.018 ± 0.006 | 0.010 ± 0.008 | 0.018 ± 0.012 | – |
| 1×10 | 0.048 ± 0.034 | 0.047 ± 0.034 | 0.052 ± 0.016 | 0.042 ± 0.032 | – |
| 5×1 | 0.023 ± 0.007 | 0.029 ± 0.011 | 0.025 ± 0.006 | 0.024 ± 0.010 | – |
| 5×5 | 0.064 ± 0.021 | 0.077 ± 0.027 | 0.074 ± 0.028 | 0.067 ± 0.022 | – |
| 5×10 | 0.070 ± 0.033 | 0.086 ± 0.042 | 0.108 ± 0.041 | 0.092 ± 0.024 | – |
| 10×1 | 0.070 ± 0.027 | 0.075 ± 0.016 | 0.346 ± 0.331 | 0.666 ± 0.408 | 0.315 ± 0.138 |
| 10×5 | 0.065 ± 0.017 | 0.077 ± 0.027 | 0.066 ± 0.013 | 0.082 ± 0.026 | – |
| 10×10 | 0.193 ± 0.178 | 0.407 ± 0.147 | 0.336 ± 0.142 | 0.407 ± 0.151 | 0.276 ± 0.174 |

Columns:

- **MLGD\*, FiniteDiff**: the paper's MLGD\* (`runs/night/grid/<cell>/G_zo_s*.json`; at (10, 1)
  `runs/night/x10_y1/E6_sel5000_n3000_s*.json`). Finite differences in the guided steps, backpropagation in
  the refinement.
- **MLGD\*, Backprop**: the same recipe with `--guide-grad pathwise`; the saved arguments differ in that flag
  only (`runs/night/grid/<cell>/G_pathwise_s*.json`, run on 1 Oct; at (10, 1)
  `runs/search_x_estimator/x10_y1/mlgd_pw_s*.json`, run on 19 Nov).
- **GB-FESO, FiniteDiff**: GB-FESO (MMD) with MLGD\*'s finite-difference gradient in phase 0 and
  backpropagation in phase 1, which mirrors MLGD\* (`gbfeso_zo_pw`, all nine cells, 19 Nov).
- **GB-FESO, Backprop**: the paper's GB-FESO (MMD) column (`table1_gbfeso_d256.md`).
- **GB-FESO, FiniteDiff both phases**: `gbfeso_zo_zo`, run at the two hard cells only (about 7 hours per run).

## The two hard cells, per seed (42–46)

| cell | arm | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|
| 10×1 | MLGD*, FiniteDiff | 0.109 | 0.038 | 0.044 | 0.066 | 0.093 |
| 10×1 | MLGD*, Backprop | 0.073 | 0.101 | 0.053 | 0.080 | 0.069 |
| 10×1 | GB-FESO, FiniteDiff then Backprop | 0.104 | 0.944 | 0.102 | 0.469 | 0.110 |
| 10×1 | GB-FESO, FiniteDiff in both phases | 0.503 | 0.090 | 0.257 | 0.387 | 0.337 |
| 10×10 | MLGD*, FiniteDiff | 0.530 | 0.066 | 0.082 | 0.219 | 0.069 |
| 10×10 | MLGD*, Backprop | 0.470 | 0.122 | 0.431 | 0.462 | 0.548 |
| 10×10 | GB-FESO, FiniteDiff then Backprop | 0.252 | 0.375 | 0.117 | 0.396 | 0.538 |
| 10×10 | GB-FESO, FiniteDiff in both phases | 0.219 | 0.088 | 0.482 | 0.107 | 0.482 |

## Reading

- **Seven easy cells.** All four corners agree within one standard deviation. Neither the search nor the
  estimator matters there.
- **(10, 1): the search decides.** The diffusion search reaches 0.07 with either estimator (0.070 with
  FiniteDiff, 0.075 with Backprop). Direct search does not: 0.67 with Backprop and 0.35 with FiniteDiff
  (0.32 with FiniteDiff in both phases). Finite differences halve the error of GB-FESO but leave it about five
  times above the diffusion search: 1 of the 10 GB-FESO runs with finite differences ends below TV 0.1,
  against 8 of the 10 diffusion-search runs.
- **(10, 10): the estimator decides, and only together with the diffusion search.** With Backprop the two
  searches are the same (0.41 and 0.41). FiniteDiff brings the diffusion search to 0.19 and GB-FESO to 0.34
  (0.28 with FiniteDiff in both phases). The run-to-run spread is large (0.13–0.18), and MLGD\* has one bad
  seed (0.53), so the difference between 0.19 and 0.34 is suggestive, not established; 3 of 5 MLGD\* runs end
  below 0.1 against 0 of 5 for GB-FESO with FiniteDiff.
- **What this means for the paper.** The (10, 1) result of MLGD\* (0.07 against 0.49 for MLGD in Table 1) is
  not due to finite differences: Backprop gives 0.075 under the same recipe. It comes from the wider sampler
  and the larger budget. The evidence for finite differences is (10, 10), 0.19 against 0.41.
- **Overall.** The claim of App. B.9 that the gap is due to the search holds at (10, 1) and not at (10, 10).
  MLGD\* with FiniteDiff is the only corner that is good at both cells.

## Where the runs are

On the cluster, `/sci/labs/orzuk/orzuk/github/scratch_2x2/CDM` (a clone at commit `1ff3388`; `runs/night` and
`runs/table1v2` are links to the main clone). Jobs `search_x_est` 46780458 (50 tasks) and 46780459 (10 tasks),
19 Nov 2026, all completed: `experiments/synthetic/slurm/slurm_search_x_estimator.sh`. Scored with

    python experiments/synthetic/pipeline/summarize_runs.py --runs-dir runs/search_x_estimator/<cell> \
        --data runs/table1v2/<cell>/data.pt
