# Table 1 (MLGD*) with the GB-FESO competitor, on the retrained samplers

TV(P(Y | x̂), G), mean ± std over 5 runs (seeds 42–46, 30 trajectories each); lower is better; **bold**: the best in
the row. All three methods use the same retrained conditional sampler per cell (diffusion, 256 units, 4 blocks:
`runs/night/grid/<cell>/d256`, at (10, 1) `runs/night/x10_y1/models/d256`) and the Table 1 data and target G
(`runs/table1v2/<cell>/data.pt`). MLGD*: zeroth-order guidance, 30 restarts, refinement 100 steps × 3,000 samples
keeping the best iterate, the 5 lowest-loss restarts re-scored 5 × 5,000 (Or's `G_zo` runs; at (10, 1)
`E6_sel5000_n3000`, the same recipe). GB-FESO (Zick & Mendels, arXiv:2607.06421): 30 starts from random training
designs, no prior over X, their optimizer (201 + 201 Adam steps), MLGD*'s budget: 250 samples per phase-0 step,
3,000 per phase-1 step, each start scored on 1,000, and MLGD*'s selection (the 5 lowest re-scored 5 × 5,000).
GB-FESO (MMD) uses the Table 1 loss, so MLGD* vs GB-FESO (MMD) differs only in the search; GB-FESO (KL) is their
method with their KDE-KL loss.
Setup and commands: `../README.md`, section "GB-FESO on the retrained samplers". Written by
`pipeline/make_table1_d256.py`.

| dim X × dim Y | MLGD* | GB-FESO (KL) | GB-FESO (MMD) |
|---|---|---|---|
| 1×1 | **0.024 ± 0.015** | 0.046 ± 0.013 | 0.030 ± 0.018 |
| 1×5 | **0.017 ± 0.008** | 0.038 ± 0.012 | 0.018 ± 0.012 |
| 1×10 | 0.048 ± 0.033 | 0.182 ± 0.090 | **0.042 ± 0.032** |
| 5×1 | 0.023 ± 0.007 | **0.019 ± 0.007** | 0.024 ± 0.010 |
| 5×5 | 0.063 ± 0.020 | **0.035 ± 0.014** | 0.067 ± 0.022 |
| 5×10 | **0.070 ± 0.034** | 0.178 ± 0.046 | 0.092 ± 0.024 |
| 10×1 | **0.070 ± 0.027** | 0.554 ± 0.163 | 0.666 ± 0.408 |
| 10×5 | 0.066 ± 0.018 | **0.041 ± 0.005** | 0.082 ± 0.026 |
| 10×10 | **0.193 ± 0.177** | 0.302 ± 0.181 | 0.407 ± 0.151 |
| time / run (s), mean | 1410 | 1721 | 1829 |

## All columns

| dim X | dim Y | method | TV mean ± std | ‖x̂ − x*‖ (sd units) | relative L2 | time / run (s) | trajectories with TV<0.5 | in x* basin |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | MLGD* | 0.024 ± 0.015 | 0.006 ± 0.004 | 0.005 ± 0.004 | 1251 | 147/150 | 146/150 |
| 1 | 1 | GB-FESO (KL) | 0.046 ± 0.013 | 0.011 ± 0.003 | 0.015 ± 0.009 | 1688 | 115/150 | 115/150 |
| 1 | 1 | GB-FESO (MMD) | 0.030 ± 0.018 | 0.007 ± 0.005 | 0.008 ± 0.008 | 1828 | 113/150 | 113/150 |
| 1 | 5 | MLGD* | 0.017 ± 0.008 | 0.016 ± 0.009 | 0.002 ± 0.001 | 1252 | 150/150 | 150/150 |
| 1 | 5 | GB-FESO (KL) | 0.038 ± 0.012 | 0.037 ± 0.012 | 0.009 ± 0.006 | 1728 | 121/150 | 121/150 |
| 1 | 5 | GB-FESO (MMD) | 0.018 ± 0.012 | 0.017 ± 0.013 | 0.003 ± 0.003 | 1802 | 128/150 | 128/150 |
| 1 | 10 | MLGD* | 0.048 ± 0.033 | 0.009 ± 0.007 | 0.017 ± 0.017 | 1250 | 150/150 | 150/150 |
| 1 | 10 | GB-FESO (KL) | 0.182 ± 0.090 | 0.037 ± 0.018 | 0.205 ± 0.129 | 1786 | 139/150 | 147/150 |
| 1 | 10 | GB-FESO (MMD) | 0.042 ± 0.032 | 0.008 ± 0.006 | 0.014 ± 0.013 | 1847 | 148/150 | 150/150 |
| 5 | 1 | MLGD* | 0.023 ± 0.007 | 0.374 ± 0.140 | 0.003 ± 0.002 | 1615 | 104/150 | 66/150 |
| 5 | 1 | GB-FESO (KL) | 0.019 ± 0.007 | 0.659 ± 0.304 | 0.002 ± 0.001 | 1648 | 87/150 | 25/150 |
| 5 | 1 | GB-FESO (MMD) | 0.024 ± 0.010 | 0.742 ± 0.549 | 0.003 ± 0.002 | 1838 | 86/150 | 33/150 |
| 5 | 5 | MLGD* | 0.063 ± 0.020 | 0.040 ± 0.013 | 0.018 ± 0.013 | 1261 | 132/150 | 132/150 |
| 5 | 5 | GB-FESO (KL) | 0.035 ± 0.014 | 0.035 ± 0.008 | 0.004 ± 0.002 | 1728 | 97/150 | 98/150 |
| 5 | 5 | GB-FESO (MMD) | 0.067 ± 0.022 | 0.038 ± 0.020 | 0.016 ± 0.011 | 1805 | 93/150 | 93/150 |
| 5 | 10 | MLGD* | 0.070 ± 0.034 | 0.044 ± 0.019 | 0.034 ± 0.038 | 1568 | 150/150 | 150/150 |
| 5 | 10 | GB-FESO (KL) | 0.178 ± 0.046 | 0.088 ± 0.032 | 0.293 ± 0.226 | 1769 | 144/150 | 150/150 |
| 5 | 10 | GB-FESO (MMD) | 0.092 ± 0.024 | 0.049 ± 0.017 | 0.047 ± 0.022 | 1852 | 150/150 | 150/150 |
| 10 | 1 | MLGD* | 0.070 ± 0.027 | 1.773 ± 0.758 | 0.023 ± 0.016 | 1497 | 39/150 | 0/150 |
| 10 | 1 | GB-FESO (KL) | 0.554 ± 0.163 | 4.132 ± 1.013 | 1.145 ± 0.470 | 1672 | 1/150 | 0/150 |
| 10 | 1 | GB-FESO (MMD) | 0.666 ± 0.408 | 3.848 ± 1.335 | 2.198 ± 1.885 | 1823 | 12/150 | 0/150 |
| 10 | 5 | MLGD* | 0.066 ± 0.018 | 0.305 ± 0.173 | 0.050 ± 0.026 | 1502 | 124/150 | 0/150 |
| 10 | 5 | GB-FESO (KL) | 0.041 ± 0.005 | 0.250 ± 0.161 | 0.012 ± 0.005 | 1701 | 61/150 | 0/150 |
| 10 | 5 | GB-FESO (MMD) | 0.082 ± 0.026 | 0.226 ± 0.092 | 0.063 ± 0.078 | 1827 | 68/150 | 0/150 |
| 10 | 10 | MLGD* | 0.193 ± 0.177 | 0.078 ± 0.013 | 0.403 ± 0.626 | 1492 | 112/150 | 0/150 |
| 10 | 10 | GB-FESO (KL) | 0.302 ± 0.181 | 0.093 ± 0.014 | 0.719 ± 0.693 | 1767 | 30/150 | 0/150 |
| 10 | 10 | GB-FESO (MMD) | 0.407 ± 0.151 | 0.069 ± 0.008 | 1.171 ± 0.586 | 1843 | 61/150 | 0/150 |
