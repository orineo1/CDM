Equilibration check: Design(N=5, free=(1, 2, 3, 4), fixed={}), 16 replicas x 4,000,000 steps, burn-in 50,000, 19750 frames per replica

| eps | basins | n active (>5%) | top basins (pop.) | early/late TV | early/late max dF | seeds TV | seeds max dF | first-10% TV |
|---|---|---|---|---|---|---|---|---|
| [0.25, 0.25, 0.25, 0.25] | tuple | 11 | ffF 0.13, fFO 0.10, ffo 0.09 | 0.020 | 0.11 | 0.027 | 0.17 | 0.032 |
| [1.4, 1.4, 1.4, 1.4] | tuple | 3 | fFf 0.54, foF 0.19, fFO 0.18 | 0.090 | 1.65 | 0.149 | 0.94 | 0.089 |
| [0.8, 0.8, 0.8, 0.8] | tuple | 9 | ffF 0.17, fFO 0.12, fff 0.10 | 0.133 | 0.77 | 0.132 | 0.95 | 0.208 |
| [1.29, 1.29, 0.9, 0.59] | tuple | 3 | fFf 0.46, foF 0.29, fFO 0.24 | 0.087 | 0.52 | 0.189 | 0.97 | 0.126 |
| [0.28, 0.72, 0.76, 0.27] | tuple | 1 | fff 0.89, ffF 0.03, ffo 0.02 | 0.022 | 0.00 | 0.025 | 0.00 | 0.039 |
| [0.28, 1.55, 1.08, 0.52] | tuple | 1 | fff 0.99, ffF 0.01, fFf 0.00 | 0.015 | 0.00 | 0.002 | 0.00 | 0.073 |
| [0.79, 1.52, 1.41, 1.34] | tuple | 4 | ffF 0.44, fff 0.20, fFf 0.14 | 0.231 | 2.24 | 0.296 | 4.78 | 0.350 |
| [0.74, 0.87, 1.12, 0.29] | tuple | 4 | fff 0.52, ffo 0.22, fFO 0.08 | 0.221 | 4.45 | 0.086 | 0.38 | 0.584 |

tuple basins, median over eps: early/late TV 0.088, seeds TV 0.109, first-10% TV 0.108; max: 0.231, 0.296, 0.584
