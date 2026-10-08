| method | targets | solved (count TV <= 0.05) | solved, random / engineered | median count TV | mean count TV | median count KL | median exact MMD | median labeled-state TV | mode = x*'s mode | median abs(x - x*) (kT) | median time (s) | median samples (M) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle (x*) | 30 | 100% | 100% / 100% | 0.002 | 0.002 | 0.0000 | 0.00e+00 | 0.002 | 100% | 0.00 | 0 | 0.00 |
| GBFESO-x0 | 30 | 43% | 46% / 33% | 0.059 | 0.066 | 0.0178 | 6.65e-02 | 0.163 | 43% | 4.39 | 30 | 0.71 |
| GBFESO-rand4 | 30 | 50% | 50% / 50% | 0.051 | 0.079 | 0.0354 | 9.22e-02 | 0.163 | 57% | 7.05 | 111 | 2.84 |
| MLGD-KL | 30 | 33% | 33% / 33% | 0.084 | 0.090 | 0.0316 | 1.56e-01 | 0.299 | 30% | 7.70 | 595 | 3.31 |
| MLGD-MMD | 30 | 33% | 33% / 33% | 0.073 | 0.074 | 0.0309 | 5.94e-02 | 0.276 | 40% | 6.50 | 614 | 3.31 |
| MLGD-KL+polish | 30 | 40% | 42% / 33% | 0.088 | 0.096 | 0.0441 | 1.25e-01 | 0.290 | 37% | 8.03 | 678 | 5.44 |
| MLGD-MMD+polish | 30 | 37% | 38% / 33% | 0.064 | 0.085 | 0.0281 | 8.86e-02 | 0.346 | 30% | 7.16 | 701 | 5.44 |

Per target mode (residue carrying the most '+' at x*): solved share, and among solved runs the share that found x*'s mode / another residue's mode

| method | x* mode res1 (n) | x* mode res2 (n) | x* mode res3 (n) |
|---|---|---|---|
| oracle (x*) | 100% solved, 100% same mode / 0% other (n=18) | 100% solved, 100% same mode / 0% other (n=10) | 100% solved, 100% same mode / 0% other (n=2) |
| GBFESO-x0 | 44% solved, 62% same mode / 38% other (n=18) | 50% solved, 20% same mode / 80% other (n=10) | 0% solved, nan% same mode / nan% other (n=2) |
| GBFESO-rand4 | 56% solved, 90% same mode / 10% other (n=18) | 50% solved, 20% same mode / 80% other (n=10) | 0% solved, nan% same mode / nan% other (n=2) |
| MLGD-KL | 28% solved, 20% same mode / 80% other (n=18) | 50% solved, 20% same mode / 80% other (n=10) | 0% solved, nan% same mode / nan% other (n=2) |
| MLGD-MMD | 33% solved, 67% same mode / 33% other (n=18) | 40% solved, 0% same mode / 100% other (n=10) | 0% solved, nan% same mode / nan% other (n=2) |
| MLGD-KL+polish | 39% solved, 57% same mode / 43% other (n=18) | 40% solved, 25% same mode / 75% other (n=10) | 50% solved, 0% same mode / 100% other (n=2) |
| MLGD-MMD+polish | 39% solved, 57% same mode / 43% other (n=18) | 40% solved, 0% same mode / 100% other (n=10) | 0% solved, nan% same mode / nan% other (n=2) |
