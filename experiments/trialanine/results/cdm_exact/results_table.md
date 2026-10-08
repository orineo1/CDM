| method | targets | solved (count TV <= 0.05) | solved (TV <= 0.02) | solved, random / engineered | median count TV | mean count TV | median count KL | median exact MMD | median labeled-state TV | mode = x*'s mode | median abs(x - x*) (kT) | median time (s) | median samples (M) | median loss evals | ESS < 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle (x*) | 30 | 100% | 100% | 100% / 100% | 0.002 | 0.002 | 0.0000 | 0.00e+00 | 0.002 | 100% | 0.00 | 0 | 0.00 | 0 | 0 |
| GBFESO-exact-x0 | 30 | 100% | 100% | 100% / 100% | 0.002 | 0.002 | 0.0000 | 1.06e-01 | 0.193 | 47% | 4.93 | 7 | 0.00 | 202 | 0 |
| GBFESO-exact-rand4 | 30 | 100% | 100% | 100% / 100% | 0.002 | 0.003 | 0.0000 | 8.09e-02 | 0.092 | 60% | 6.40 | 28 | 0.00 | 808 | 0 |
| GBFESO-exact-randEq | 30 | 100% | 100% | 100% / 100% | 0.002 | 0.002 | 0.0000 | 1.34e-01 | 0.169 | 50% | 7.34 | 138 | 0.00 | 4040 | 0 |
| MLGD-exact | 30 | 100% | 100% | 100% / 100% | 0.001 | 0.002 | 0.0000 | 1.18e-01 | 0.139 | 47% | 7.69 | 157 | 0.00 | 3995 | 2 |
| MLGD-exact+polish | 30 | 100% | 100% | 100% / 100% | 0.001 | 0.002 | 0.0000 | 1.18e-01 | 0.139 | 47% | 7.69 | 178 | 0.00 | 4601 | 2 |

Per target mode (residue carrying the most '+' at x*): solved share, and among solved runs the share that found x*'s mode / another residue's mode

| method | x* mode res1 (n) | x* mode res2 (n) | x* mode res3 (n) |
|---|---|---|---|
| oracle (x*) | 100% solved, 100% same mode / 0% other (n=18) | 100% solved, 100% same mode / 0% other (n=10) | 100% solved, 100% same mode / 0% other (n=2) |
| GBFESO-exact-x0 | 100% solved, 56% same mode / 44% other (n=18) | 100% solved, 40% same mode / 60% other (n=10) | 100% solved, 0% same mode / 100% other (n=2) |
| GBFESO-exact-rand4 | 100% solved, 56% same mode / 44% other (n=18) | 100% solved, 70% same mode / 30% other (n=10) | 100% solved, 50% same mode / 50% other (n=2) |
| GBFESO-exact-randEq | 100% solved, 67% same mode / 33% other (n=18) | 100% solved, 20% same mode / 80% other (n=10) | 100% solved, 50% same mode / 50% other (n=2) |
| MLGD-exact | 100% solved, 67% same mode / 33% other (n=18) | 100% solved, 20% same mode / 80% other (n=10) | 100% solved, 0% same mode / 100% other (n=2) |
| MLGD-exact+polish | 100% solved, 67% same mode / 33% other (n=18) | 100% solved, 20% same mode / 80% other (n=10) | 100% solved, 0% same mode / 100% other (n=2) |
