| method | targets | solved (D <= 0.02) | solved (D <= 0.05) | global basin found | median D | median loss | median abs(x - x*) (kT) | median loss evals | median time (s) | ESS < 500 |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle (x*) | 24 | 100% | 100% | 100% | 0.001 | 0.0000 | 0.00 | 0 | 0 | 0 |
| GBFESO-exact-x0 | 24 | 42% | 75% | 96% | 0.027 | 0.0001 | 6.43 | 202 | 7 | 0 |
| GBFESO-exact-rand4 | 24 | 71% | 83% | 100% | 0.008 | 0.0000 | 6.31 | 808 | 23 | 0 |
| GBFESO-exact-randEq | 24 | 96% | 100% | 100% | 0.002 | 0.0000 | 6.73 | 4040 | 112 | 0 |
| MLGD-exact | 24 | 96% | 100% | 100% | 0.001 | 0.0000 | 7.27 | 3995 | 134 | 0 |
| MLGD-exact+polish | 24 | 100% | 100% | 100% | 0.001 | 0.0000 | 7.13 | 4601 | 152 | 0 |
