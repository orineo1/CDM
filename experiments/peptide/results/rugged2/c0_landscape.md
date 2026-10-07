Reweighted (MBAR) landscape runs/rugged2/c0/landscape.json: design {'n_beads': 4, 'free': ['charge1', 'charge2', 'charge3', 'charge4'], 'fixed': {'eps1': 0.5, 'eps2': 0.5, 'eps3': 0.5}, 'box': {}, 'prior': 'uniform', 'elec': {'A': 1.0, 'debye': 2.0, 'rcut': 6.0}}

MBAR: K = 256 pool designs x 2000 frames, 19 iterations (last change 9.6e-08). Loss = MMD^2 (U-statistic), bandwidths 0.91, 1.82, 3.64, 7.28.

| target | floor | L(q*) MBAR | ESS(q*) | basin TV(q*) MBAR | median L random | best random / L(q*) | starts solved | local minima (>= 2 starts) | spurious | best spurious L / L(q*) | its barrier (MBAR line) | perturbations higher |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -1.7e-04 | 4.1e-04 | 472897 | 0.013 | 1.4e-01 | 4.9 | 1.00 | 3 | 0 | - | - | - |
| 1 | -5.9e-05 | 3.0e-04 | 382274 | 0.006 | 1.5e-01 | 15.0 | 0.98 | 2 | 0 | - | - | - |
| 2 | -1.6e-04 | 4.9e-04 | 454919 | 0.009 | 1.2e-01 | 2.0 | 0.99 | 3 | 0 | - | - | - |
| 3 | 6.2e-05 | 1.7e-04 | 317830 | 0.006 | 1.9e-01 | 10.0 | 1.00 | 2 | 0 | - | - | - |
| 4 | -1.8e-04 | 5.6e-04 | 212776 | 0.012 | 2.7e-01 | 4.3 | 1.00 | 2 | 0 | - | - | - |
| 5 | 2.4e-04 | 2.4e-04 | 331179 | 0.003 | 1.4e-01 | 9.7 | 1.00 | 6 | 0 | - | - | - |
| 6 | 1.6e-04 | 3.0e-04 | 479542 | 0.003 | 1.4e-01 | 8.9 | 1.00 | 3 | 0 | - | - | - |
| 7 | 7.1e-05 | 4.8e-05 | 194417 | 0.009 | 1.7e-01 | 100.9 | 0.96 | 2 | 1 | 1173.7 (n=4) | 5.5e-03 | 1.00 |

Mean fraction of descent starts that reach the global basin: 0.99 (targets with >= 1 spurious minimum: 1/8).

Direct simulation (no reweighting) along the line spurious minimum (lam 0) -> q* (lam 1): MMD^2 / energy distance / basin TV

| target | spurious # | lam: MMD^2 | max interior - endpoint (MMD^2) | TV at lam 0 | TV at lam 1 |
|---|---|---|---|---|---|
| 7 | 0 | 5.9e-02 6.3e-02 6.4e-02 6.4e-02 5.7e-02 5.3e-02 4.4e-02 3.3e-02 1.8e-02 5.1e-03 1.6e-04 | 4.7e-03 | 0.102 | 0.012 |
