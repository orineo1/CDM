Reweighted (MBAR) landscape runs/rugged2/c1/landscape.json: design {'n_beads': 4, 'free': ['charge1', 'charge2', 'charge3', 'charge4'], 'fixed': {'eps1': 0.5, 'eps2': 0.5, 'eps3': 0.5}, 'box': {}, 'prior': 'uniform', 'elec': {'A': 3.0, 'debye': 2.0, 'rcut': 6.0}}

MBAR: K = 256 pool designs x 2000 frames, 135 iterations (last change 8.9e-08). Loss = MMD^2 (U-statistic), bandwidths 0.86, 1.71, 3.43, 6.86.

| target | floor | L(q*) MBAR | ESS(q*) | basin TV(q*) MBAR | median L random | best random / L(q*) | starts solved | local minima (>= 2 starts) | spurious | best spurious L / L(q*) | its barrier (MBAR line) | perturbations higher |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -6.8e-05 | 1.5e-04 | 268100 | 0.013 | 5.3e-01 | 89.3 | 1.00 | 1 | 0 | - | - | - |
| 1 | -5.0e-05 | 5.1e-04 | 165746 | 0.008 | 6.2e-01 | 39.8 | 1.00 | 2 | 0 | - | - | - |
| 2 | 8.9e-05 | 3.8e-04 | 243751 | 0.008 | 5.1e-01 | 9.4 | 1.00 | 4 | 0 | - | - | - |
| 3 | -2.2e-04 | 1.5e-03 | 76650 | 0.006 | 1.0e+00 | 7.2 | 1.00 | 3 | 0 | - | - | - |
| 4 | -1.4e-04 | 9.6e-04 | 69061 | 0.003 | 9.1e-01 | 3.7 | 1.00 | 3 | 0 | - | - | - |
| 5 | 1.7e-04 | 2.3e-04 | 122647 | 0.011 | 5.3e-01 | 32.4 | 1.00 | 3 | 0 | - | - | - |
| 6 | -7.2e-05 | 6.1e-04 | 297109 | 0.004 | 5.3e-01 | 9.5 | 1.00 | 1 | 0 | - | - | - |
| 7 | -9.2e-05 | 3.5e-04 | 13130 | 0.019 | 1.6e+00 | 41.0 | 0.85 | 3 | 2 | 3662.5 (n=16) | 1.1e-02 | 0.98 |

Mean fraction of descent starts that reach the global basin: 0.98 (targets with >= 1 spurious minimum: 1/8).

Direct simulation (no reweighting) along the line spurious minimum (lam 0) -> q* (lam 1): MMD^2 / energy distance / basin TV

| target | spurious # | lam: MMD^2 | max interior - endpoint (MMD^2) | TV at lam 0 | TV at lam 1 |
|---|---|---|---|---|---|
| 7 | 0 | 1.3e+00 1.3e+00 1.3e+00 1.2e+00 1.1e+00 9.7e-01 7.7e-01 5.0e-01 2.1e-01 4.0e-02 1.3e-05 | 8.2e-03 | 0.864 | 0.007 |
| 7 | 1 | 1.3e+00 1.3e+00 1.3e+00 1.3e+00 1.2e+00 1.1e+00 8.8e-01 5.9e-01 2.6e-01 4.8e-02 -4.8e-07 | 1.0e-02 | 0.868 | 0.004 |
