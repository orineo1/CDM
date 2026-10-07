Reweighted (MBAR) landscape runs/rugged2/c2/landscape.json: design {'n_beads': 4, 'free': ['charge1', 'charge2', 'charge3', 'charge4'], 'fixed': {'eps1': 0.5, 'eps2': 0.5, 'eps3': 0.5}, 'box': {}, 'prior': 'uniform', 'elec': {'A': 3.0, 'debye': 1.0, 'rcut': 6.0}}

MBAR: K = 256 pool designs x 2000 frames, 70 iterations (last change 9.1e-08). Loss = MMD^2 (U-statistic), bandwidths 0.88, 1.77, 3.53, 7.06.

| target | floor | L(q*) MBAR | ESS(q*) | basin TV(q*) MBAR | median L random | best random / L(q*) | starts solved | local minima (>= 2 starts) | spurious | best spurious L / L(q*) | its barrier (MBAR line) | perturbations higher |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -1.7e-04 | 1.5e-04 | 361393 | 0.009 | 3.7e-01 | 42.9 | 1.00 | 1 | 0 | - | - | - |
| 1 | 1.1e-04 | 7.9e-04 | 236320 | 0.010 | 4.4e-01 | 19.2 | 1.00 | 2 | 0 | - | - | - |
| 2 | 2.8e-04 | 1.2e-03 | 330316 | 0.024 | 3.5e-01 | 3.1 | 1.00 | 5 | 0 | - | - | - |
| 3 | 1.7e-04 | 7.9e-06 | 132546 | 0.003 | 6.5e-01 | 351.9 | 1.00 | 2 | 0 | - | - | - |
| 4 | -1.7e-04 | 2.1e-03 | 96757 | 0.003 | 6.8e-01 | 1.7 | 1.00 | 2 | 0 | - | - | - |
| 5 | 8.4e-05 | 1.5e-04 | 186609 | 0.009 | 3.7e-01 | 25.6 | 1.00 | 4 | 0 | - | - | - |
| 6 | -6.3e-05 | 4.1e-04 | 382578 | 0.004 | 3.4e-01 | 10.2 | 0.99 | 2 | 0 | - | - | - |
| 7 | -3.9e-05 | 5.3e-05 | 27418 | 0.012 | 6.6e-01 | 165.0 | 0.95 | 2 | 1 | 7501.7 (n=7) | 1.5e-02 | 1.00 |

Mean fraction of descent starts that reach the global basin: 0.99 (targets with >= 1 spurious minimum: 1/8).

Direct simulation (no reweighting) along the line spurious minimum (lam 0) -> q* (lam 1): MMD^2 / energy distance / basin TV

| target | spurious # | lam: MMD^2 | max interior - endpoint (MMD^2) | TV at lam 0 | TV at lam 1 |
|---|---|---|---|---|---|
| 7 | 0 | 4.0e-01 4.0e-01 4.1e-01 4.1e-01 3.8e-01 3.3e-01 2.7e-01 1.9e-01 1.1e-01 3.4e-02 4.6e-05 | 9.4e-03 | 0.436 | 0.009 |
