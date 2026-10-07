Reweighted (MBAR) landscape runs/rugged2/c4/landscape.json: design {'n_beads': 4, 'free': ['sigma1', 'sigma2', 'sigma3', 'sigma4', 'charge1', 'charge2', 'charge3', 'charge4'], 'fixed': {'eps1': 0.5, 'eps2': 0.5, 'eps3': 0.5}, 'box': {'sigma': [0.7, 1.5]}, 'prior': 'uniform', 'elec': {'A': 3.0, 'debye': 2.0, 'rcut': 6.0}}

MBAR: K = 512 pool designs x 2000 frames, 340 iterations (last change 9.7e-08). Loss = MMD^2 (U-statistic), bandwidths 0.90, 1.80, 3.60, 7.20.

| target | floor | L(q*) MBAR | ESS(q*) | basin TV(q*) MBAR | median L random | best random / L(q*) | starts solved | local minima (>= 2 starts) | spurious | best spurious L / L(q*) | its barrier (MBAR line) | perturbations higher |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -1.2e-04 | 3.1e-03 | 5821 | 0.017 | 6.1e-01 | 29.9 | 0.95 | 3 | 0 | - | - | - |
| 1 | -7.1e-05 | 5.1e-03 | 1398 | 0.007 | 1.0e+00 | 18.9 | 0.96 | 10 | 0 | - | - | - |
| 2 | -1.6e-04 | 2.7e-03 | 6129 | 0.036 | 6.2e-01 | 25.7 | 0.97 | 5 | 0 | - | - | - |
| 3 | 6.5e-05 | 2.5e-05 | 7805 | 0.013 | 1.5e+00 | 3029.4 | 0.75 | 11 | 1 | 16853.0 (n=32) | 3.8e-01 | 1.00 |
| 4 | -7.3e-05 | 3.3e-03 | 4818 | 0.009 | 8.0e-01 | 10.2 | 0.99 | 2 | 0 | - | - | - |
| 5 | 1.7e-04 | -1.2e-04 | 4678 | 0.014 | 5.8e-01 | 61291933.1 | 0.95 | 2 | 1 | 71822285.7 (n=2) | 7.2e-02 | 1.00 |
| 6 | -7.6e-05 | 1.4e-04 | 4842 | 0.011 | 1.2e+00 | 559.1 | 0.84 | 7 | 1 | 3434.8 (n=17) | 2.5e-01 | 1.00 |
| 7 | -2.2e-04 | 9.4e-04 | 4261 | 0.006 | 7.9e-01 | 18.5 | 0.97 | 5 | 1 | 341.4 (n=2) | 4.8e-01 | 1.00 |

Mean fraction of descent starts that reach the global basin: 0.92 (targets with >= 1 spurious minimum: 4/8).

Direct simulation (no reweighting) along the line spurious minimum (lam 0) -> q* (lam 1): MMD^2 / energy distance / basin TV

| target | spurious # | lam: MMD^2 | max interior - endpoint (MMD^2) | TV at lam 0 | TV at lam 1 |
|---|---|---|---|---|---|
| 3 | 0 | 7.7e-01 7.4e-01 7.6e-01 7.7e-01 7.8e-01 7.5e-01 6.6e-01 4.3e-01 1.6e-01 2.7e-02 1.0e-04 | 5.4e-03 | 0.937 | 0.007 |
| 5 | 0 | 7.3e-02 7.6e-02 1.0e-01 1.2e-01 1.4e-01 1.4e-01 1.3e-01 1.1e-01 7.7e-02 2.7e-02 1.6e-04 | 6.9e-02 | 0.203 | 0.012 |
| 6 | 0 | 5.4e-01 6.5e-01 7.1e-01 7.2e-01 6.5e-01 5.6e-01 4.3e-01 2.6e-01 1.2e-01 3.2e-02 5.9e-04 | 1.8e-01 | 0.396 | 0.008 |
| 7 | 0 | 5.8e-01 6.1e-01 6.5e-01 7.5e-01 7.8e-01 6.5e-01 4.1e-01 1.9e-01 6.0e-02 1.3e-02 2.8e-03 | 2.1e-01 | 0.943 | 0.004 |
