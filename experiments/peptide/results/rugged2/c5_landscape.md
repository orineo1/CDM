Reweighted (MBAR) landscape runs/rugged2/c5/landscape.json: design {'n_beads': 4, 'free': ['sigma1', 'sigma2', 'sigma3', 'sigma4'], 'fixed': {'eps1': 1.0, 'eps2': 1.0, 'eps3': 1.0}, 'box': {'sigma': [0.7, 1.5]}, 'prior': 'uniform'}

MBAR: K = 512 pool designs x 2000 frames, 869 iterations (last change 9.9e-08). Loss = MMD^2 (U-statistic), bandwidths 0.91, 1.81, 3.63, 7.26.

| target | floor | L(q*) MBAR | ESS(q*) | basin TV(q*) MBAR | median L random | best random / L(q*) | starts solved | local minima (>= 2 starts) | spurious | best spurious L / L(q*) | its barrier (MBAR line) | perturbations higher |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.2e-04 | 1.3e-02 | 4339 | 0.051 | 3.0e-01 | 1.8 | 0.96 | 2 | 1 | 30.2 (n=2) | 1.1e-01 | 1.00 |
| 1 | -8.9e-05 | 1.3e-02 | 3732 | 0.021 | 4.3e-01 | -0.1 | 0.95 | 3 | 1 | 30.2 (n=2) | 6.9e-02 | 1.00 |
| 2 | 4.2e-04 | 4.1e-02 | 206 | 0.096 | 4.5e-01 | 0.1 | 0.75 | 3 | 2 | 1.6 (n=25) | 2.8e-02 | 1.00 |
| 3 | -4.4e-05 | 1.1e-01 | 3187 | 0.171 | 4.1e-01 | 0.4 | 0.98 | 4 | 0 | - | - | - |
| 4 | -1.9e-04 | 2.4e-02 | 2392 | 0.127 | 4.2e-01 | 1.5 | 0.94 | 2 | 1 | 15.2 (n=2) | 9.0e-02 | 1.00 |
| 5 | -2.2e-04 | 3.4e-02 | 263 | 0.142 | 4.4e-01 | 0.6 | 0.97 | 4 | 1 | 1.8 (n=2) | 2.3e-02 | 1.00 |
| 6 | -2.1e-04 | 1.2e-02 | 6432 | 0.082 | 3.3e-01 | 2.1 | 0.91 | 5 | 3 | 4.7 (n=3) | 1.0e-01 | 1.00 |
| 7 | -1.6e-04 | 1.7e-02 | 269 | 0.099 | 6.5e-01 | -5.8 | 0.97 | 5 | 0 | - | - | - |

Mean fraction of descent starts that reach the global basin: 0.93 (targets with >= 1 spurious minimum: 6/8).

Direct simulation (no reweighting) along the line spurious minimum (lam 0) -> q* (lam 1): MMD^2 / energy distance / basin TV

| target | spurious # | lam: MMD^2 | max interior - endpoint (MMD^2) | TV at lam 0 | TV at lam 1 |
|---|---|---|---|---|---|
| 0 | 0 | 5.7e-01 4.8e-01 4.0e-01 3.3e-01 2.7e-01 2.1e-01 1.7e-01 1.0e-01 7.4e-02 2.5e-02 2.1e-02 | -9.4e-02 | 0.179 | 0.112 |
| 1 | 0 | 6.3e-01 5.2e-01 5.4e-01 3.5e-01 3.2e-01 2.5e-01 1.8e-01 1.5e-01 5.8e-02 6.2e-02 3.1e-02 | -9.3e-02 | 0.193 | 0.089 |
| 2 | 0 | 6.8e-02 8.0e-02 6.8e-02 3.5e-02 3.0e-02 3.6e-02 2.7e-02 1.2e-02 3.8e-02 5.8e-02 7.0e-02 | 1.2e-02 | 0.155 | 0.100 |
| 2 | 1 | 1.1e-01 6.2e-02 5.8e-02 4.0e-02 7.8e-02 8.0e-02 5.1e-02 2.6e-02 2.2e-02 4.0e-02 4.6e-02 | -3.2e-02 | 0.087 | 0.085 |
| 4 | 0 | 5.6e-01 4.2e-01 3.7e-01 2.9e-01 2.5e-01 1.9e-01 1.3e-01 1.2e-01 9.2e-02 1.3e-02 1.5e-02 | -1.4e-01 | 0.237 | 0.065 |
| 5 | 0 | 1.5e-01 1.4e-01 1.0e-01 1.1e-01 7.9e-02 7.1e-02 5.7e-02 5.0e-02 3.9e-02 4.8e-02 2.7e-02 | -1.3e-02 | 0.117 | 0.056 |
| 6 | 0 | 2.9e-01 2.5e-01 2.2e-01 1.6e-01 1.3e-01 1.2e-01 1.1e-01 4.7e-02 7.3e-02 4.9e-02 2.7e-02 | -3.5e-02 | 0.152 | 0.051 |
| 6 | 1 | 5.0e-01 4.7e-01 3.5e-01 2.8e-01 1.9e-01 1.4e-01 8.6e-02 4.7e-02 3.8e-02 6.1e-02 3.6e-02 | -3.0e-02 | 0.362 | 0.151 |
