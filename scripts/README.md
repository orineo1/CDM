# `scripts/`

Command-line entry points. Every `.py` here has a `main` and `--help`; nothing here is
imported as a library by the packages.

## Pipeline

| Script | What it does |
| --- | --- |
| `generate_data.py` | Step 1: sample `P(X,Y)` from a mixture, define the target `G(Y)`, save a `.pt`. |
| `train_models.py` | Step 2: train the prior `P(X)` and the conditional sampler `P(Y\|X)` (diffusion / cm / fm). |
| `run_optimization.py` | Step 3: load the trained models, search for `x_hat`, evaluate. |
| `run_experiment.py` | Convenience wrapper running steps 1-3 in sequence. |

## Method

| Script | What it does |
| --- | --- |
| `find_xstar.py` | The paper's method as one command: prior + conditional sampler + target `G(Y)` -> `x*`. |

## Paper artifacts

| Script | What it does |
| --- | --- |
| `fig1_toy_example.py` | Figure 1 (toy example): simulate (cached) then plot. |
| `fig_smc_correction.py` | Twisted-SMC bias correction on the Figure 1 setting. |
| `make_table1.py` | Table 1 end to end over all (dim X, dim Y) scenarios. |
| `check_models.py` | Checks the trained networks against the analytic mixture, independently of the `x*` search. |

## Controls

| Script | What it does |
| --- | --- |
| `control_subsample.py` | Appendix G control: gradient subsampling (Uniform / Witness) vs full backprop. |


## Infra

| File | What it does |
| --- | --- |
| `run_all_tests.sh` | Runs the test suite (`pytest tests/`). |
| `slurm_fig1.sh` | Figure 1 on the cluster, all settings, with restarts. |
| `slurm_table1.sh` | Table 1 on the cluster: one array task per scenario, then a collect job. |
| `cluster_setup.txt` | Cluster setup instructions. |

## A note on imports

The repo has no `pyproject.toml` / `setup.py`, so scripts put the repo root on the path
themselves:

```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
```

This boilerplate (and the `# noqa: E402` it forces on every import below it) is the
consequence of there being no package install step. Run scripts as
`python scripts/<name>.py` from the repo root.
