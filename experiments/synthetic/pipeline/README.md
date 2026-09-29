# `experiments/synthetic/pipeline/`

The command-line entry points of the synthetic experiments. Every `.py` here has a
`main` and `--help`; nothing here is imported by the `cdm` package.

## Pipeline

| Script | What it does |
| --- | --- |
| `generate_data.py` | Step 1: sample `P(X,Y)` from a mixture, define the target `G(Y)`, save a `.pt`. |
| `train_models.py` | Step 2: train the prior `P(X)` and the conditional sampler `P(Y\|X)` (diffusion / cm). |
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
| `../../../tests/run_all_tests.sh` | Runs the test suite (`pytest tests/`). |
| `../slurm/slurm_fig1.sh` | Figure 1 on the cluster, all settings, with restarts. |
| `../slurm/slurm_table1.sh` | Table 1 on the cluster: one array task per scenario, then a collect job. |
| `../slurm/cluster_setup.txt` | Cluster setup instructions. |

## Imports

The scripts import the method from the installed `cdm` package (`pip install -e .`)
and each other as siblings, so they are run as
`python experiments/synthetic/pipeline/<name>.py` from the repo root.
