# GlobalConditional

Code for **Conditional Distribution Matching (CDM)**: given a frozen conditional
generator `P(Y|X)` and a target distribution `G(Y)`, find an input `x*` whose
induced conditional `P(Y|X=x*)` matches `G`. The solver is loss-guided diffusion
over `X` with a plug-in distributional loss (MMD or SWD) estimated from samples
of a fast conditional sampler (MLGD-F in the paper).

Everything is run from the repository root on the command line; every script
prints `--help`, auto-detects a GPU and falls back to CPU (`--device` overrides).

## Install

```bash
git clone https://github.com/orineo1/GlobalConditional.git
cd GlobalConditional
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU is enough for the synthetic examples
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick start: a synthetic Mixture-of-Gaussians run

One command draws a joint `P(X,Y)`, trains the prior on `X` and the conditional
sampler on `Y|X`, runs the guided optimisation and evaluates the recovered
`x_hat` against the known optimum:

```bash
python scripts/run_experiment.py --random-mog --dim 4 --n-components 3 --condition-on 3 \
    --n-samples 10000 --epochs 2000 --method LGD_CM --n-attempts 5 --work-dir runs/mog4d
```

`--condition-on` is `dim(X)`; the remaining coordinates are `Y`. `--method` is
`LGD_CM` (MLGD-F, consistency-model sampler), `LGD` (MLGD, diffusion sampler;
add `--architecture diffusion`) or `DFLOW`. Outputs go to `--work-dir`:
`data.pt`, `models/`, `results.pt`.

The same run as three separate steps:

```bash
# 1. data + target: draws from P(X,Y), and G(Y) = P(Y | X = x*)
python scripts/generate_data.py --random-mog --dim 4 --n-components 3 --condition-on 3 \
    --n-samples 10000 --output runs/mog4d/data.pt
#    or one of the paper's fixed settings:
python scripts/generate_data.py --params-file SimulationParameters/mog_2d_full.txt --output runs/mog2d/data.pt

# 2. models: unconditional diffusion on X, conditional sampler on Y|X
python scripts/train_models.py --data runs/mog4d/data.pt --model both --architecture cm \
    --epochs 2000 --batch-size 1024 --output-dir runs/mog4d/models

# 3. guided optimisation + evaluation
python scripts/run_optimization.py --data runs/mog4d/data.pt --model-dir runs/mog4d/models \
    --method LGD_CM --n-attempts 25 --optim-samples 250 --output runs/mog4d/results.pt
```

## Paper figures

| Figure | Command |
|---|---|
| Fig. 1, toy example: joint, induced conditional vs. target, CDMS beta-sweep | `python scripts/fig1_toy_example.py` |
| App. G subsampling control on the 10D setting | `python scripts/control_subsample.py --sim-dir /path/to/conditional-matching-paper/simulations --n-restarts 25` |

`fig1_toy_example.py` trains its three small networks on first use (cached under
`checkpoints/toy/<setting>/`), writes the simulation to `results/fig1_<setting>_sim.json`
and `_sweep.npz`, and draws the figure from those files. Re-running only redraws;
`--rerun-sim` recomputes. `--setting {bimodal,paper,mog2d}` picks the joint,
`--quick` is a one-minute smoke test, `--uncond/--cond/--cm {auto,train,load}`
control each network. CPU: about 5 minutes with cached models, 9 with training.
`control_subsample.py` needs the paper repository's 10D checkpoints, downloaded
from HuggingFace on first use.

Scripts for the remaining figures (2D/5D/10D tables, MNIST, Stable Diffusion)
will follow the same pattern.

## Using the optimizer from Python

```python
from guidance import GuidanceConfig, optimize_guided
out = optimize_guided(model_uncond, model_cond, loss_fn, GuidanceConfig(...))
# out: x_final, final_loss, best_loss, best_x0, n_steps
```

`GuidanceConfig` holds every knob as an explicit strategy: initialisation,
Tweedie perturbation, aggregation over Monte-Carlo perturbations, step-size rule,
DDIM step style and stochasticity (`eta`), target sampling, and gradient
subsampling through `cond_output_transform`. The original function-per-variant
entry points (`optimize_LGD`, `optimize_DPS`, `optimize_LGD_CM`, `optimize_LGD_cfg`,
`optimize_DFLOW`, `optimize_LGD_latent`) live in `legacy/Optimization.py` with
their original signatures.

## Layout

```
guidance/       the guided-diffusion optimizer: core.py (optimize_guided), config.py (GuidanceConfig)
priors/         unconditional diffusion models over X
samplers/       conditional samplers for P(Y|X): consistency models, flow matching
losses/         MMD/RBF, circular, sliced Wasserstein
utils/          networks, mixtures and the exact L2 between GMMs, evaluation, superposition
gcio/           run persistence (results/runs.jsonl)
legacy/         function-per-variant entry points and the frozen pre-refactor reference
scripts/        all command-line entry points, test runners, cluster templates
tests/          smoke, equivalence, determinism
notebooks/      exploratory notebooks (not maintained; the scripts supersede them)
SimulationParameters/   the paper's fixed MoG settings
MnistModels/    a small pretrained MNIST checkpoint
```

## Tests

```bash
bash scripts/run_all_tests.sh                               # smoke, equivalence, determinism
python scripts/run_regression.py --legacy --update-golden   # baseline against published numbers, once
python scripts/run_regression.py                            # check against it
```

`tests/test_equivalence.py` checks each refactored optimizer against the frozen
reference in `legacy/Optimization_legacy.py` on fixed seeds. Known open issue:
`test_optimize_DPS_equivalence` fails inside the reference DPS code itself, which
unpacks two values from a sampler that returns three.

## Results

Runs call `gcio.results.save_run(...)` and tables are built from the store,
`results/runs.jsonl`.

## Cluster

`scripts/cluster_setup.txt` has SLURM and PBS templates for the same commands.
