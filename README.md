# GlobalConditional

Code for **Conditional Distribution Matching (CDM)**: given a frozen conditional
generator `P(Y|X)` and a target distribution `G(Y)`, find an input `x*` whose
induced conditional `P(Y|X=x*)` matches `G`. The solver is loss-guided diffusion
over `X` with a plug-in distributional loss (MMD or SWD) estimated from samples
of a fast conditional sampler (MLGD-F in the paper).

The method is the `cdm` package; `experiments/` holds one directory per paper
experiment (synthetic Gaussian mixtures, rotated MNIST, Stable Diffusion), each
with the exact command, its requirements and the published reference numbers.
Everything is run from the repository root on the command line; every script
prints `--help`, auto-detects a GPU and falls back to CPU (`--device` overrides).

## Install

```bash
git clone https://github.com/orineo1/GlobalConditional.git
cd GlobalConditional
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU is enough for the synthetic examples
pip install -e .                           # makes `import cdm` work from anywhere
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121
# Stable Diffusion experiment only: pip install -r experiments/stable_diffusion/requirements.txt
```

## Finding x*: the main script

`experiments/synthetic/pipeline/find_xstar.py` is the method. It takes a prior over `X`, a conditional
sampler for `Y|X` and a target `G(Y)`, runs guided reverse diffusion with the
distributional loss as guidance, as independent restarts or as a resampled particle
population, and returns the best `x_hat`:

```bash
python experiments/synthetic/pipeline/find_xstar.py \
    --prior   runs/mog4d/models/model_unconditional.pt \
    --sampler runs/mog4d/models/model_conditional.pt \
    --target  runs/mog4d/data.pt \
    --n-attempts 10 --out runs/mog4d/xstar.json
```

The prior and sampler are checkpoints written by `experiments/synthetic/pipeline/train_models.py` (see
the next section for producing them); the target is either a `generate_data.py`
file, which describes `G` as a mixture and also carries `x*` for evaluation, or a
`.pt`/`.npy` array of target samples. The algorithm is chosen on the command line:

| Flag | Meaning | Default |
|---|---|---|
| `--sampler-type cm\|diffusion\|auto` | few-step sampler (MLGD-F) or full conditional diffusion (MLGD) | auto, from the checkpoint |
| `--guidance lgd\|dps` | LGD: Monte-Carlo perturbations of the Tweedie point, log-sum-exp aggregate, constant step; DPS: single Tweedie point, step `zeta' * beta_t` | lgd |
| `--n-cond`, `--n-target` | conditional / target samples per step | 250 / n_cond |
| `--loss mmd\|swd`, `--rbf-kernels`, `--rbf-mul`, `--swd-projections` | the distributional distance | mmd, 5 bandwidths |
| `--select all\|uniform`, `--n-sel` | backpropagate through all samples or a random subset | all |
| `--n-mc`, `--perturb`, `--aggregate` | the outer-loop Monte-Carlo estimator | preset by `--guidance` |
| `--step-size`, `--zeta`, `--eta`, `--guidance-cutoff-t`, `--init` | step rule, DDIM stochasticity, initialisation | 1.0, 1.0, 0, none, randn |
| `--algo lgd\|particles` | independent restarts (best final loss wins), or one population of `--n-particles` trajectories run in lockstep and resampled by `exp(-beta * loss)` whenever the effective sample size drops below `--ess-frac` of the population (`--resample-beta`, `--resample-every`, `--weight-mode`) | lgd |
| `--refine-steps`, `--refine-lr`, `--refine-lambda` | post-refinement for either algorithm: plain gradient steps on the plug-in loss from the returned `x_hat`, no diffusion, with an optional prior-score term | 0 |
| `--n-attempts`, `--seed`, `--device`, `--final-n-cond` | restarts (lgd) or populations (particles) and reporting | 10 / 1, 42, auto, 4 n_cond |

Every attempt prints its `x_hat`, final loss and, when `x*` is known, the
distance to it and the exact L2 distance between `P(Y|X=x_hat)` and `G`; the
JSON holds all of them and the best.

## Producing a prior, a sampler and a target: the synthetic pipeline

One command draws a joint `P(X,Y)`, trains the prior on `X` and the conditional
sampler on `Y|X`, runs the guided optimisation and evaluates the recovered
`x_hat` against the known optimum (a few minutes on CPU, most of it training):

```bash
python experiments/synthetic/pipeline/run_experiment.py --random-mog --dim 4 --n-components 3 --condition-on 3 \
    --n-samples 10000 --epochs 2000 --method LGD_CM --n-attempts 5 --work-dir runs/mog4d
```

`--condition-on` is `dim(X)`; the remaining coordinates are `Y`. `--method` is
`LGD_CM` (MLGD-F, consistency-model sampler) or `LGD` (MLGD, diffusion sampler;
add `--architecture diffusion`). Outputs go to `--work-dir`:
`data.pt`, `models/`, `results.pt`.

The same run as three separate steps:

```bash
# 1. data + target: draws from P(X,Y), and G(Y) = P(Y | X = x*)
python experiments/synthetic/pipeline/generate_data.py --random-mog --dim 4 --n-components 3 --condition-on 3 \
    --n-samples 10000 --output runs/mog4d/data.pt
#    or one of the paper's fixed settings:
python experiments/synthetic/pipeline/generate_data.py --params-file experiments/synthetic/configs/mog/mog_2d_full.txt --output runs/mog2d/data.pt

# 2. models: unconditional diffusion on X, conditional sampler on Y|X
python experiments/synthetic/pipeline/train_models.py --data runs/mog4d/data.pt --model both --architecture cm \
    --epochs 2000 --batch-size 1024 --output-dir runs/mog4d/models

# 3. guided optimisation + evaluation
python experiments/synthetic/pipeline/run_optimization.py --data runs/mog4d/data.pt --model-dir runs/mog4d/models \
    --method LGD_CM --n-attempts 25 --optim-samples 250 --output runs/mog4d/results.pt
```

## Paper figures

| Figure | Command |
|---|---|
| Fig. 1, toy example: joint, induced conditional vs. target, CDMS beta-sweep | `python experiments/synthetic/pipeline/fig1_toy_example.py` |
| Model check: score network vs. exact noised-mixture score, unguided samples vs. P(X), conditional sampler vs. exact P(Y\|X=x) at x*, the cluster centres and over x | `python experiments/synthetic/pipeline/check_models.py --setting hard` or `--prior ... --sampler ... --data ...` |
| Table 1, all scenarios: data, networks, model check, the four search algorithms, and the table (`--settings 5x5 10x1 ...` for a subset; `experiments/synthetic/slurm/slurm_table1.sh` runs one scenario per SLURM array task) | `python experiments/synthetic/pipeline/make_table1.py` |
| App. G subsampling control on the 10D setting | `python experiments/synthetic/pipeline/control_subsample.py --sim-dir /path/to/conditional-matching-paper/simulations --n-restarts 25` |

`fig1_toy_example.py` trains its three small networks on first use (cached under
`checkpoints/toy/<setting>/`), writes the simulation to `results/fig1_<setting>_sim.json`
and `_sweep.npz`, and draws the figure from those files. Re-running only redraws;
`--rerun-sim` recomputes. `--setting {hard,bimodal,paper,mog2d}` picks the joint
(`hard`: four tilted clusters in `x`, one of which induces the target; the others
are decoys with a different number, width or position of modes). `--methods`
chooses among `MLGD`, `MLGD-F` (independent restarts, `--n-restarts`) and
`PMLGD`, `PMLGD-F` (one resampled particle population with the same budget,
`--resample-beta`, `--ess-frac`); `--refine-steps` adds post-refinement to every
method. The figure shows the joint with every recovered `x_hat`, the population
loss `L(x)` next to the sample loss the optimiser sees, the induced conditionals
against the target, and the beta-sweep against the analytic `Q_beta`
(`--n-beta-samples`, `--betas`, `--layout lr|tb`). `--quick` is a smoke test,
`--uncond/--cond/--cm {auto,train,load}` control each network. On a GPU the
`hard` setting with four methods and ten restarts takes about 8 minutes;
`experiments/synthetic/slurm/slurm_fig1.sh` is the cluster job.
`control_subsample.py` needs the paper repository's 10D checkpoints, downloaded
from HuggingFace on first use.

`experiments/` holds one directory per paper table and figure — the Gaussian
simulations, the MNIST rotated-digits task and the Stable Diffusion experiment —
each with the exact command, its requirements and the published reference
numbers. Start at `experiments/synthetic/README.md`.

## Using the optimizer from Python

```python
from guidance import GuidanceConfig, optimize_guided
out = optimize_guided(model_uncond, model_cond, loss_fn, GuidanceConfig(...))
# out: x_final, final_loss, best_loss, best_x0, n_steps
```

`GuidanceConfig` holds every knob as an explicit strategy: initialisation,
Tweedie perturbation, aggregation over Monte-Carlo perturbations, step-size rule,
DDIM step style and stochasticity (`eta`), target sampling, and gradient
subsampling through `cond_output_transform`.

## Layout

```
cdm/                          THE METHOD (Section 2 of the paper), installable: `import cdm`
  guidance/                   Algorithm 1, the guided reverse-diffusion outer loop:
                              core.py (optimize_guided, optimize_guided_population), config.py (GuidanceConfig)
  losses/                     Algorithm 2, the inner estimator: MMD/RBF (functions.py), sliced Wasserstein, circular
  models/                     the frozen networks: diffusion.py (DiffusionModel), priors/ (P(X)),
                              samplers/ (P(Y|X): consistency models, exact),
                              mnist/ and sd/ (the MNIST and Stable Diffusion instantiations)
  utils/                      Gaussian mixtures, the exact L2 / TV between GMMs, network blocks, evaluation
experiments/                  THE EXPERIMENTS (Section 3), one directory per paper table or figure
  synthetic/                  3.1  Gaussian mixtures: pipeline/ (the scripts), configs/mog/, slurm/, one dir per experiment
  mnist/                      3.2  rotated digits: train/, table_mlgdf/, figures/, checkpoints/
  stable_diffusion/           3.3  sketch-to-portrait: four_targets/, its own requirements.txt
tests/                        smoke, determinism, population, targeted guards (see tests/README.md)
```

Each experiment directory has the same shape: `README.md` (what it produces, which
paper table or figure, the command), `run.sh` (a reduced `smoke` configuration and
the paper's `full` one), `config/`, `results/` (the published numbers) and
`outputs/` (gitignored, written by a local run).

## Tests

```bash
bash tests/run_all_tests.sh        # smoke, determinism, population, guards
```

The same suite runs on GitHub Actions for every push. It is CPU-only and uses
tiny untrained models: it shows the code runs and is deterministic, not that it
reproduces the paper's numbers (see `tests/README.md`).

## Cluster

`experiments/synthetic/slurm/cluster_setup.txt` explains the environment (modules, venv, CUDA
torch). `experiments/synthetic/slurm/slurm_fig1.sh` is a ready SLURM job for the Figure 1 runs
(`sbatch experiments/synthetic/slurm/slurm_fig1.sh hard`, `N_RESTARTS=50` to change the restart
count); it writes the figures and their simulation caches to `results/`, which
you copy back and redraw locally without recomputing.
