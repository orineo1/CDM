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

## Finding x*: the main script

`scripts/find_xstar.py` is the method. It takes a prior over `X`, a conditional
sampler for `Y|X` and a target `G(Y)`, runs guided reverse diffusion with the
distributional loss as guidance, as independent restarts or as a resampled particle
population, and returns the best `x_hat`:

```bash
python scripts/find_xstar.py \
    --prior   runs/mog4d/models/model_unconditional.pt \
    --sampler runs/mog4d/models/model_conditional.pt \
    --target  runs/mog4d/data.pt \
    --n-attempts 10 --out runs/mog4d/xstar.json
```

The prior and sampler are checkpoints written by `scripts/train_models.py` (see
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
JSON holds all of them and the best. Attempts are also appended to
`results/runs.jsonl`.

## Producing a prior, a sampler and a target: the synthetic pipeline

One command draws a joint `P(X,Y)`, trains the prior on `X` and the conditional
sampler on `Y|X`, runs the guided optimisation and evaluates the recovered
`x_hat` against the known optimum (a few minutes on CPU, most of it training):

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
| Model check: score network vs. exact noised-mixture score, unguided samples vs. P(X), conditional sampler vs. exact P(Y\|X=x) at x*, the cluster centres and over x | `python scripts/check_models.py --setting hard` or `--prior ... --sampler ... --data ...` |
| Table 1, all scenarios: data, networks, model check, the four search algorithms, and the table (`--settings 5x5 10x1 ...` for a subset; `scripts/slurm_table1.sh` runs one scenario per SLURM array task) | `python scripts/make_table1.py` |
| App. G subsampling control on the 10D setting | `python scripts/control_subsample.py --sim-dir /path/to/conditional-matching-paper/simulations --n-restarts 25` |
| Witness vs. uniform gradient subsampling in the Table 1 format: TV of both rules at every (`--n-cond-list`, `--k-fracs`) point, reusing Table 1's data and networks (`scripts/slurm_witness_vs_uniform.sh` per scenario) | `python scripts/witness_vs_uniform_table.py` |

`witness_vs_uniform_table.py` keeps the loss value on all `n_cond` conditional samples but the
gradient on only `k = k/n * n_cond` of them (`find_xstar.py --select uniform|witness --n-sel k`):
`uniform` picks them at random, `witness` proportionally to the MMD witness `|w(y_i)|` (the loss's
kernel, scored against the loss's own target batch) mixed with `--witness-floor` of uniform
(default 0), by systematic PPS sampling. With `--rescale ht` (the table's default) each kept
gradient is divided by its inclusion probability (Horvitz-Thompson), so both rules give an
unbiased estimate of the full gradient (`utils/witness.py`, `tests/test_witness.py`). Defaults:
`--n-cond-list 250 --k-fracs 0.5 --rules uniform witness`. With floor 0, a sample with a
near-zero witness score is rarely picked but then weighted by up to `1/pi`, which can be large;
`--witness-floor 0.1`-`0.3` bounds it.

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
`scripts/slurm_fig1.sh` is the cluster job.
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
scripts/        all command-line entry points (find_xstar.py is the method), test runners, cluster templates
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
reference in `legacy/Optimization_legacy.py` on fixed seeds. The same suite runs
on GitHub Actions for every push.

## Results

Runs call `gcio.results.save_run(...)` and tables are built from the store,
`results/runs.jsonl`.

## Cluster

`scripts/cluster_setup.txt` explains the environment (modules, venv, CUDA
torch). `scripts/slurm_fig1.sh` is a ready SLURM job for the Figure 1 runs
(`sbatch scripts/slurm_fig1.sh hard`, `N_RESTARTS=50` to change the restart
count); it writes the figures and their simulation caches to `results/`, which
you copy back and redraw locally without recomputing.
