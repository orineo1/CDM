# CDM: Inverse Design for Conditional Distribution Matching

Code for the paper *Inverse Design for Conditional Distribution Matching* (Meidler, Tolkowsky, Zuk).
**Conditional Distribution Matching (CDM)**: given a joint distribution `P(X, Y)`, available through
trained generative models, and a target distribution `G(Y)`, find an input `x*` whose induced
conditional `P(Y | X = x*)` matches `G`. The solver, MLGD, is loss-guided reverse diffusion over `X`
with a distributional loss (MMD or sliced Wasserstein) estimated from samples of a conditional
sampler for `Y | X`; MLGD-F uses a few-step sampler.

The method is the `cdm` package; `experiments/` holds one directory per paper experiment (Gaussian
mixtures, rotated MNIST, Stable Diffusion, and the four-bead peptide comparison), each with its
driver, job scripts, README and the published reference numbers.
Everything is run from the repository root on the command line; every script
prints `--help`, auto-detects a GPU and falls back to CPU (`--device` overrides).

## Install

```bash
git clone https://github.com/orineo1/CDM.git
cd CDM
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU is enough for the synthetic examples
pip install -e .                           # makes `import cdm` work from anywhere
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121
# Stable Diffusion experiment only: pip install -r experiments/stable_diffusion/requirements.txt
#   then `wandb login`; experiments/stable_diffusion/README.md has the cluster (sbatch) commands
```

## Quick start: a small example on CPU (about 2 minutes)

Draw a random 4-dimensional Gaussian mixture `P(X, Y)` with `dim X = 3`, train a prior on `X` and a
few-step conditional sampler for `Y | X`, and search for `x*`. The target is `G = P(Y | X = x*)` for a
known `x*`, so the output reports how close the recovered `x_hat` is (`l2_to_xstar`, and `l2_gmm`, the
exact L2 distance between `P(Y | X = x_hat)` and `G`).

```bash
python experiments/synthetic/pipeline/generate_data.py --random-mog --dim 4 --n-components 3 --condition-on 3 \
    --n-samples 10000 --output runs/quick/data.pt
python experiments/synthetic/pipeline/train_models.py --data runs/quick/data.pt --model both --architecture cm \
    --epochs 200 --batch-size 1024 --output-dir runs/quick/models
python experiments/synthetic/pipeline/search_mog.py --prior runs/quick/models/model_unconditional.pt \
    --sampler runs/quick/models/model_conditional.pt --target runs/quick/data.pt --n-restarts 3 --out runs/quick/xstar.json
```

More epochs (e.g. 2000) give better networks and a closer `x_hat`. For the paper's fixed settings use
`generate_data.py --params-file experiments/synthetic/configs/mog/mog_2d_full.txt`.

## Finding x*: the search

The method is one function, `cdm.guidance.search.search` (guided reverse diffusion with the
distributional loss as guidance in `cdm/guidance/core.py`): R independent guided trajectories,
optional refinement, and selection of one design. Each experiment has a driver named after its
dataset that loads its models and target and calls it, with the same search flags everywhere
(`--n-restarts`, `--guide-grad`, `--refine-*`, `--select-*`):

| experiment | driver |
|---|---|
| Gaussian mixtures (Table 1) | `experiments/synthetic/pipeline/search_mog.py` |
| rotated MNIST (Table 2) | `experiments/mnist/table_mlgdf/search_mnist.py` |
| four-bead peptide (comparison with GB-FESO) | `experiments/peptide/pipeline/search_peptide.py` |

For vector data with trained checkpoints, `search_mog.py` takes a prior over `X`, a conditional
sampler for `Y|X` and a target `G(Y)`, runs the search (or a resampled particle population), and
returns the best `x_hat`:

```bash
python experiments/synthetic/pipeline/search_mog.py \
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

## Paper figures and tables: which script produces each

Numbers follow the AISTATS 2027 version of the paper. Each experiment README has the exact flags.

| paper item | content | script (from the repository root) |
|---|---|---|
| Figure 1 | toy example: joint, loss landscape, recovered conditionals, CDMS beta-sweep with twisted-SMC correction | `experiments/synthetic/pipeline/fig1_toy_example.py --setting hard` and `fig_smc_correction.py`; cluster: `experiments/synthetic/slurm/slurm_fig1.sh hard`; see `experiments/synthetic/README.md#figure-1` |
| Table 1 | synthetic grid, TV per (dim X, dim Y) and method | `experiments/synthetic/pipeline/make_table1.py` (all flags in `experiments/synthetic/README.md`, "Reproducing the paper"); cluster: `experiments/synthetic/slurm/slurm_table1.sh` |
| Table 2 | MNIST rotated digits, SWD and recovered digits | `bash experiments/mnist/table_mlgdf/run.sh full` (runs `search_mnist.py`), then `make_table.py` |
| Figure 2 | MNIST optimized digits and angle histograms | `bash experiments/mnist/figures/run.sh full` |
| Table 3 | Stable Diffusion, four distributional targets, MMD against baselines | `bash experiments/stable_diffusion/four_targets/run.sh full <scenario>`, then `run.sh baselines` and `run.sh eval` |
| Figure 3 | Stable Diffusion source and optimized scribbles with their portraits | written by the `four_targets` full runs (`scribble_source.png`, `scribble_mlgdd.png`, conditional samples) |
| Table 4 | accuracy of the trained synthetic networks | written by `make_table1.py` (`check_models.py` per scenario) |
| Table 5 | synthetic grid, appendix companion (distances, ablations) | `make_table1.py` with the ablation tags in `experiments/synthetic/README.md` |
| Table 6, Figure 4 | no-star target | `experiments/synthetic/README.md#no-star-target` |
| Tables 7–9 | MNIST network architecture, training and MLGD-F hyperparameters | `experiments/mnist/train/` (training) and `experiments/mnist/table_mlgdf/config/targets.tsv` (search settings) |
| Figures 5–7 | MNIST, all 15 seeds per target | `bash experiments/mnist/figures/run.sh full` |
| Table 10 | Stable Diffusion hyperparameters | `experiments/stable_diffusion/four_targets/config/<scenario>.env` |
| Table 11 | Stable Diffusion MMD and SWD, all targets | `experiments/stable_diffusion/four_targets/run.sh eval` |
| Figure 8 | where MLGD-F edits the scribble | scribble-difference heatmap, `cdm/models/sd/visualization.py`, from a `four_targets` run |
| Figure 9 | gradient saliency and its alignment with the edits | not in this repository (upstream notebook `gender_saliency_eval.ipynb`; see `experiments/stable_diffusion/README.md`) |
| Figures 10–17 | Stable Diffusion qualitative results per target (scribbles, conditional samples, age grid) | the `four_targets` full runs; interpolation figures: `experiments/stable_diffusion/interpolation/interpolation.ipynb` |
| Table 12 | guidance rules | text only |
| Table 13 | per-step time of the Stable Diffusion inner estimator | not in this repository (upstream notebook `measure_dps_step_memory.ipynb`) |
| Table 14, Figure 18 | Jacobian error of the few-step sampler | `experiments/stable_diffusion/jacobian_error/eps_g_experiment.ipynb` |
| Table 15 | licenses | text only |

Beyond the paper: `experiments/peptide/` compares MLGD with GB-FESO (Zick and Mendels, 2026) on their
four-bead peptide (`search_peptide.py`; results in `experiments/peptide/results/`), and
`experiments/synthetic/README.md` ("Improved search at (10, 1)") documents a stronger search recipe.

## Using the optimizer from Python

```python
from cdm.guidance import GuidanceConfig, optimize_guided
from cdm.guidance.search import search
out = optimize_guided(model_uncond, model_cond, loss_fn, GuidanceConfig(...))   # one trajectory
# out: x_final, final_loss, best_loss, best_x0, n_steps, trace, refine
res = search(model_uncond, model_cond, loss_fn, make_cfg, seeds=range(10), select_top=5)   # R trajectories + selection
# res: best (the chosen run), runs, time_s; make_cfg(seed) returns the GuidanceConfig of one trajectory
```

`GuidanceConfig` holds every knob as an explicit strategy: initialisation,
Tweedie perturbation, aggregation over Monte-Carlo perturbations, step-size rule,
DDIM step style and stochasticity (`eta`), target sampling, and gradient
subsampling through `cond_output_transform`.

## Layout

```
cdm/                          THE METHOD (Section 2 of the paper), installable: `import cdm`
  guidance/                   Algorithm 1, the guided reverse-diffusion outer loop:
                              core.py (optimize_guided, optimize_guided_population, refine), config.py (GuidanceConfig),
                              search.py (R trajectories + selection, and the shared search flags)
  losses/                     Algorithm 2, the inner estimator: MMD/RBF (functions.py), sliced Wasserstein, circular
  models/                     the frozen networks: diffusion.py (DiffusionModel), priors/ (P(X)),
                              samplers/ (P(Y|X): consistency models, exact),
                              mnist/ and sd/ (the MNIST and Stable Diffusion instantiations)
  utils/                      Gaussian mixtures, the exact L2 / TV between GMMs, network blocks, evaluation
experiments/                  THE EXPERIMENTS (Section 3), one directory per paper table or figure
  synthetic/                  3.1  Gaussian mixtures: pipeline/ (search_mog.py and the scripts), slurm/, configs/mog/, reference/, results/
  mnist/                      3.2  rotated digits: train/, table_mlgdf/ (search_mnist.py), figures/, checkpoints/
  stable_diffusion/           3.3  sketch-to-portrait: four_targets/, interpolation/, jacobian_error/, its own requirements.txt
  peptide/                    four-bead peptide: CDM against GB-FESO (Zick & Mendels 2026); pipeline/, slurm/, results/
tests/                        smoke, determinism, population, targeted guards (see tests/README.md)
```

The MNIST and Stable Diffusion experiment directories have the same shape: `README.md`
(what it produces, which paper table or figure, the command), `run.sh` (a reduced
`smoke` configuration and the paper's `full` one), `config/`, `results/` (the
published numbers) and `outputs/` (gitignored, written by a local run). The synthetic
experiments share one pipeline, so `experiments/synthetic/README.md` lists every result
with its command instead.

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
