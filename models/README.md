# `models/`

The frozen networks CDM treats as given. The method itself lives in `guidance/` and
`losses/`; nothing here knows about the search for `x*`.

## The two roles

| package | role | what it is |
|---|---|---|
| `models/priors/` | `P(X)` | the unconditional model over `X` — `X` is what the method optimises |
| `models/samplers/` | `P(Y\|X)` | the frozen conditional generator: given `x`, draw `y` |

Map to the SD experiment: **`X` = the scribble** (the thing being optimised),
**`Y` = the generated image**.

## Why `DiffusionModel` sits at `models/` level, not in `priors/`

`models/diffusion.py:11` defines a single class that serves **both** roles. It takes
`condition: bool`, and with `condition=True` it *is* `P(Y|X)` — the diffusion-sampler
path (MLGD). `scripts/fig1_toy_example.py:275-277` builds both from the same class and
passes the conditional one to `optimize_guided` as `model_cond` (line 386).

Putting that class inside a package named `priors/` would name it for one of the two
roles it plays, which is why it sits at `models/` level instead.

`models/priors/` therefore contains **only an `__init__.py` of re-exports**. It is kept
because it is the `P(X)` name in the paper's split and the counterpart to
`models/samplers/`; the classes it names live one directory up.

## `diffusion.py` vs `diffusion_general.py`

Both define a class called `DiffusionModel`. They are different models, not duplicates
(see the naming note at `models/diffusion_general.py:447-466`).

| module | formulation | exported as |
|---|---|---|
| `diffusion.py` | **discrete-time**: timestep-indexed DDPM/DDIM, `betas`/`alphas` schedule, `sample_ddim_step` | `models.DiffusionModel` |
| `diffusion_general.py` | **continuous-time SDE**: drift/diffusion coefficients, score-matching loss | `models.DiffusionModelCT` |

**"Discrete-time" means timestep-indexed. It does NOT mean discrete-state diffusion —
there is none in this repo.**

**Only the discrete-time class is used.** Every import in the live code
(`scripts/`, `tests/`, `legacy/`) is `from models.diffusion import DiffusionModel` or
`from models import diffusion as Diffusion`. `diffusion_general.py` has **no importers**
outside `models/__init__.py` and `models/priors/__init__.py`, which only re-export it.
As far as the paper is concerned it is dead code — flagged, not deleted.

It also carries an in-module alias `DiffusionModelSDE = DiffusionModel`
(`diffusion_general.py:468`), so the continuous-time class answers to three names: `models.diffusion_general.DiffusionModel`,
`...DiffusionModelSDE`, and `models.DiffusionModelCT`.

The notebooks import a top-level `DiffusionGeneral` module that does not exist in this
repository; `notebooks/` is unmaintained.

## `models/samplers/`

| file | what is in it |
|---|---|
| `consistency.py` | `ConsistencyModel` and `ConsistencyModeliCT` — consistency-model samplers (MLGD-F) |
| `exact.py` | `ExactConditionalSampler` — oracle draws from `P(Y\|X)` of a known Gaussian mixture, no network |
| `flow_matching.py` | `FMModel` — flow-matching sampler over an ODE path |
| `fm_solver.py` | not a sampler: patches a `flow_matching` `ODESolver` with a gradient-carrying `compute_likelihood_allow_grad`, used by `flow_matching.py` |

`guidance/core.py:73` calls `model_cond.sample(nsamples=..., condition_x=..., device=...)`.
`consistency.py`, `exact.py` and `models/diffusion.py` all implement that signature and
are interchangeable there. **`FMModel` does not** — its signature is
`sample(num_sample, y=None, ...)`, and it is driven through `optimize_DFLOW`
(`legacy/Optimization.py:265`) rather than through `optimize_guided`.

## Which class a checkpoint comes from

Checkpoints written by `scripts/train_models.py` carry an explicit `"architecture"` key
alongside `"state_dict"` and `"config"` (`train_models.py:207-234`), and
`scripts/run_optimization.py:71-97` dispatches on it:

| `architecture` | class reconstructed |
|---|---|
| `"diffusion"` | `models.diffusion.DiffusionModel` (discrete-time) |
| `"cm"` | `models.samplers.consistency.ConsistencyModeliCT` — note: the iCT variant, not `ConsistencyModel` |
| `"fm"` | `models.samplers.flow_matching.FMModel` |

No checkpoint format in this repo reconstructs `DiffusionModelCT`.

`MnistModels/MnistConditional500Epoch.pt` is a different format — its keys are `epoch`,
`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `loss`,
`losses_history`, `nfeatures`, `img_features`, with **no architecture field** — and
nothing in this repo constructs a model from it. Its class cannot be determined from
this repo.
