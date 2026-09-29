# `cdm/models/`

The frozen networks CDM treats as given. The method itself lives in `cdm/guidance/` and
`cdm/losses/`; nothing here knows about the search for `x*`.

## The two roles

| package | role | what it is |
|---|---|---|
| `cdm/models/priors/` | `P(X)` | the unconditional model over `X` — `X` is what the method optimises |
| `cdm/models/samplers/` | `P(Y\|X)` | the frozen conditional generator: given `x`, draw `y` |

Map to the SD experiment: **`X` = the scribble** (the thing being optimised),
**`Y` = the generated image**.

## Why `DiffusionModel` sits at `cdm/models/` level, not in `priors/`

`cdm/models/diffusion.py:11` defines a single class that serves **both** roles. It takes
`condition: bool`, and with `condition=True` it *is* `P(Y|X)` — the diffusion-sampler
path (MLGD). `experiments/synthetic/pipeline/fig1_toy_example.py:275-277` builds both from the same class and
passes the conditional one to `optimize_guided` as `model_cond` (line 386).

Putting that class inside a package named `priors/` would name it for one of the two
roles it plays, which is why it sits at `cdm/models/` level instead.

`cdm/models/priors/` therefore contains **only an `__init__.py` of re-exports**. It is kept
because it is the `P(X)` name in the paper's split and the counterpart to
`cdm/models/samplers/`; the classes it names live one directory up.

## `diffusion.py`

`DiffusionModel` is **discrete-time**: timestep-indexed DDPM/DDIM, `betas`/`alphas`
schedule, `sample_ddim_step`. Exported as `models.DiffusionModel`.

**"Discrete-time" means timestep-indexed. It does NOT mean discrete-state diffusion —
there is none in this repo.**

## `cdm/models/samplers/`

| file | what is in it |
|---|---|
| `consistency.py` | `ConsistencyModel` and `ConsistencyModeliCT` — consistency-model samplers (MLGD-F) |
| `exact.py` | `ExactConditionalSampler` — oracle draws from `P(Y\|X)` of a known Gaussian mixture, no network |

`cdm/guidance/core.py:73` calls `model_cond.sample(nsamples=..., condition_x=..., device=...)`.
`consistency.py`, `exact.py` and `cdm/models/diffusion.py` all implement that signature and
are interchangeable there.

## Which class a checkpoint comes from

Checkpoints written by `experiments/synthetic/pipeline/train_models.py` carry an explicit `"architecture"` key
alongside `"state_dict"` and `"config"` (`train_models.py:207-234`), and
`experiments/synthetic/pipeline/run_optimization.py:71-97` dispatches on it:

| `architecture` | class reconstructed |
|---|---|
| `"diffusion"` | `models.diffusion.DiffusionModel` (discrete-time) |
| `"cm"` | `models.samplers.consistency.ConsistencyModeliCT` — note: the iCT variant, not `ConsistencyModel` |

`experiments/mnist/checkpoints/MnistConditional500Epoch.pt` is a different format — its keys are `epoch`,
`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `loss`,
`losses_history`, `nfeatures`, `img_features`, with **no architecture field** — and
nothing in this repo constructs a model from it. Its class cannot be determined from
this repo.
