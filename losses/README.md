# `losses/`

Distance functions between an induced conditional sample and a target sample. They are
consumed by `guidance/`; nothing in here knows about diffusion.

## Files

| file | what is in it |
|---|---|
| `functions.py` | `RBF` kernel, `MMDLoss`, and the functional SWD pair `sliced_wasserstein_distance` / `compute_swd` |
| `sliced.py` | `SupportAwareSlicedWassersteinLoss` — the SWD used by the MNIST rotation experiment |
| `circular.py` | angle <-> `(cos, sin)` conversion and a MoG-on-the-circle density evaluator |
| `__init__.py` | imports the three submodules; exports no symbols of its own |

## Where to find each distance

The file names do not predict the contents. Concretely:

- **MMD and the RBF kernel** -> `functions.py` (`MMDLoss`, `RBF`).
- **SWD -> both files.** The *functions* `sliced_wasserstein_distance` and `compute_swd`
  (POT-backed, `ot.wasserstein_1d`) are in `functions.py`; the *class*
  `SupportAwareSlicedWassersteinLoss` (pure-torch, differentiable) is in `sliced.py`.
  Importers are split the same way: `scripts/check_models.py` and `utils/evaluation.py`
  take the function from `functions`, `legacy/Optimization.py` and `scripts/find_xstar.py`
  take the class from `sliced`. The class's `forward` is *not* the loss contract below —
  it takes four arguments and returns `(loss, dict)`; call sites that need a plain loss
  use its `sliced_wasserstein_distance` method instead (see `scripts/find_xstar.py`).
- **`circular.py` holds no distance.** It is target-distribution helper code for the MNIST
  angle representation. Only `legacy/Optimization.py` imports it.

## RBF defaults that matter when reading results

`RBF(n_kernels=5, mul_factor=2.0, bandwidth=None, device='cpu', detach_bandwidth=True)`.

- `n_kernels=5`, `mul_factor=2.0` — a mixture of 5 bandwidths, geometric with ratio 2,
  centred on the data-dependent bandwidth (the mean off-diagonal squared pairwise
  distance of the stacked `[X; Y]` batch).
- `detach_bandwidth=True` — the data-dependent bandwidth (computed from the batch when
  `bandwidth is None`) is treated as a constant of the kernel. `False` is the legacy
  behaviour and lets the gradient flow through it as well.

`MMDLoss()` with no kernel constructs `RBF(device=device)`, i.e. all of the above.

## The contract `guidance/` relies on

A loss is any callable `(X, Y) -> scalar tensor` — predicted samples first, target samples
second — injected into `optimize_guided` / `optimize_guided_population`
(`guidance/core.py`). Adding a distance therefore means adding a file here and nothing
else; `guidance/` does not need to change.

## Known messiness

SWD is split across `functions.py` and `sliced.py`, `circular.py` is filed here but holds
helpers rather than a loss, and two blocks of commented-out dead code (an old training
example and a superseded `sliced_wasserstein_distance`) were deleted from `functions.py`.
This is known, not a trap; the layout was left as-is deliberately.
