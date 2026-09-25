# `losses/`

Distance functions between an induced conditional sample and a target sample. They are
consumed by `guidance/`; nothing in here knows about diffusion.

## Files

| file | what is in it |
|---|---|
| `functions.py` | `compute_mmd` (the paper's, see below), `RBF` kernel, `MMDLoss`, and the functional SWD pair `sliced_wasserstein_distance` / `compute_swd` |
| `sliced.py` | `SupportAwareSlicedWassersteinLoss` — the SWD used by the MNIST rotation experiment |
| `circular.py` | angle <-> `(cos, sin)` conversion and a MoG-on-the-circle density evaluator |
| `__init__.py` | imports the three submodules; exports no symbols of its own |

## Where to find each distance

The file names do not predict the contents. Concretely:

- **MMD and the RBF kernel** -> `functions.py` (`compute_mmd`, `MMDLoss`, `RBF`).
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

`MMDLoss()` with no kernel does not construct an `RBF`; it calls `compute_mmd`, which
builds its own single-bandwidth kernel (see below). The mixture above is what every
call site injects explicitly, `MMDLoss(kernel=RBF(...))`.

## The MMD is the paper's, not a local one

`compute_mmd` in `functions.py` is

    conditional-matching-paper  SD_cond_SD_controlnet/src/metrics.py :: compute_mmd

the function that produced the paper's Stable Diffusion tables — same maths, same
median-heuristic bandwidth, same `cache_target` path, same `sqrt(|mmd_sq| + 1e-8)`
return. It takes `bandwidth`, `bandwidth_scale`, `kernel_alpha` and `cache_target`.

`MMDLoss` is the `nn.Module` wrapper around it and returns the same quantity:

- `MMDLoss()` (no kernel) calls `compute_mmd` directly.
- `MMDLoss(kernel=...)` keeps the injection seam every call site in this repo uses: the
  injected kernel builds the Gram matrix of the stacked `[X; Y]` batch and
  `_mmd_from_gram` finishes with `compute_mmd`'s own closing arithmetic. An injected
  kernel (the mixture `RBF`) cannot be expressed through `compute_mmd`, which builds its
  own kernel, which is why the seam exists.

Both paths return the square-rooted value, as upstream does — a non-negative distance,
not the squared statistic.

## The contract `guidance/` relies on

A loss is any callable `(X, Y) -> scalar tensor` — predicted samples first, target samples
second — injected into `optimize_guided` / `optimize_guided_population`
(`guidance/core.py`). Adding a distance therefore means adding a file here and nothing
else; `guidance/` does not need to change.

## Known messiness

SWD is split across `functions.py` and `sliced.py`, and `circular.py` is filed here but
holds helpers rather than a loss. The layout is deliberate; the section above says where
to look for each distance.
