# `cdm/losses/`

Sample distances between generated samples `X` and target samples `Y`. A loss is any callable
`(X, Y) -> scalar tensor`; `cdm/guidance/` takes it as an argument, so adding a distance needs no change there.

| file | contents |
|---|---|
| `functions.py` | `compute_mmd` (the MMD of the Stable Diffusion experiment), the multi-bandwidth `RBF` kernel and `MMDLoss` (synthetic experiments), sliced Wasserstein functions, the pointwise squared-error baseline |
| `sliced.py` | `SupportAwareSlicedWassersteinLoss`, a differentiable sliced Wasserstein loss |
| `circular.py` | angle <-> (cos, sin) helpers and a wrapped Gaussian-mixture density (MNIST) |
| `exact_mmd.py` | closed-form MMD between a mixture's conditional and a mixture target (an oracle loss) |
| `kde_kl.py` | GB-FESO's KDE estimate of the KL divergence, copied unchanged (MIT) |

`RBF(n_kernels=5, mul_factor=2.0)` mixes 5 bandwidths, spaced by a factor of 2 around the mean pairwise
squared distance of the merged sample. That bandwidth is treated as a constant (`detach_bandwidth=True`).
`compute_mmd` returns `sqrt(|MMD²| + 1e-8)`.
