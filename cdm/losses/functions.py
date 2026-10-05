import numpy as np
import torch
from torch import nn
import ot
from cdm.utils import dist as dist_utils
#### MMD Loss #####

class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, device='cpu', detach_bandwidth=True):
        super().__init__()
        self.device = device
        self.bandwidth_multipliers = (mul_factor ** (torch.arange(n_kernels, device=device) - n_kernels // 2))
        self.bandwidth = bandwidth
        # the data-dependent bandwidth is a constant of the kernel, not a function to differentiate:
        # with detach_bandwidth=False (as in the Table 1 runs) the gradient also flows through it
        self.detach_bandwidth = detach_bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            bw = L2_distances.sum() / (n_samples ** 2 - n_samples)
            return bw.detach() if self.detach_bandwidth else bw
        return self.bandwidth

    def forward(self, X):
        X = X.to(self.device)
        L2_distances = torch.cdist(X, X, p=2) ** 2
        bandwidth = self.get_bandwidth(L2_distances)
        scaled = L2_distances[None, ...] / (bandwidth * self.bandwidth_multipliers[:, None, None])
        return torch.exp(-scaled).sum(dim=0)


# ---------------------------------------------------------------------------
# The MMD estimator itself.
#
# `_TARGET_CACHE`, `_target_stats` and `compute_mmd` below are the MMD that produced
# the paper's Stable Diffusion tables: median-heuristic bandwidth, optional target
# cache, sqrt(|mmd_sq| + 1e-8) return value. Changing it changes the reported numbers.
# ---------------------------------------------------------------------------

_TARGET_CACHE = {}


def _target_stats(y):
    key = (y.data_ptr(), tuple(y.shape), str(y.device), str(y.dtype))
    st = _TARGET_CACHE.get(key)
    if st is None:
        with torch.no_grad():
            y_sq = (y ** 2).sum(dim=1, keepdim=True)
            d_yy = (y_sq + y_sq.T - 2 * torch.mm(y, y.T)).clamp_min(0.0)
        st = {"y_sq": y_sq, "d_yy": d_yy}
        _TARGET_CACHE.clear()          # one target set per run
        _TARGET_CACHE[key] = st
    return st


def compute_mmd(x, y, bandwidth=None, bandwidth_scale=1.0, kernel_alpha=1.0,
                cache_target=False):
    """
    Unbiased Maximum Mean Discrepancy with a generalised RBF kernel.

    Args:
        x:               [n, d] generated embeddings (grad flows through).
        y:               [m, d] target embeddings (detached).
        bandwidth:       kernel bandwidth; estimated via median heuristic if None.
        bandwidth_scale: multiplicative scale applied after median estimation.
        kernel_alpha:    RBF exponent. 1 = standard Gaussian.
                         >1 = flatter centre, sharper falloff.
        cache_target:    reuse the fixed target set's pairwise distances across calls
                         (exact; see _target_stats). Off by default.

    Returns:
        Scalar MMD estimate (sqrt of unbiased MMD²).
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    dev = x.device
    x = x.float().to(dev)
    y = y.float().to(dev).detach()

    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    if y.dim() > 2:
        y = y.reshape(y.shape[0], -1)

    n, m = x.shape[0], y.shape[0]

    def rbf_kernel(a, b, bw, alpha):
        a_sq = (a ** 2).sum(dim=1, keepdim=True)
        b_sq = (b ** 2).sum(dim=1, keepdim=True)
        dist_sq = a_sq + b_sq.T - 2 * torch.mm(a, b.T)
        return torch.exp(-(dist_sq / (2 * bw ** 2)) ** alpha)

    if bandwidth is None:
        ss = min(1000, n, m)
        with torch.no_grad():
            x_sq = (x[:ss].detach() ** 2).sum(dim=1, keepdim=True)
            y_sq = (y[:ss] ** 2).sum(dim=1, keepdim=True)
            dists = x_sq + y_sq.T - 2 * torch.mm(x[:ss].detach(), y[:ss].T)
            dists = dists[dists > 0]
            bandwidth = (
                torch.sqrt(torch.median(dists) / 2)
                if len(dists) > 0
                else torch.tensor(1.0, device=dev)
            )
        bandwidth = bandwidth.detach() * bandwidth_scale

    K_xx = rbf_kernel(x, x, bandwidth, kernel_alpha)
    if cache_target:
        st = _target_stats(y)
        with torch.no_grad():
            K_yy = torch.exp(-(st["d_yy"] / (2 * bandwidth ** 2)) ** kernel_alpha)
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        dist_xy = x_sq + st["y_sq"].T - 2 * torch.mm(x, y.T)
        K_xy = torch.exp(-(dist_xy / (2 * bandwidth ** 2)) ** kernel_alpha)
    else:
        K_yy = rbf_kernel(y, y, bandwidth, kernel_alpha)
        K_xy = rbf_kernel(x, y, bandwidth, kernel_alpha)

    # Skip K_xx diagonal term when n=1 to avoid 0/0
    xx_term = (K_xx.sum() - K_xx.trace()) / (n * (n - 1)) if n > 1 else 0.0
    yy_term = (K_yy.sum() - K_yy.trace()) / (m * (m - 1)) if m > 1 else 0.0
    xy_term = 2 * K_xy.sum() / (n * m)

    mmd_sq = xx_term - xy_term + yy_term
    # abs() before sqrt handles slightly-negative unbiased estimates
    return torch.sqrt(mmd_sq.abs() + 1e-8)


def _mmd_from_gram(K, n, m):
    """`compute_mmd`'s closing arithmetic, applied to an already-built Gram matrix.

    Only used by `MMDLoss` when a kernel object is injected: an injected kernel (e.g.
    the mixture `RBF` above) is handed the stacked `[X; Y]` batch and so cannot be
    expressed through `compute_mmd`, which builds its own single-bandwidth kernel.
    The lines are the ones above, in the same order, with the same sqrt wrapper.
    """
    K_xx, K_yy, K_xy = K[:n, :n], K[n:, n:], K[:n, n:]
    xx_term = (K_xx.sum() - K_xx.trace()) / (n * (n - 1)) if n > 1 else 0.0
    yy_term = (K_yy.sum() - K_yy.trace()) / (m * (m - 1)) if m > 1 else 0.0
    xy_term = 2 * K_xy.sum() / (n * m)
    mmd_sq = xx_term - xy_term + yy_term
    return torch.sqrt(mmd_sq.abs() + 1e-8)


class MMDLoss(nn.Module):
    """`(X, Y) -> scalar` MMD, computed by the ported `compute_mmd` above.

    `kernel=None` (the default) calls `compute_mmd` directly: the paper's
    single-bandwidth generalised RBF with the median heuristic.

    A kernel object may still be injected -- every existing call site in this repo does,
    passing the mixture `RBF` -- in which case the Gram matrix of the stacked `[X; Y]`
    batch comes from that kernel and `_mmd_from_gram` finishes the calculation with
    `compute_mmd`'s own arithmetic. Either way the returned value is the paper's:
    sqrt(|MMD^2| + 1e-8).
    """

    def __init__(self, kernel=None, device='cpu'):
        super().__init__()
        self.device = device
        self.kernel = kernel
        self.to(device)

    def forward(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)
        if self.kernel is None:
            return compute_mmd(X, Y)
        K = self.kernel(torch.vstack([X, Y]))
        return _mmd_from_gram(K, X.shape[0], Y.shape[0])


def sliced_wasserstein_distance(X, Y, num_projections=100, p=2,
                                mu_list_sliced=None, Sigma_list_sliced=None, alpha=None,Flag=False):
    n, d = X.shape
    projections = torch.randn(num_projections, d, device=X.device, dtype=X.dtype)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)  # normalize

    swd = 0.0
    for proj in projections:
        X_proj = (X @ proj).cpu().numpy()
        Y_proj = (Y @ proj).cpu().numpy()

        X_proj.sort()
        Y_proj.sort()
        swd += ot.wasserstein_1d(X_proj, Y_proj)

    swd = swd / num_projections

    # Optional normalization by sqrt(trace(cov)/d)
    if mu_list_sliced is not None and Sigma_list_sliced is not None and alpha is not None:
        cov = dist_utils.mog_covariance(mu_list_sliced, Sigma_list_sliced, alpha)
        norm_coef = torch.sqrt(torch.trace(cov) / d).item()
        norm_swd = swd / norm_coef

    if Flag:
      print(f"Raw SWD (before normalization): {swd:.6f}")
      print(f"Covariance trace: {torch.trace(cov):.6f}")
      print(f"Dimensionality d: {d}")
      print(f"Normalization coefficient: {norm_coef:.6f}")
      print(f"Normalized SWD: {swd:.6f}")

    return swd,norm_swd


def compute_swd(
        best_x0_sample,
        mu_list,
        Sigma_list,
        alpha,
        mog_means,
        mog_variances,
        weights,
        decoder=None,
        scaler=None,
        device="cpu",
        nsamples=2_500,
        num_projections=200,
        p=2,
        Flag=False
):
    """
    Full pipeline:
    latent -> (optional decode/denormalize) -> conditionals -> MoG samples -> SWD

    If `decoder` and `scaler` are None, the function skips decoding and uses best_x0_sample directly.
    """
    with torch.no_grad():
        if decoder is not None and scaler is not None:
            # Decode from latent
            decoded_scaled = decoder(best_x0_sample)

            # Denormalize
            decoded_numpy = scaler.inverse_transform(decoded_scaled.cpu().numpy())
            decoded_original = torch.tensor(decoded_numpy, device=device, dtype=torch.float32)
            if Flag:
                print("decoded_x_star", decoded_original)
        else:
            # Use latent directly
            decoded_original = best_x0_sample.to(device).float()
            if Flag:
                print("Using raw latent (no decoding):", decoded_original)

        # Conditionals
        mu_temp_optim, Sigma_temp_optim = dist_utils.compute_conditionals(
            mu_list, Sigma_list, decoded_original.reshape(-1, 1)
        )

        # Mixture weights
        temp_alpha_optim = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, decoded_original)


        # --- Ensure covariances are symmetric ---
        for i in range(Sigma_temp_optim.shape[0]):
            original = Sigma_temp_optim[i].clone()
            Sigma_temp_optim[i] = (Sigma_temp_optim[i] + Sigma_temp_optim[i].T) / 2
            diff_norm = torch.norm(Sigma_temp_optim[i] - original).item()
            if diff_norm > 1e-8:
                print(f"Covariance matrix {i} symmetrized. Frobenius norm of change: {diff_norm:.6f}")

        # Optimized samples
        mog_samples_optim = dist_utils.generate_mog_samples_not_differentiable(
            nsamples, mu_temp_optim, Sigma_temp_optim, temp_alpha_optim
        )
        mog_samples_optim = mog_samples_optim.float()

        # Raw samples
        mog_samples_raw = dist_utils.generate_mog_samples_not_differentiable(
            nsamples, mog_means, mog_variances, weights
        )
        mog_samples_raw = mog_samples_raw.float()



        swd,norm_swd = sliced_wasserstein_distance(mog_samples_optim, mog_samples_raw,
                                          num_projections, p,
                                          mog_means, mog_variances, weights
                                          # mu_list_sliced, Sigma_list_sliced, alpha
        )

    return swd,norm_swd



# ---------------------------------------------------------------------------
# Pointwise (non-distributional) baseline loss.
#
# Added for the no-star experiment (experiments/synthetic/README.md), whose two
# "pointwise" arms need a squared-error objective rather than a distributional one:
# ((target_samples - mog_samples) ** 2).mean(), an unpaired,
# per-row squared distance between two independently drawn batches. It is exactly
# the paper's pointwise (y - y*)^2 when the target is degenerate (every row equal,
# i.e. sigma_t -> 0), which is the only setting the paper uses it in; for a
# non-degenerate target it depends on the arbitrary row pairing between the two
# batches, so it is not a distance and is deliberately not offered as a general
# alternative to MMD/SWD.
# ---------------------------------------------------------------------------
def pointwise_squared_error(model_samples, target_samples):
    """Mean squared distance between two equal-length sample batches, row by row.

    Both arguments are [n, dim_y]. Differentiable in `model_samples`.
    """
    if model_samples.shape[0] != target_samples.shape[0]:
        n = min(model_samples.shape[0], target_samples.shape[0])
        model_samples, target_samples = model_samples[:n], target_samples[:n]
    return ((target_samples - model_samples) ** 2).mean()
