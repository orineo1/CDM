"""
Generate fixed representative points from a target distribution G (MoG)
with associated weights, for use in the likelihood-based optimizer.

For low-dimensional Y (dim <= 2): uses a deterministic grid with density-based weights.
For higher-dimensional Y: uses stratified Monte Carlo sampling from G components.

Usage:
    from representative_points import get_representative_points

    # G is specified by its MoG parameters
    y_rep, w_rep = get_representative_points(
        target_means, target_variances, target_weights,
        n_rep=200, device="cpu",
    )
    # y_rep: [n_rep, dim_y] tensor of representative points
    # w_rep: [n_rep] tensor of normalized weights (sum to 1)
"""

import torch
import numpy as np
from torch.distributions import MultivariateNormal


def get_representative_points(target_means, target_variances, target_weights,
                              n_rep=200, device="cpu", grid_scale=4.0,
                              method=None):
    """
    Generate n_rep representative points from G with weights.

    Args:
        target_means: list or [K, d] tensor of component means
        target_variances: list or [K, d, d] tensor of component covariances
        target_weights: [K] tensor of mixture weights
        n_rep: number of representative points
        device: torch device
        grid_scale: number of stds to extend grid in each direction (grid method)
        method: "grid" or "stratified" or None (auto-select based on dim)

    Returns:
        y_rep: [n_rep, d] tensor of representative points
        w_rep: [n_rep] tensor of normalized weights
    """
    # Normalize inputs
    if isinstance(target_means, list):
        means = torch.stack([m.flatten() for m in target_means]).to(device).float()
    else:
        means = target_means.reshape(-1, target_means.shape[-1]).to(device).float()

    if isinstance(target_variances, list):
        covs = torch.stack([v.float() for v in target_variances]).to(device)
    else:
        covs = target_variances.to(device).float()
    # Ensure covs are [K, d, d]
    if covs.dim() == 2:
        covs = torch.stack([torch.diag(c) for c in covs])

    if isinstance(target_weights, list):
        weights = torch.tensor(target_weights, device=device, dtype=torch.float32)
    else:
        weights = target_weights.to(device).float()
    weights = weights / weights.sum()

    d = means.shape[1]

    if method is None:
        method = "grid" if d <= 2 else "stratified"

    if method == "grid":
        return _grid_representatives(means, covs, weights, n_rep, grid_scale, device)
    else:
        return _stratified_representatives(means, covs, weights, n_rep, device)


def _grid_representatives(means, covs, weights, n_rep, grid_scale, device):
    """
    Deterministic grid with weights = g(y_i) * cell_volume, then normalized.
    For d=1: uniform 1D grid. For d=2: uniform 2D grid.
    """
    d = means.shape[1]

    # Compute bounding box from all components
    stds = torch.sqrt(torch.diagonal(covs, dim1=1, dim2=2))  # [K, d]
    lo = (means - grid_scale * stds).min(dim=0).values
    hi = (means + grid_scale * stds).max(dim=0).values

    if d == 1:
        n_grid = n_rep
        y_vals = torch.linspace(lo[0].item(), hi[0].item(), n_grid, device=device)
        y_grid = y_vals.unsqueeze(1)  # [n_grid, 1]
    elif d == 2:
        n_per_dim = max(int(np.sqrt(n_rep)), 2)
        g1 = torch.linspace(lo[0].item(), hi[0].item(), n_per_dim, device=device)
        g2 = torch.linspace(lo[1].item(), hi[1].item(), n_per_dim, device=device)
        g1g, g2g = torch.meshgrid(g1, g2, indexing='ij')
        y_grid = torch.stack([g1g.flatten(), g2g.flatten()], dim=1)  # [n_per_dim^2, 2]
    else:
        raise ValueError("Grid method only supports d <= 2")

    # Evaluate MoG density at grid points
    log_probs = _mog_log_pdf(y_grid, means, covs, weights)
    w = torch.exp(log_probs)

    # Cell volume (uniform grid)
    if d == 1:
        cell_vol = (hi[0] - lo[0]) / (len(y_grid) - 1)
    else:
        cell_vol = ((hi[0] - lo[0]) / (n_per_dim - 1)) * ((hi[1] - lo[1]) / (n_per_dim - 1))

    w = w * cell_vol

    # Filter out negligible points (keep top n_rep by weight)
    if len(y_grid) > n_rep:
        _, topk_idx = torch.topk(w, n_rep)
        y_grid = y_grid[topk_idx]
        w = w[topk_idx]

    # Normalize weights
    w = w / w.sum()

    return y_grid, w


def _stratified_representatives(means, covs, weights, n_rep, device):
    """
    Stratified sampling: allocate n_k = round(w_k * n_rep) samples to each
    component k, then sample from N(mu_k, Sigma_k). Weights are all equal
    (since samples are drawn proportional to mixture weights).
    """
    K = means.shape[0]

    # Allocate samples per component
    n_per = (weights * n_rep).floor().int()
    remainder = n_rep - n_per.sum().item()
    # Distribute remainder to components with largest fractional parts
    fracs = (weights * n_rep) - n_per.float()
    _, top_idx = torch.topk(fracs, int(remainder))
    for idx in top_idx:
        n_per[idx] += 1

    points = []
    for k in range(K):
        nk = n_per[k].item()
        if nk <= 0:
            continue
        dist = MultivariateNormal(means[k], covs[k])
        pts = dist.sample((nk,))
        points.append(pts)

    y_rep = torch.cat(points, dim=0).to(device)

    # Equal weights (stratified sampling already accounts for mixture proportions)
    w_rep = torch.ones(y_rep.shape[0], device=device) / y_rep.shape[0]

    return y_rep, w_rep


def _mog_log_pdf(y, means, covs, weights):
    """
    Compute log p_G(y) for a MoG with given parameters.

    Args:
        y: [N, d] query points
        means: [K, d] component means
        covs: [K, d, d] component covariances
        weights: [K] mixture weights

    Returns:
        [N] log-density values
    """
    K = means.shape[0]
    log_components = []

    for k in range(K):
        dist = MultivariateNormal(means[k], covs[k])
        log_p_k = dist.log_prob(y)  # [N]
        log_components.append(torch.log(weights[k]) + log_p_k)

    # log-sum-exp over components
    log_components = torch.stack(log_components, dim=0)  # [K, N]
    return torch.logsumexp(log_components, dim=0)  # [N]


def get_noised_representatives(y_rep, w_rep, t, target_means, target_variances,
                               target_weights, device="cpu"):
    """
    Get the "noised" version of G at time t along the conditional OT path.

    For conditional OT:  y_t = (1-t) * z + t * y_1,  where z ~ N(0,I), y_1 ~ G
    So G_t is a MoG with:
        mean_k(t)  = t * mu_k
        cov_k(t)   = t^2 * Sigma_k + (1-t)^2 * I

    Instead of re-sampling, we noise the existing representatives:
        y_t_i = t * y_rep_i + (1-t) * z_i,   z_i ~ N(0, I)

    Args:
        y_rep: [n, d] representative points from G
        w_rep: [n] weights
        t: scalar or [n] tensor of times in [0, 1]
        target_means, target_variances, target_weights: G parameters (unused here
            but kept for potential analytical noising)
        device: torch device

    Returns:
        y_t: [n, d] noised representative points
        w_rep: [n] weights (unchanged)
    """
    d = y_rep.shape[1]
    z = torch.randn_like(y_rep, device=device)

    if isinstance(t, (int, float)):
        t_val = t
    else:
        t_val = t.view(-1, 1) if t.dim() == 1 else t

    y_t = t_val * y_rep + (1 - t_val) * z
    return y_t, w_rep
