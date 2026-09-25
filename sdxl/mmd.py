"""
sdxl/mmd.py -- the SD experiment's MMD, built ON TOP of the shared estimator.

THE DIVISION OF LABOUR, which is the point of this file:

  * The ESTIMATOR is shared. The U-statistic arithmetic -- drop the diagonal from
    each within-sample block, divide by n(n-1) and m(m-1), keep 1/(nm) on the cross
    term -- lives in `losses.functions.MMDLoss` and is CALLED, not copied. That class
    was a V-statistic until 2026-09-25; it is now the U-statistic, which is what the
    SD pipeline always used and what produced the paper's Tables 3 and 15.

  * The KERNEL is SD-specific, so it is defined here. The shared `losses.RBF` is a
    5-bandwidth mixture (mul_factor 2.0) with a MEAN-based bandwidth. SD uses a
    SINGLE bandwidth from the MEDIAN heuristic on the cross-distances, raised to a
    generalised exponent `alpha`:  k(a,b) = exp(-(||a-b||^2 / 2bw^2)^alpha).
    These are different kernels and give different numbers. Substituting the shared
    RBF here would silently change the published SD results, so it is not done.
    `MMDLoss` takes an injected kernel, which is exactly the seam we need.

  * The sqrt WRAPPER is SD-specific and stays here. SD reports
    sqrt(|MMD^2_U| + 1e-8). The absolute value is load-bearing: an unbiased estimator
    legitimately returns small negative values and the paper's MMD column is a
    square root. `mmd_report` is for DISPLAY ONLY -- the loss and its gradient use
    the plain squared statistic, as upstream does.

Upstream equivalent: SD_cond_SD_controlnet/src/metrics.py at
orineo1/conditional-matching-paper @ cb8bc91.
"""

import numpy as np
import torch
from torch import nn

from losses.functions import MMDLoss   # <- the shared U-statistic estimator


def _as_2d_float(t):
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    t = t.float()
    return t.reshape(t.shape[0], -1) if t.dim() > 2 else t


def estimate_bandwidth(x, y, bandwidth_scale=1.0):
    """Median-heuristic bandwidth on the x-y cross distances (detached)."""
    dev = x.device
    ss = min(1000, x.shape[0], y.shape[0])
    with torch.no_grad():
        xs, ys = x[:ss].detach(), y[:ss].detach()
        d = (xs ** 2).sum(1, keepdim=True) + (ys ** 2).sum(1, keepdim=True).T - 2 * xs @ ys.T
        d = d[d > 0]
        bw = torch.sqrt(torch.median(d) / 2) if d.numel() > 0 else torch.tensor(1.0, device=dev)
    return bw.detach() * bandwidth_scale


class GeneralizedRBF(nn.Module):
    """exp(-(||a-b||^2 / 2bw^2)^alpha) over one stacked matrix, at a FIXED bandwidth.

    Fixed rather than data-derived on purpose: SD estimates the bandwidth from the
    x-vs-y cross distances, but the shared MMDLoss hands its kernel the CONCATENATION
    of X and Y and so cannot express that split. The bandwidth is therefore computed
    by `sd_mmd` before the kernel is built, and passed in.
    """

    def __init__(self, bandwidth, alpha=1.0):
        super().__init__()
        self.bandwidth = bandwidth
        self.alpha = alpha

    def forward(self, Z):
        sq = (Z ** 2).sum(1, keepdim=True)
        dist_sq = sq + sq.T - 2 * Z @ Z.T
        return torch.exp(-(dist_sq / (2 * self.bandwidth ** 2)) ** self.alpha)


def sd_mmd(x, y, bandwidth=None, bandwidth_scale=1.0, kernel_alpha=1.0):
    """Unbiased squared MMD (MMD^2_U) in CLIP space, via the shared estimator.

    Gradients flow through `x`; `y` (the target) is detached, as upstream.
    Returns the SQUARED statistic -- no square root. Use `mmd_report` for display.
    """
    x = _as_2d_float(x)
    y = _as_2d_float(y).detach()
    y = y.to(x.device)
    if bandwidth is None:
        bandwidth = estimate_bandwidth(x.detach(), y, bandwidth_scale)
    kernel = GeneralizedRBF(bandwidth, kernel_alpha)
    return MMDLoss(kernel=kernel, device=x.device)(x, y)


def mmd_report(mmd_squared):
    """sqrt(|MMD^2_U| + 1e-8) -- DISPLAY ONLY. Never feed this into a loss."""
    if not torch.is_tensor(mmd_squared):
        mmd_squared = torch.tensor(mmd_squared)
    return torch.sqrt(mmd_squared.abs() + 1e-8)


def compute_witness_scores(x, y, bandwidth=None, bandwidth_scale=1.0, kernel_alpha=1.0):
    """Per-sample witness w(x_l) = mean_i k(x_l,x_i) - mean_j k(x_l,y_j).

    Used by the 'witness' back-selection rule. The paper's runs use 'uniform'
    selection, so this is present for completeness, not for Tables 3/15.
    """
    with torch.no_grad():
        x = _as_2d_float(x).detach()
        y = _as_2d_float(y).detach().to(x.device)
        if bandwidth is None:
            bandwidth = estimate_bandwidth(x, y, bandwidth_scale)

        def k(a, b):
            d = (a ** 2).sum(1, keepdim=True) + (b ** 2).sum(1, keepdim=True).T - 2 * a @ b.T
            return torch.exp(-(d / (2 * bandwidth ** 2)) ** kernel_alpha)

        scores = k(x, x).mean(dim=1) - k(x, y).mean(dim=1)
    return scores, bandwidth
