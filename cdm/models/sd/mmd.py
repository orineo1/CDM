"""
cdm/models/sd/mmd.py -- the SD experiment's MMD, which IS the shared one.

`compute_mmd` in `cdm.losses.functions` is the port of the function that produced the
paper's SD tables (conditional-matching-paper SD_cond_SD_controlnet/src/metrics.py).
SD's kernel -- a single median-heuristic bandwidth raised to a generalised exponent
`alpha`, k(a,b) = exp(-(||a-b||^2 / 2bw^2)^alpha) -- is that function's own kernel, so
there is nothing SD-specific left to build here: `sd_mmd` is a thin alias that keeps the
name the ported pipeline code uses.

What remains in this file is the bandwidth helper and the witness scores, which the
pipeline imports separately.

Upstream equivalent: SD_cond_SD_controlnet/src/metrics.py at
orineo1/conditional-matching-paper @ cb8bc91.
"""

import numpy as np
import torch

from cdm.losses.functions import compute_mmd as _shared_compute_mmd


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


def sd_mmd(x, y, bandwidth=None, bandwidth_scale=1.0, kernel_alpha=1.0, cache_target=False):
    """The shared `cdm.losses.functions.compute_mmd`, under the name the SD code uses.

    Gradients flow through `x`; `y` (the target) is detached inside `compute_mmd`.
    Returns what upstream returns: sqrt(|MMD^2| + 1e-8).
    """
    return _shared_compute_mmd(x, y, bandwidth=bandwidth, bandwidth_scale=bandwidth_scale,
                               kernel_alpha=kernel_alpha, cache_target=cache_target)


def mmd_report(mmd_value):
    """Pass-through, kept so the pipeline's `mmd_report(compute_mmd(...))` calls still read.

    `compute_mmd` already returns the square-rooted value that upstream reports, so there
    is nothing left for this to do. Tensors and floats both come back as a tensor.
    """
    if not torch.is_tensor(mmd_value):
        mmd_value = torch.tensor(mmd_value)
    return mmd_value


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
