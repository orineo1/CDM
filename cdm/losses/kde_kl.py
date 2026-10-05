"""KDE estimate of the KL divergence between a target and a generated sample set (GB-FESO's loss).

`_mixed_sqdist`, `_log_kde_eval` and `kde_kl_target_to_generated` are copied unchanged from GB-FESO,
Zick & Mendels, "Gradient-Based Inverse Design of Free-Energy Landscapes with Diffusion Models"
(arXiv:2607.06421), file `GB-FESO/loss.py` of
https://github.com/MendelsResearchGroup/Gradient-Based-Free-Energy-Surface-Optimization
(commit 4e021a4), under its license:

    MIT License

    Copyright (c) 2026 Mendels Research Group

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

`KDEKLLoss` is ours: it puts the function behind this repository's `loss(pred, target)` signature.
Their default treats coordinates as angles (all of them for 2-D data, the last two otherwise); vector
data with Euclidean coordinates must pass `angle_indices=()`, which `KDEKLLoss` does by default.
"""
import math
from typing import Optional, Sequence, Union

import torch


def _mixed_sqdist(
    a: torch.Tensor,
    b: torch.Tensor,
    angle_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Pairwise squared distance for mixed Euclidean/periodic coordinates.

    For angle-only tensors [theta123, theta234], both dimensions are wrapped.
    For the geometry tensors [r12, r23, r34, theta123, theta234], only the
    last two angle dimensions are wrapped.
    """
    d = a[:, None, :] - b[None, :, :]
    ndim = a.shape[-1]

    if angle_indices is None:
        angle_indices = tuple(range(ndim)) if ndim == 2 else (ndim - 2, ndim - 1)

    if len(angle_indices) > 0:
        idx = [i if i >= 0 else ndim + i for i in angle_indices]
        d_angle = d[..., idx]
        d[..., idx] = torch.atan2(torch.sin(d_angle), torch.cos(d_angle))

    return d.square().sum(dim=-1)


def _log_kde_eval(
    eval_x: torch.Tensor,
    centers: torch.Tensor,
    bandwidth: Union[float, torch.Tensor],
    angle_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    if torch.is_tensor(bandwidth):
        if torch.any(bandwidth <= 0):
            raise ValueError("bandwidth values must be positive")
        bw = bandwidth.to(device=eval_x.device, dtype=eval_x.dtype)
        d = eval_x[:, None, :] - centers[None, :, :]
        ndim = eval_x.shape[-1]
        if angle_indices is None:
            angle_indices = tuple(range(ndim)) if ndim == 2 else (ndim - 2, ndim - 1)
        idx = [i if i >= 0 else ndim + i for i in angle_indices]
        if idx:
            d_angle = d[..., idx]
            d[..., idx] = torch.atan2(torch.sin(d_angle), torch.cos(d_angle))
        sqdist = ((d / bw) ** 2).sum(dim=-1)
        log_norm = math.log(centers.shape[0]) + torch.log(bw).sum() + 0.5 * ndim * math.log(2.0 * math.pi)
        return torch.logsumexp(-0.5 * sqdist, dim=1) - log_norm

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    d = eval_x.shape[-1]
    sqdist = _mixed_sqdist(eval_x, centers, angle_indices=angle_indices)
    log_k = -0.5 * sqdist / (bandwidth * bandwidth)
    log_norm = math.log(centers.shape[0]) + d * math.log(bandwidth) + 0.5 * d * math.log(2.0 * math.pi)
    return torch.logsumexp(log_k, dim=1) - log_norm


def kde_kl_target_to_generated(
    target_rad: torch.Tensor,
    generated_rad: torch.Tensor,
    bandwidth: Union[float, torch.Tensor] = 0.18,
    kl_lambda: float = 1.0,
    angle_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Differentiable KDE KL-style loss for angle-only or 5D geometry samples.

    Inputs can be [theta123, theta234] angle tensors or
    [r12, r23, r34, theta123, theta234] geometry tensors. Angle columns are
    compared periodically; distance columns are compared Euclideanly.
    """
    log_p_target = _log_kde_eval(target_rad, target_rad.detach(), bandwidth, angle_indices).detach()
    log_q_target = _log_kde_eval(target_rad, generated_rad, bandwidth, angle_indices)
    forward = (log_p_target - log_q_target).mean()

    if float(kl_lambda) == 0.0:
        return forward

    log_q_generated = _log_kde_eval(generated_rad, generated_rad.detach(), bandwidth, angle_indices)
    log_p_generated = _log_kde_eval(generated_rad, target_rad.detach(), bandwidth, angle_indices)
    reverse = (log_q_generated - log_p_generated).mean()

    return forward + float(kl_lambda) * reverse


class KDEKLLoss:
    """`loss(pred, target)`: GB-FESO's KDE-KL with the generated samples `pred` and the target samples `target`.
    Euclidean coordinates by default (`angle_indices=()`)."""

    def __init__(self, bandwidth=0.5, kl_lambda=1.0, angle_indices=()):
        self.bandwidth, self.kl_lambda, self.angle_indices = bandwidth, kl_lambda, angle_indices

    def __call__(self, pred, target):
        return kde_kl_target_to_generated(target, pred, bandwidth=self.bandwidth, kl_lambda=self.kl_lambda,
                                          angle_indices=self.angle_indices)
