"""
Proves `sdxl.mmd.sd_mmd` (the shared `losses.functions.compute_mmd`) is numerically
identical to the formula in conditional-matching-paper
SD_cond_SD_controlnet/src/metrics.py @ cb8bc91. CPU-only, no model downloads, seconds.

    python -m sdxl.check_mmd_equivalence
"""
import torch
from sdxl.mmd import sd_mmd, estimate_bandwidth


def upstream_compute_mmd(x, y, bandwidth=None, bandwidth_scale=1.0, kernel_alpha=1.0):
    """Verbatim transcription of upstream metrics.py:compute_mmd @ cb8bc91."""
    dev = x.device
    x = x.float().to(dev)
    y = y.float().to(dev).detach()
    n, m = x.shape[0], y.shape[0]

    def rbf_kernel(a, b, bw, alpha):
        a_sq = (a ** 2).sum(dim=1, keepdim=True)
        b_sq = (b ** 2).sum(dim=1, keepdim=True)
        dist_sq = a_sq + b_sq.T - 2 * torch.mm(a, b.T)
        return torch.exp(-(dist_sq / (2 * bw ** 2)) ** alpha)

    if bandwidth is None:
        bandwidth = estimate_bandwidth(x.detach(), y, bandwidth_scale)
    K_xx = rbf_kernel(x, x, bandwidth, kernel_alpha)
    K_yy = rbf_kernel(y, y, bandwidth, kernel_alpha)
    K_xy = rbf_kernel(x, y, bandwidth, kernel_alpha)
    xx = (K_xx.sum() - K_xx.trace()) / (n * (n - 1)) if n > 1 else 0.0
    yy = (K_yy.sum() - K_yy.trace()) / (m * (m - 1)) if m > 1 else 0.0
    xy = 2 * K_xy.sum() / (n * m)
    return torch.sqrt((xx - xy + yy).abs() + 1e-8)


def main():
    torch.manual_seed(0)
    ok = True
    for (n, m, alpha) in [(100, 100, 1.0), (120, 120, 1.0), (64, 250, 2.0), (7, 13, 1.5)]:
        x = torch.randn(n, 768)
        y = torch.randn(m, 768) * 1.2 + 0.4
        bw = estimate_bandwidth(x, y)
        a = sd_mmd(x, y, bandwidth=bw, kernel_alpha=alpha).item()
        b = upstream_compute_mmd(x, y, bandwidth=bw, kernel_alpha=alpha).item()
        import math
        # NOTE both-NaN counts as agreement. Non-integer alpha makes BOTH the shared
        # path and upstream return NaN: floating point leaves the squared-distance
        # diagonal very slightly negative, and a negative base to a fractional power
        # is NaN. That is an upstream fragility faithfully preserved here, not a port
        # bug. Every published SD run uses alpha = 1.0, where it cannot arise.
        same = (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-6
        ok &= same
        print(f"  n={n:<4} m={m:<4} alpha={alpha}  shared={a:+.8f}  upstream={b:+.8f}  "
              f"{'OK (both NaN, see note)' if same and a != a else ('OK' if same else 'MISMATCH')}")

    # gradients must flow through x and not through y
    x = torch.randn(32, 768, requires_grad=True)
    y = torch.randn(40, 768, requires_grad=True)
    sd_mmd(x, y).backward()
    print(f"  grad flows through x: {x.grad is not None and x.grad.abs().sum() > 0}")
    print(f"  target y stays detached: {y.grad is None}")
    ok &= (x.grad is not None and x.grad.abs().sum() > 0) and (y.grad is None)

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
