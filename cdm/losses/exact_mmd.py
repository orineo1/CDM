"""Closed-form MMD^2 between the induced conditional P(Y | X = x) of a Gaussian mixture and a
Gaussian-mixture target G, with a multi-bandwidth Gaussian kernel k(u) = sum_l exp(-|u|^2 / h_l).

For Y ~ N(a, A), Y' ~ N(b, B) independent and k_h(u) = exp(-|u|^2 / h):
    E k_h(Y - Y') = det(I + 2 (A + B) / h)^(-1/2) exp(-1/2 (a - b)^T (A + B + h/2 I)^(-1) (a - b)).
The loss is differentiable in x through the component means, the conditional covariances and the
mixture weights pi_k(x); a pathwise estimator through samples only sees the first. Used as an
oracle ("--sampler exactloss" in find_xstar.py) to test whether the sample-based gradient is the
bottleneck of the search.
"""
import torch

from cdm.utils.dist import compute_conditionals, compute_alpha


def _gauss_kernel_mean(a, A, b, B, h):
    d = a.shape[0]
    S = A + B
    eye = torch.eye(d, dtype=a.dtype, device=a.device)
    diff = (a - b).reshape(-1)
    M = S + 0.5 * h * eye
    quad = diff @ torch.linalg.solve(M, diff)
    return torch.linalg.det(eye + 2.0 * S / h) ** -0.5 * torch.exp(-0.5 * quad)


def _cross(m1, S1, w1, m2, S2, w2, hs):
    tot = 0.0
    for a, A, p in zip(m1, S1, w1):
        for b, B, q in zip(m2, S2, w2):
            tot = tot + p * q * sum(_gauss_kernel_mean(a.reshape(-1), A, b.reshape(-1), B, h) for h in hs)
    return tot


class ExactConditionalMMD:
    def __init__(self, mu_list, Sigma_list, alpha, target, n_kernels=5, mul_factor=2.0):
        self.mu_list = [torch.as_tensor(m).double() for m in mu_list]
        self.Sigma_list = [torch.as_tensor(S).double() for S in Sigma_list]
        self.alpha = torch.as_tensor(alpha).double()
        tm, tv, tw = target
        self.tm = [torch.as_tensor(m).double().reshape(-1) for m in tm]
        self.tv = [torch.as_tensor(v).double() for v in tv]
        self.tw = torch.as_tensor(tw).double()
        d = self.tm[0].shape[0]
        self.tv = [v if v.dim() == 2 else torch.diag(v.reshape(-1)) if v.numel() == d else v.reshape(d, d) for v in self.tv]
        # bandwidth scale: mean pairwise squared distance of target draws = 2 tr Cov(G)
        mean = sum(w * m for w, m in zip(self.tw, self.tm))
        cov = sum(w * (v + torch.outer(m - mean, m - mean)) for w, m, v in zip(self.tw, self.tm, self.tv))
        s0 = float(2.0 * torch.trace(cov))
        self.hs = [s0 * mul_factor ** (l - n_kernels // 2) for l in range(n_kernels)]
        self.gg = _cross(self.tm, self.tv, self.tw, self.tm, self.tv, self.tw, self.hs)

    def __call__(self, x):
        x = x.reshape(-1).double().cpu()
        mu_p, S_p = compute_conditionals(self.mu_list, self.Sigma_list, x)
        w_p = compute_alpha(self.mu_list, self.Sigma_list, self.alpha, x)
        mu_p = [m.reshape(-1) for m in mu_p]
        pp = _cross(mu_p, S_p, w_p, mu_p, S_p, w_p, self.hs)
        pg = _cross(mu_p, S_p, w_p, self.tm, self.tv, self.tw, self.hs)
        return pp - 2.0 * pg + self.gg


class IdentitySampler(torch.nn.Module):
    """Stand-in 'sampler' for the exact-loss oracle: returns its condition, so the inner estimator
    hands x_hat_0 itself to the loss, which evaluates ExactConditionalMMD at it."""
    condition_on = 0

    def sample(self, nsamples, condition_x=None, device="cpu", **kw):
        return condition_x, None, None
