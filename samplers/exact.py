"""Oracle conditional sampler: exact draws from P(Y | X = x) of a known Gaussian mixture.

Used as f_phi in place of a trained network to measure what the search algorithm achieves
when the sampler has no error. Differentiable in x by reparametrisation: with responsibilities
r_k(x) the component k is drawn (no gradient through the discrete choice), and
y = mu_y_k + A_k (x - mu_x_k) + L_k eps with A_k = Sigma_yx Sigma_xx^{-1} and L_k the Cholesky
factor of the conditional covariance, so dy/dx = A_k for the drawn component.
"""
import math

import torch


class ExactConditionalSampler(torch.nn.Module):
    def __init__(self, mu_list, Sigma_list, alpha, condition_on):
        super().__init__()
        c = int(condition_on)
        self.condition_on = c
        mus = [m.float().view(-1) for m in mu_list]
        Sigs = [S.float() for S in Sigma_list]
        self.nfeatures = mus[0].numel() - c
        self.mu_x = torch.stack([m[:c] for m in mus])                     # (K, dx)
        self.mu_y = torch.stack([m[c:] for m in mus])                     # (K, dy)
        Sxx = torch.stack([S[:c, :c] for S in Sigs])                      # (K, dx, dx)
        Sxy = torch.stack([S[:c, c:] for S in Sigs])                      # (K, dx, dy)
        Syy = torch.stack([S[c:, c:] for S in Sigs])                      # (K, dy, dy)
        Sxx_inv = torch.linalg.inv(Sxx)
        self.A = Sxy.transpose(1, 2) @ Sxx_inv                            # (K, dy, dx)
        C = Syy - self.A @ Sxy                                            # (K, dy, dy)
        C = 0.5 * (C + C.transpose(1, 2)) + 1e-8 * torch.eye(self.nfeatures)
        self.L = torch.linalg.cholesky(C)
        self.log_alpha = torch.log(alpha.float().view(-1))
        self.Sxx_inv = Sxx_inv
        self.Sxx_logdet = torch.logdet(Sxx)
        self.register_buffer("_dev", torch.zeros(0))

    def _to(self, device):
        for k in ("mu_x", "mu_y", "A", "L", "log_alpha", "Sxx_inv", "Sxx_logdet"):
            setattr(self, k, getattr(self, k).to(device))

    def responsibilities(self, x):
        """r_k(x) proportional to alpha_k N(x; mu_x_k, Sxx_k); x: (n, dx)."""
        d = x[:, None, :] - self.mu_x[None]                               # (n, K, dx)
        maha = torch.einsum("nki,kij,nkj->nk", d, self.Sxx_inv, d)
        logp = self.log_alpha[None] - 0.5 * (maha + self.Sxx_logdet[None] + x.shape[1] * math.log(2 * math.pi))
        return torch.softmax(logp, -1)

    def sample(self, nsamples, condition_x=None, device="cpu", **kw):
        self._to(device)
        x = condition_x.to(device).float()
        if x.shape[0] != nsamples:
            x = x.view(1, -1).repeat(nsamples, 1)
        with torch.no_grad():
            k = torch.multinomial(self.responsibilities(x.detach()), 1).view(-1)   # component per row
        eps = torch.randn(nsamples, self.nfeatures, device=device)
        y = self.mu_y[k] + torch.einsum("nij,nj->ni", self.A[k], x - self.mu_x[k]) + torch.einsum("nij,nj->ni", self.L[k], eps)
        return y, None, None
