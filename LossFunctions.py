import torch
from torch import nn

#### MMD Loss #####

class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, device='cpu'):
        super().__init__()
        self.device = device
        self.bandwidth_multipliers = (mul_factor ** (torch.arange(n_kernels, device=device) - n_kernels // 2))
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        X = X.to(self.device)
        L2_distances = torch.cdist(X, X, p=2) ** 2
        bandwidth = self.get_bandwidth(L2_distances)
        scaled = L2_distances[None, ...] / (bandwidth * self.bandwidth_multipliers[:, None, None])
        return torch.exp(-scaled).sum(dim=0)


class MMDLoss(nn.Module):
    def __init__(self, kernel=None, device='cpu'):
        super().__init__()
        self.device = device
        self.kernel = kernel if kernel else RBF(device=device)
        self.to(device)

    def forward(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY



# Example code
# mmd_loss = MMDLoss(kernel=RBF())
#
# for epoch in range(epochs):
#
#     mu_list_cond, Sigma_list_cond = L2_Distance_pytorch.compute_conditionals(mu_list, Sigma_list, x_optim)
#     alpha_cond = L2_Distance_pytorch.compute_alpha(mu_list, Sigma_list, alpha, x_optim)
#
#     target_samples = generate_mog_samples(numb_sample, mu_list_cond, Sigma_list_cond, alpha_cond)
#     samples = generate_mog_samples(numb_sample, xi, S, beta, tau=10.0)
#     # Use the MMD loss implementation
#     loss = mmd_loss(target_samples, samples)


# Compute distance
def sliced_wasserstein_distance(X, Y, num_projections=100, p=2):
        n, d = X.shape
        X,Y = X.to(torch.float32), Y.to(torch.float32)

        projections = torch.randn(num_projections, d, device=X.device)
        projections = projections / torch.norm(projections, dim=1, keepdim=True)  # normalize


        swd = 0.0
        for proj in projections:
            X_proj = (X @ proj).cpu().numpy()
            Y_proj = (Y @ proj).cpu().numpy()

            X_proj.sort()
            Y_proj.sort()

            swd += ot.wasserstein_1d(X_proj, Y_proj, p=p)

        return swd / num_projections

def compute_swd_distance_optimization(mu_list, Sigma_list, alpha,x_star,x_optim,n_samples,num_projections=100, p=2):
    #param for target
    mog_means, mog_variances = compute_conditionals(mu_list, Sigma_list, x_star.reshape(-1, 1))
    weights = compute_alpha(mu_list, Sigma_list, alpha, x_star.reshape(-1, 1))

    # param for optimized
    mog_means_optim, mog_variances_optim = compute_conditionals(mu_list, Sigma_list, x_optim.reshape(-1, 1))
    weights_optim = compute_alpha(mu_list, Sigma_list, alpha, x_optim.reshape(-1, 1))

    # sample theo and optim
    samples_tehoretical = generate_mog_samples_not_differentiable(n_samples, mog_means, mog_variances, weights)
    samples_optimized = generate_mog_samples_not_differentiable(n_samples,mog_means_optim, mog_variances_optim, weights_optim)

    dist = sliced_wasserstein_distance(samples_tehoretical, samples_optimized, num_projections=100, p=2)

    return dist