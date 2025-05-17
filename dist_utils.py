import torch
import numpy as np
from scipy.optimize import fsolve
import torch.nn.functional as F
from scipy.stats import qmc
from torch.distributions import Normal, MultivariateNormal
from tqdm import tqdm

########### Mixture of Gaussian's ###########
#### Sampling ####
def sample_univariate_gaussian(mean, variance):
    """
    Sample from univariate Gaussian using reparameterization trick

    Args:
        mean: tensor of shape [1] or scalar
        variance: tensor of shape [1] or scalar

    Returns:
        sample: tensor of shape [1]
    """
    std = torch.sqrt(variance).squeeze()
    epsilon = torch.randn(1, dtype=mean.dtype)
    return mean.squeeze() + std * epsilon

def sample_multivariate_gaussian(mean, covariance):
    """
    Sample from multivariate Gaussian using reparameterization trick

    Args:
        mean: tensor of shape [D]
        covariance: tensor of shape [D, D]

    Returns:
        sample: tensor of shape [D]
    """
    # Ensure covariance matrix is positive definite
    eps = 1e-6 * torch.eye(covariance.size(0), dtype=covariance.dtype)
    L = torch.linalg.cholesky(covariance + eps)

    # Generate standard normal noise
    epsilon = torch.randn(mean.size(), dtype=mean.dtype)

    # Apply reparameterization trick
    return mean + torch.matmul(L, epsilon)

def generate_mog_samples(n_samples, xi, S, beta, tau=7.5):
    """
    Generate samples from mixture of Gaussians using reparameterization trick

    Args:
        n_samples: number of samples to generate
        xi: list of means (each can be scalar or vector)
        S: list of variances/covariance matrices
        beta: mixture weights
        tau: temperature for Gumbel-Softmax
    """
    # Ensure xi and S are tensors
    if isinstance(xi, list):
        xi = torch.stack(xi)
    if isinstance(S, list):
        S = torch.stack(S)

    # Get dimensions
    is_multivariate = len(xi.shape) > 1 and xi.shape[-1] > 1

    # Compute logits from beta
    logits = torch.log(beta)

    # Sample component assignments using Gumbel-Softmax
    component_weights = F.gumbel_softmax(logits, tau=tau, hard=False)

    # Sample component indices
    component_idx = torch.multinomial(component_weights, n_samples, replacement=True)
    means = xi[component_idx]
    vars_or_covs = S[component_idx]

    # Generate samples using appropriate sampling function
    samples = []
    for mean, var_or_cov in zip(means, vars_or_covs):
        if is_multivariate:
            sample = sample_multivariate_gaussian(mean, var_or_cov)
        else:
            sample = sample_univariate_gaussian(mean, var_or_cov)
        samples.append(sample)

    samples = torch.stack(samples)
    return samples


def generate_mog_samples_not_differentiable(num_samples, means, variances, weights=None):
    """
    Generate samples from a Mixture of Gaussians (MoG).
    """
    components = len(means)
    if weights is None:
        weights = torch.ones(components) / components

    # Ensure weights sum to 1
    weights = weights / weights.sum()

    samples = []
    for _ in range(num_samples):
        component = torch.multinomial(weights, 1).item()
        sample = torch.normal(means[component], variances[component]**0.5, size=(1,))
        samples.append(sample)
    return torch.cat(samples, dim=0).unsqueeze(1)


def mog_pdf(x, means, variances, weights=None):
    """
    Compute the Probability Density Function (PDF) of a Mixture of Gaussians (MoG).
    """
    components = len(means)
    if weights is None:
        weights = torch.ones(components) / components  # Equal weights by default

    pdf = torch.zeros_like(x)
    for mean, var, weight in zip(means, variances, weights):
        pdf += weight * torch.exp(-0.5 * ((x - mean)**2) / var) / (torch.sqrt(2 * torch.pi * var))
    return pdf

def mog_multivariate_pdf(x, means, covariances, weights=None, log=False):
    """
    Compute PDF or log-PDF of a Mixture of Gaussians (1D or multivariate).

    Args:
        x: Tensor of shape [num_points, dim]
        means: List of tensors, each shape [dim]
        covariances: List of tensors, shape [dim] for 1D or [dim, dim] for multivariate
        weights: Tensor of shape [num_components]
        log: Return log-probabilities if True

    Returns:
        Tensor of shape [num_points]
    """
    device = x.device
    num_points, dim = x.shape
    num_components = len(means)

    # Move everything to correct device
    means = [m.to(device) for m in means]
    covariances = [c.to(device) for c in covariances]
    if weights is None:
        weights = torch.ones(num_components, device=device) / num_components
    else:
        weights = weights.to(device)

    total_prob = torch.zeros(num_points, device=device)

    for i in range(num_components):
        if dim == 1:
            # Univariate case
            mean = means[i].view(-1)
            std = covariances[i].sqrt().view(-1)
            dist = Normal(mean, std)
            comp_log_prob = dist.log_prob(x.squeeze(-1))  # shape: [num_points]
        else:
            # Multivariate case
            dist = MultivariateNormal(means[i], covariances[i])
            comp_log_prob = dist.log_prob(x)  # shape: [num_points]

        total_prob += weights[i] * torch.exp(comp_log_prob)

    if log:
        return torch.log(total_prob + 1e-10)  # Avoid log(0)
    else:
        return total_prob



def log_likelihood(samples, means, variances, weights):
    """
    Compute the log likelihood of the Mixture of Gaussians (MoG) given samples.
    """
    pdf_values = mog_pdf(samples, means, variances, weights)
    return torch.log(pdf_values + 1e-10).sum()

# Function to calculate the CDF of the MoG
def mog_cdf(x, means, variances, weights=None):
    """
    Compute the CDF of a Mixture of Gaussians (MoG).
    """
    components = len(means)
    if weights is None:
        weights = torch.ones(components, device=x.device) / components

    cdf = torch.zeros_like(x)
    for mean, var, weight in zip(means, variances, weights):
        mean = torch.tensor(mean, device=x.device, dtype=torch.float32)
        var = torch.tensor(var, device=x.device, dtype=torch.float32)
        weight = torch.tensor(weight, device=x.device, dtype=torch.float32)
        cdf += weight * 0.5 * (1 + torch.erf((x - mean) / torch.sqrt(2 * var + 1e-6)))
    return cdf

def generate_gmm_parameters(d: int, num_components: int, seed: int = 42, epsilon: float = 1e-4):
    """
    Generate random parameters for a Gaussian Mixture Model (GMM).

    Args:
        d (int): Number of dimensions.
        num_components (int): Number of Gaussian components.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (mu_list, Sigma_list, alpha) where
            - mu_list: List of mean vectors.
            - Sigma_list: List of covariance matrices.
            - alpha: Mixture weights tensor.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Generate random means for each component
    mu_list = [torch.randn(d, dtype=torch.float64) * 3 for _ in range(num_components)]

    # Generate random covariance matrices (positive semi-definite)
    Sigma_list = []
    for _ in range(num_components):
        A = torch.randn(d, d, dtype=torch.float64)  # Random matrix
        Sigma = A @ A.T  # Create a symmetric positive semi-definite covariance matrix
        # Add a small epsilon to the diagonal to make it positive-definite and improve conditioning
        Sigma += torch.eye(d, dtype=torch.float64) * epsilon  # Regularization term

        Sigma_list.append(Sigma)

    # Mixture weights (equal weights)
    alpha = torch.full((num_components,), 1 / num_components, dtype=torch.float64)

    return mu_list, Sigma_list, alpha

# Example usage
# d = 5  # Number of dimensions
# num_components = 2  # Number of Gaussian components
# mu_list, Sigma_list, alpha = generate_gmm_parameters(d, num_components)
#
# # Print the generated parameters
# print("Means:", mu_list)
# print("Covariance Matrices:", Sigma_list)
# print("Mixture Weights:", alpha)

#### Compute conditionals of MoG ####

def split_mean_cov(mu: torch.Tensor, Sigma: torch.Tensor, x_cond: torch.Tensor, cond_indices: list = None) -> tuple:
    """
    Split the mean and covariance matrix for GMM conditioning.
    Args:
        mu (torch.Tensor): Mean vector of shape (d,)
        Sigma (torch.Tensor): Covariance matrix of shape (d, d)
        x_cond (torch.Tensor): Conditioning values
        cond_indices (list, optional): Indices for conditioning. Defaults to last n indices.
    Returns:
        tuple: (mu_1, mu_2, Sigma_11, Sigma_12, Sigma_21, Sigma_22)
    """
    # Get dimensions and device
    d = mu.shape[0]
    device = mu.device

    # Set default cond_indices if not provided
    if cond_indices is None:
        cond_indices = list(range(0, x_cond.shape[0]))

    # Convert indices to torch tensors for indexing, ensuring they're on the same device
    uncond_indices = torch.tensor([i for i in range(d) if i not in cond_indices], device=device)
    cond_indices = torch.tensor(cond_indices, device=device)

    # Split mean
    mu_1 = mu.index_select(0, uncond_indices)
    mu_2 = mu.index_select(0, cond_indices)

    # Split covariance using fancy indexing
    Sigma_11 = Sigma.index_select(0, uncond_indices).index_select(1, uncond_indices)
    Sigma_12 = Sigma.index_select(0, uncond_indices).index_select(1, cond_indices)
    Sigma_21 = Sigma.index_select(0, cond_indices).index_select(1, uncond_indices)
    Sigma_22 = Sigma.index_select(0, cond_indices).index_select(1, cond_indices)

    return mu_1, mu_2, Sigma_11, Sigma_12, Sigma_21, Sigma_22


def conditional_gaussian(mu: torch.Tensor, Sigma: torch.Tensor, x_cond: torch.Tensor,
                         cond_indices: list = None) -> tuple:
    """
    Calculate conditional Gaussian distribution given observations.
    Args:
        mu (torch.Tensor): Mean vector
        Sigma (torch.Tensor): Covariance matrix
        x_cond (torch.Tensor): Conditioning values
        cond_indices (list, optional): Indices for conditioning
    Returns:
        tuple: (conditional mean, conditional covariance)
    """
    # Ensure x_cond is on the same device as mu
    x_cond = x_cond.to(mu.device)
    mu_1, mu_2, Sigma_11, Sigma_12, Sigma_21, Sigma_22 = split_mean_cov(mu, Sigma, x_cond, cond_indices)
    # Compute conditional mean
    # Reshape for matrix multiplication
    mu_1 = mu_1.reshape(-1, 1)
    mu_2 = mu_2.reshape(-1, 1)
    x_cond = x_cond.reshape(-1, 1)
    # Calculate using matrix operations
    Sigma_22_inv = torch.inverse(Sigma_22)  # Keep original implementation
    mu_cond = mu_1 + Sigma_12 @ Sigma_22_inv @ (x_cond - mu_2)
    # Compute conditional covariance
    Sigma_cond = Sigma_11 - Sigma_12 @ Sigma_22_inv @ Sigma_21
    return mu_cond, Sigma_cond


def compute_conditionals(mu_list: list, Sigma_list: list, x_cond: torch.Tensor) -> tuple:
    """
    Calculate conditional distributions for all GMM components.
    Args:
        mu_list (list): List of mean vectors
        Sigma_list (list): List of covariance matrices
        x_cond (torch.Tensor): Conditioning values
    Returns:
        tuple: (conditional means, conditional covariances)
    """
    # Determine device from first mean tensor if available
    device = mu_list[0].device if len(mu_list) > 0 else x_cond.device
    # Ensure x_cond is on the right device
    x_cond = x_cond.to(device)
    condi_mu = []
    condi_sigma = []
    for mu, Sigma in zip(mu_list, Sigma_list):
        # Ensure tensors are on the same device
        mu = mu.to(device)
        Sigma = Sigma.to(device)
        mu_cond, Sigma_cond = conditional_gaussian(mu, Sigma, x_cond)
        condi_mu.append(mu_cond)
        condi_sigma.append(Sigma_cond)
    # Stack tensors along first dimension
    condi_mu = torch.stack(condi_mu)
    condi_sigma = torch.stack(condi_sigma)
    return condi_mu, condi_sigma


def compute_mean_cov(mu_list, Sigma_list, x_cond):
    """
    A function that arranges the output of split_mean_cov in a convenient way to work within a dictionary
    """
    # Initialize an empty dictionary to hold results
    results = {}
    # Determine device from first mean tensor if available
    device = mu_list[0].device if len(mu_list) > 0 else x_cond.device
    # Ensure x_cond is on the right device
    x_cond = x_cond.to(device)
    # Loop through each index in mu_list
    for i in range(len(mu_list)):
        # Ensure tensors are on the same device
        mu = mu_list[i].to(device)
        Sigma = Sigma_list[i].to(device)
        # Call the split_mean_cov function to get the values
        mu_1i, mu_2i, Sigma_11i, Sigma_12i, Sigma_21i, Sigma_22i = split_mean_cov(mu, Sigma, x_cond)

        # Store the results in the dictionary using i as the key
        results[i] = {
            'mu_1': mu_1i,
            'mu_2': mu_2i,
            'Sigma_11': Sigma_11i,
            'Sigma_12': Sigma_12i,
            'Sigma_21': Sigma_21i,
            'Sigma_22': Sigma_22i
        }

    return results


def compute_alpha(mu_list: list, Sigma_list: list, alpha: torch.Tensor, x_cond: torch.Tensor,
                  eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate updated GMM weights given conditioning values, avoiding numerical underflow.

    Args:
        mu_list (list): List of mean vectors
        Sigma_list (list): List of covariance matrices
        alpha (torch.Tensor): Initial mixture weights
        x_cond (torch.Tensor): Conditioning values
        eps (float): Small value to prevent log(0)

    Returns:
        torch.Tensor: Updated mixture weights
    """
    # Get device from alpha tensor
    device = alpha.device

    # Ensure x_cond is on the same device
    x_cond = x_cond.to(device).flatten()

    num = torch.zeros(len(alpha), device=device)
    log_probs = torch.zeros(len(alpha), device=device)

    for i in range(len(alpha)):
        # Ensure the tensors are on the same device
        mu = mu_list[i].to(device)
        Sigma = Sigma_list[i].to(device)

        # Split components for the i-th Gaussian
        _, mu_2, _, _, _, Sigma_22 = split_mean_cov(mu, Sigma, x_cond)
        diff = x_cond - mu_2
        Sigma_22_inv = torch.linalg.inv(Sigma_22)

        # Exponent computation (avoid large negative values)
        exponent = -0.5 * diff.T @ Sigma_22_inv @ diff
        exponent = torch.clamp(exponent, min=-1000)  # Prevent extreme negative values

        # Determinant computation (avoid log(0))
        det_Sigma_22 = torch.linalg.det(Sigma_22)
        det_Sigma_22 = torch.clamp(det_Sigma_22, min=eps)  # Ensure non-zero determinant

        norm_const = -0.5 * len(x_cond) * torch.log(
            torch.tensor(2 * torch.pi, device=device)) - 0.5 * torch.log(det_Sigma_22)

        log_probs[i] = torch.log(torch.clamp(alpha[i], min=eps)) + norm_const + exponent

        if torch.isnan(log_probs[i]):
            raise ValueError(f"log_probs[{i}] is NaN! Stopping execution.")

    log_probs_max = torch.max(log_probs)
    log_probs = log_probs - log_probs_max  # Subtract max for numerical stability

    probs = torch.exp(log_probs)
    probs = probs / (probs.sum() + eps)  # Normalize
    return probs

#### Compute L1 Distance - MoG ####
def compute_L1_distance_dynamic(means_p_raw, variances_p_raw, weights_p_raw,
                                means_q_raw, variances_q_raw, weights_q_raw,
                                scale_factor=3.0,
                                initial_num_points=100,
                                max_error=1e-3,
                                max_iters=7):
    """
    Compute the L1 distance between two Mixture of Gaussians distributions.
    Supports arbitrary dimensions. Uses grid integration for D <= 3, and Sobol-based QMC for D > 3.
    Auto-corrects dimensional formatting for 1D/2D inputs.
    """

    def fix_dimensions(means, variances, weights):
        # Convert means to shape [K, D]
        if isinstance(means, list):
            means = torch.cat([m.reshape(1, -1) for m in means], dim=0)
        elif means.ndim == 3 and means.shape[-2] == 1:
            means = means.squeeze(-2)  # squeeze middle dim if shape is [K, 1, D] or [K, D, 1]
        elif means.ndim > 2:
            means = means.squeeze()

        # Convert variances to shape [K, D, D]
        if isinstance(variances, list):
            variances = torch.stack([v if v.ndim == 2 else v.reshape(1, 1) for v in variances], dim=0)
        elif variances.ndim == 3 and variances.shape[1] == 1 and variances.shape[2] == 1:
            variances = variances.squeeze(1).squeeze(1).unsqueeze(-1).unsqueeze(-1)  # go to [K, 1, 1]

        # Ensure weights is tensor
        weights = torch.tensor(weights, dtype=means.dtype) if not isinstance(weights, torch.Tensor) else weights

        return means, variances, weights

    means_p, variances_p, weights_p = fix_dimensions(means_p_raw, variances_p_raw, weights_p_raw)
    means_q, variances_q, weights_q = fix_dimensions(means_q_raw, variances_q_raw, weights_q_raw)
    D = means_p.shape[1]  # Dimension

    def compute_bounds():
        stds_p = torch.sqrt(torch.diagonal(variances_p, dim1=1, dim2=2))  # [N, D]
        stds_q = torch.sqrt(torch.diagonal(variances_q, dim1=1, dim2=2))  # [M, D]
        all_means = torch.cat([means_p, means_q], dim=0)
        all_stds = torch.cat([stds_p, stds_q], dim=0)
        lower_bounds = (all_means - scale_factor * all_stds).min(dim=0).values
        upper_bounds = (all_means + scale_factor * all_stds).max(dim=0).values
        return lower_bounds, upper_bounds

    def compute_integral_grid(num_points, lower_bounds, upper_bounds):
        grids = [torch.linspace(lower_bounds[d], upper_bounds[d], num_points) for d in range(D)]
        mesh = torch.meshgrid(*grids, indexing='ij')
        points = torch.stack([m.reshape(-1) for m in mesh], dim=-1)  # [num_points**D, D]

        p_vals = mog_multivariate_pdf(points, means_p, variances_p, weights_p)
        q_vals = mog_multivariate_pdf(points, means_q, variances_q, weights_q)
        diff = torch.abs(p_vals - q_vals)

        delta = [(upper_bounds[d] - lower_bounds[d]) / (num_points - 1) for d in range(D)]
        cell_volume = torch.prod(torch.tensor(delta))
        return diff.sum() * cell_volume

    def compute_integral_qmc(num_samples, lower_bounds, upper_bounds):
        sampler = qmc.Sobol(d=D, scramble=True)
        samples = sampler.random_base2(m=int(np.log2(num_samples)))
        lower = lower_bounds.detach().numpy()
        upper = upper_bounds.detach().numpy()
        scale = upper - lower
        points_np = lower + samples * scale
        points = torch.tensor(points_np, dtype=means_p.dtype)

        p_vals = mog_multivariate_pdf(points, means_p, variances_p, weights_p)
        q_vals = mog_multivariate_pdf(points, means_q, variances_q, weights_q)
        diff = torch.abs(p_vals - q_vals)

        volume = torch.prod(upper_bounds - lower_bounds)
        return diff.mean() * volume

    # Adaptive integration loop
    lower_bounds, upper_bounds = compute_bounds()
    prev_result = None
    num_points = initial_num_points

    for _ in tqdm(range(max_iters)):
        if D <= 1:
            result = compute_integral_grid(num_points, lower_bounds, upper_bounds)
        else:
            result = compute_integral_qmc(num_points, lower_bounds, upper_bounds)

        if prev_result is not None and torch.abs(result - prev_result) < max_error:
            break

        prev_result = result
        num_points *= 2

    return result

def warpper_L1_distance(x_optim,mu_list, Sigma_list, alpha, mog_means, mog_variances, weights):
    mu_temp, Sigma_temp = compute_conditionals(mu_list, Sigma_list, x_optim)
    temp_alpha = compute_alpha(mu_list, Sigma_list, alpha, x_optim)
    l1_distance = compute_L1_distance_dynamic(
        mu_temp, Sigma_temp, temp_alpha,
        mog_means, mog_variances, weights,
        max_error=1e-6
    )
    return l1_distance


