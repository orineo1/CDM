import torch
import numpy as np
from scipy.optimize import fsolve
import torch.nn.functional as F,

########### Mixture of Gaussian's
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
    Compute the Probability Density Function (PDF) of a Mixture of Gaussians (MoG)
    using PyTorch's built-in multivariate normal distribution.

    Args:
        x: Points at which to evaluate the PDF, shape [num_points, dim]
        means: List of mean vectors, each of shape [dim]
        covariances: List of covariance matrices, each of shape [dim, dim]
        weights: Mixture weights, shape [num_components]. Default: equal weights

    Returns:
        PDF values at points x, shape [num_points]
    """
    from torch.distributions.multivariate_normal import MultivariateNormal

    device = x.device  # Get current device
    components = len(means)
    batch_size = x.shape[0]

    # Move means and covariances to same device as x
    means = [m.to(device) for m in means]
    covariances = [c.to(device) for c in covariances]
    if weights is None:
        weights = torch.ones(components, device=device) / components  # Equal weights
    else:
        weights = weights.to(device)

    pdf = torch.zeros(batch_size, device=device)

    for i in range(components):
        # Create multivariate normal distribution for this component
        mvn = MultivariateNormal(means[i], covariances[i])

        # Evaluate log_prob and convert to prob
        component_pdf = torch.exp(mvn.log_prob(x))

        # Add weighted component to mixture
        pdf += weights[i] * component_pdf
    if log:
        pdf = torch.log(pdf + 1e-10)  # Add epsilon to avoid log(0)

    return pdf


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


