"""
Likelihood-based optimizer for finding x_hat using a trained flow matching model.

Loss:  L(x) = -sum_i w_i * log p_hat_t(y_t_i | x)

At each gradient step:
  1. Sample t ~ U[0,1]
  2. Noise the representative G points to time t:  y_t = t*y + (1-t)*z
  3. Compute log p_hat_t(y_t | x) by integrating the learned velocity field
     backward from t to 0 and tracking the log-determinant (CNF approach)
  4. Backpropagate through x

The log-likelihood at intermediate time t is computed via a standalone ODE
integrator (torchdiffeq) that does NOT use FM_Solver_Extension_module.py,
thus avoiding its time_grid[0]==1.0 assertion.

Usage:
    from likelihood_optimizer import optimize_likelihood

    x_hat, loss_val = optimize_likelihood(
        model_cond,       # trained FMModel (conditional)
        model_uncond,     # trained FMModel (unconditional, for P(X) sampling)
        target_means, target_variances, target_weights,
        n_rep=200, n_steps=200, lr=0.05, device="cpu",
    )
"""

import math
import time
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Independent, Normal
from torchdiffeq import odeint
from flow_matching.utils import gradient

from representative_points import get_representative_points, get_noised_representatives


def compute_log_likelihood_at_t(model, y_t, t_val, x_cond, step_size=0.02,
                                exact_divergence=True, device="cpu"):
    """
    Compute log p_hat_t(y_t | x) by integrating the learned velocity field
    backward from time t to 0, tracking the log-determinant via the
    instantaneous change of variables formula.

    Uses torchdiffeq.odeint directly (no FM_Solver_Extension_module dependency).

    Args:
        model: trained FMModel with .forward(x, t, x_cond) method
        y_t: [n, d] points at time t
        t_val: scalar, the time level in [0, 1]
        x_cond: [n, cond_dim] or [1, cond_dim] conditioning values
        step_size: ODE solver step size
        exact_divergence: use exact divergence (True) or Hutchinson estimator (False)
        device: torch device

    Returns:
        log_p_t: [n] tensor of log p_hat_t(y_t | x_cond)
    """
    n, d = y_t.shape

    # Standard normal base density
    log_p0 = Independent(
        Normal(torch.zeros(d, device=device), torch.ones(d, device=device)), 1
    ).log_prob

    # Expand x_cond if needed
    if x_cond.shape[0] == 1 and n > 1:
        x_cond_expanded = x_cond.expand(n, -1)
    else:
        x_cond_expanded = x_cond

    # For Hutchinson estimator, fix random projection
    if not exact_divergence:
        z_hutch = (torch.randn_like(y_t) < 0).float() * 2.0 - 1.0

    def dynamics_func(t_ode, states):
        """ODE dynamics: (dy/dt, d(log_det)/dt) = (v(y,t), tr(dv/dy))"""
        yt = states[0]
        with torch.enable_grad():
            yt = yt.detach().requires_grad_(True)

            # Model velocity: v(y, t | x)
            t_batch = t_ode.expand(yt.shape[0])
            vt = model.forward(yt, t_batch, x_cond=x_cond_expanded)

            if exact_divergence:
                # Exact divergence: sum of diagonal of Jacobian
                div = torch.zeros(yt.shape[0], device=device)
                for i in range(d):
                    div += gradient(vt[:, i], yt, create_graph=True)[:, i]
            else:
                # Hutchinson estimator
                vt_dot_z = torch.einsum("ij,ij->i", vt, z_hutch)
                grad_vt_dot_z = gradient(vt_dot_z, yt, create_graph=True)
                div = torch.einsum("ij,ij->i", grad_vt_dot_z, z_hutch)

        return vt, div

    # Time grid from t down to 0
    if t_val <= 1e-6:
        # At t~0, density is just the base density
        return log_p0(y_t)

    n_steps_ode = max(int(t_val / step_size) + 1, 3)
    time_grid = torch.linspace(t_val, 0.0, n_steps_ode, device=device)

    y_init = (y_t, torch.zeros(n, device=device))
    ode_opts = {"step_size": step_size}

    with torch.enable_grad():
        sol, log_det = odeint(
            dynamics_func,
            y_init,
            time_grid,
            method="midpoint",
            options=ode_opts,
            atol=1e-5,
            rtol=1e-5,
        )

    # sol[-1] is the pushed-back point at t=0
    y_0 = sol[-1]
    log_p_base = log_p0(y_0)

    # log p_t(y_t) = log p_0(y_0) + log|det(dy_0/dy_t)|
    # The log_det accumulated along the path gives the total log-det
    log_p_t = log_p_base + log_det[-1]

    return log_p_t


def likelihood_loss(model, y_rep, w_rep, x_cond, t_val,
                    target_means, target_variances, target_weights,
                    step_size=0.02, exact_divergence=True, device="cpu"):
    """
    Compute the weighted negative log-likelihood loss at time t.

    L(x) = -sum_i w_i * log p_hat_t(y_t_i | x)

    Args:
        model: trained FMModel
        y_rep: [n_rep, d] representative points from G
        w_rep: [n_rep] weights
        x_cond: [1, cond_dim] the conditioning value being optimized
        t_val: scalar time in [0, 1]
        target_means/variances/weights: G parameters (for noising)
        step_size: ODE step size
        exact_divergence: bool
        device: torch device

    Returns:
        loss: scalar tensor (negative weighted log-likelihood)
    """
    # Noise representative points to time t
    y_t, w_t = get_noised_representatives(
        y_rep, w_rep, t_val,
        target_means, target_variances, target_weights,
        device=device,
    )

    # Expand x_cond for all representative points
    n_rep = y_t.shape[0]
    if x_cond.dim() == 1:
        x_cond = x_cond.unsqueeze(0)
    x_cond_expanded = x_cond.expand(n_rep, -1)

    # Compute log-likelihood at time t
    log_p = compute_log_likelihood_at_t(
        model, y_t, t_val, x_cond_expanded,
        step_size=step_size, exact_divergence=exact_divergence,
        device=device,
    )

    # Weighted negative log-likelihood
    loss = -torch.sum(w_t * log_p)
    return loss


def optimize_likelihood(model_cond, model_uncond, target_means, target_variances,
                        target_weights, n_rep=200, n_steps=200, lr=0.05,
                        step_size=0.02, exact_divergence=True,
                        t_distribution="uniform", device="cpu",
                        x_init=None, verbose=True):
    """
    Optimize x to minimize the likelihood-based loss.

    At each gradient step:
      1. Sample t ~ U[0,1]  (or from t_distribution)
      2. Noise the G representatives to time t
      3. Compute L(x) = -sum w_i log p_hat_t(y_t_i | x)
      4. Gradient step on x

    Args:
        model_cond: trained conditional FMModel P(Y|X)
        model_uncond: trained unconditional FMModel P(X) (used for init if x_init=None)
        target_means, target_variances, target_weights: G parameters
        n_rep: number of representative points from G
        n_steps: number of optimization steps
        lr: learning rate for Adam
        step_size: ODE step size for likelihood computation
        exact_divergence: use exact or Hutchinson divergence
        t_distribution: "uniform" for t~U[0,1], or "beta" for t~Beta(2,1) (bias toward t=1)
        device: torch device
        x_init: initial x value [1, dim_x] or None (random init)
        verbose: print progress

    Returns:
        best_x: [1, dim_x] best x found
        best_loss: scalar, best loss value
        history: list of dicts with per-step info
    """
    model_cond.eval()

    dim_x = model_cond.condition_on

    # Generate representative points from G (once, at the beginning)
    y_rep, w_rep = get_representative_points(
        target_means, target_variances, target_weights,
        n_rep=n_rep, device=device,
    )

    if verbose:
        print(f"  Generated {y_rep.shape[0]} representative points (dim={y_rep.shape[1]})")

    # Initialize x
    if x_init is not None:
        x = x_init.clone().to(device).float().requires_grad_(True)
    else:
        x = torch.randn(1, dim_x, device=device, dtype=torch.float32, requires_grad=True)

    optimizer = optim.Adam([x], lr=lr)

    best_x = x.detach().clone()
    best_loss = float("inf")
    history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Sample t ~ distribution
        if t_distribution == "uniform":
            t_val = torch.rand(1, device=device).item()
            # Clamp away from 0 to avoid trivial loss
            t_val = max(t_val, 0.05)
        elif t_distribution == "beta":
            t_val = float(torch.distributions.Beta(2.0, 1.0).sample())
            t_val = max(t_val, 0.05)
        else:
            t_val = 0.95  # fixed, near t=1

        # Map x from latent to data space via unconditional model
        # For flow matching: x_data = FM(x_noise) i.e. sample with x as init
        x_data = model_uncond.sample(num_sample=1, y=None, x_init=x)

        # Compute loss
        loss = likelihood_loss(
            model_cond, y_rep, w_rep, x_data, t_val,
            target_means, target_variances, target_weights,
            step_size=step_size, exact_divergence=exact_divergence,
            device=device,
        )

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_x = x.detach().clone()

        history.append({
            "step": step,
            "loss": loss_val,
            "t": t_val,
            "x": x.detach().cpu().squeeze().numpy().copy(),
        })

        if verbose and (step % 20 == 0 or step == n_steps - 1):
            x_np = x.detach().cpu().squeeze().numpy()
            print(f"  [{step:4d}/{n_steps}] t={t_val:.2f}, loss={loss_val:.4f}, x={x_np}")

    # Final x in data space
    with torch.no_grad():
        best_x_data = model_uncond.sample(num_sample=1, y=None, x_init=best_x)

    return best_x_data, best_loss, history


def optimize_likelihood_direct(model_cond, target_means, target_variances,
                               target_weights, n_rep=200, n_steps=200, lr=0.05,
                               step_size=0.02, exact_divergence=True,
                               t_distribution="uniform", device="cpu",
                               x_init=None, verbose=True):
    """
    Optimize x directly in data space (no unconditional model mapping).
    Simpler variant: x is directly the conditioning value.

    This is appropriate when X is low-dimensional and we don't need
    the P(X) prior/regularization from the unconditional model.

    Returns:
        best_x: [1, dim_x] best x found (in data space)
        best_loss: scalar
        history: list of dicts
    """
    model_cond.eval()
    dim_x = model_cond.condition_on

    # Generate representative points from G
    y_rep, w_rep = get_representative_points(
        target_means, target_variances, target_weights,
        n_rep=n_rep, device=device,
    )

    if verbose:
        print(f"  Generated {y_rep.shape[0]} representative points (dim={y_rep.shape[1]})")

    # Initialize x
    if x_init is not None:
        x = x_init.clone().to(device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.requires_grad_(True)
    else:
        x = torch.randn(1, dim_x, device=device, dtype=torch.float32, requires_grad=True)

    optimizer = optim.Adam([x], lr=lr)

    best_x = x.detach().clone()
    best_loss = float("inf")
    history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Sample t ~ distribution
        if t_distribution == "uniform":
            t_val = torch.rand(1, device=device).item()
            t_val = max(t_val, 0.05)
        elif t_distribution == "beta":
            t_val = float(torch.distributions.Beta(2.0, 1.0).sample())
            t_val = max(t_val, 0.05)
        else:
            t_val = 0.95

        # Compute loss directly with x as conditioning
        loss = likelihood_loss(
            model_cond, y_rep, w_rep, x, t_val,
            target_means, target_variances, target_weights,
            step_size=step_size, exact_divergence=exact_divergence,
            device=device,
        )

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_x = x.detach().clone()

        history.append({
            "step": step,
            "loss": loss_val,
            "t": t_val,
            "x": x.detach().cpu().squeeze().numpy().copy(),
        })

        if verbose and (step % 20 == 0 or step == n_steps - 1):
            x_np = x.detach().cpu().squeeze().numpy()
            print(f"  [{step:4d}/{n_steps}] t={t_val:.2f}, loss={loss_val:.4f}, x={x_np}")

    return best_x, best_loss, history
