"""Configuration for the unified guided-diffusion optimizer.

Every axis on which the historical optimizer variants differed is an explicit
field here. Strategy callables are used instead of string flags so that each
legacy variant is reproduced *exactly*, including its RNG call order.

See guidance/core.py for how each field is consumed, and Optimization.py for
the wrappers that reconstruct each legacy variant's behaviour.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any

import math
import torch


# --------------------------------------------------------------------------
# Initialization strategies
# --------------------------------------------------------------------------

def init_randn(nfeatures: int, device: str) -> torch.Tensor:
    """N(0, I) start. Used by DPS, LGD_CM, LGD_cfg, LGD_latent."""
    return torch.randn(nfeatures, device=device, requires_grad=True)


def init_zeros(nfeatures: int, device: str) -> torch.Tensor:
    """All-zeros start. Used by the original optimize_LGD ONLY.

    NOTE: this is a faithful reproduction of legacy behaviour, not a
    recommendation. The paper (Alg. 1 line 1) states x_T ~ N(0, I), and
    optimize_LGD_CM carries the comment "CHANGED: zeros -> randn", i.e. the
    zeros init was already recognised as wrong and fixed in the newer variant
    but never backported. Do not copy this into new experiments.
    """
    return torch.zeros(nfeatures, device=device, requires_grad=True)


# --------------------------------------------------------------------------
# Monte-Carlo perturbation scales
#
# All three legacy formulas are kept verbatim. They disagree with each other
# and with Alg. 3 of the paper; see revision/CHANGES.md.
# --------------------------------------------------------------------------

def perturb_none(model_uncond, t, i) -> Optional[torch.Tensor]:
    """No MC perturbation: condition directly on pred_x0 (DPS)."""
    return None


def perturb_beta_ratio(model_uncond, t, i) -> torch.Tensor:
    """r_t = beta_t / sqrt(1 + beta_t^2); perturbation std = r_t.  (optimize_LGD)"""
    current_var = model_uncond.betas[t]
    return current_var / torch.sqrt(1 + current_var ** 2)


def perturb_beta_prod_ratio_sq(model_uncond, t, i) -> torch.Tensor:
    """r_t = sqrt(1-abar_t) / sqrt(1 + (1-abar_t)); perturbation std = r_t^2.

    (optimize_LGD_CM). Note the square: the caller applies r_t**2.
    """
    beta_prod_t = 1 - model_uncond.scheduler.alphas_cumprod[t]
    current_var = torch.sqrt(beta_prod_t)
    return current_var / torch.sqrt(1 + current_var ** 2)


def perturb_sqrt_beta_prod(model_uncond, t, i) -> torch.Tensor:
    """r_t = sqrt(1 - abar_t); perturbation std = r_t^2 = (1 - abar_t).

    (optimize_LGD_cfg / MNIST).
    """
    beta_prod_t = 1 - model_uncond.scheduler.alphas_cumprod[t]
    return torch.sqrt(beta_prod_t)


# --------------------------------------------------------------------------
# Loss aggregation over the n_mc perturbations
# --------------------------------------------------------------------------

def aggregate_logsumexp(neg_losses) -> torch.Tensor:
    """-log( (1/n) sum_j exp(-L_j) ).

    `neg_losses` is the list of -L_j, matching legacy call sites which append
    `-loss_val`. Equivalent to a softmin-weighted average of the per-
    perturbation gradients.

    CAVEAT (see revision/algorithm_plan.md): when L_j is itself a noisy
    estimate, Jensen makes this objective ~ L - (1/2)Var(L), i.e. it rewards
    moving toward regions where the loss estimator is noisy. Kept as-is for
    equivalence; ablate against `aggregate_mean` afterwards.
    """
    n = len(neg_losses)
    return -torch.logsumexp(torch.stack(neg_losses), dim=0) + math.log(n)


def aggregate_mean(neg_losses) -> torch.Tensor:
    """Plain mean of the losses. Not used by any legacy variant; provided as
    the ablation arm against aggregate_logsumexp."""
    return -torch.stack(neg_losses).mean()


def aggregate_single(neg_losses) -> torch.Tensor:
    """Single loss, no aggregation (DPS, which has no MC loop)."""
    assert len(neg_losses) == 1, "aggregate_single requires exactly one loss"
    return -neg_losses[0]


# --------------------------------------------------------------------------
# Update rules
# --------------------------------------------------------------------------

def update_ddim_minus_grad(x_t, x_t_minus_1, grad, step, **kw):
    """x_{t-1} - step * grad.  (LGD: step=1.0; DPS: step=beta_t; CM/cfg: schedule)"""
    return x_t_minus_1.detach().clone() - step * grad


@dataclass
class GuidanceConfig:
    """All knobs of the unified guided-diffusion optimizer.

    Defaults reproduce `optimize_LGD_CM`, the most general legacy variant.
    """

    # --- sampling budget ---
    n_cond: int = 250          # samples drawn from f_phi per candidate
    n_mc: int = 3              # MC perturbations of pred_x0 per outer step
    final_n_cond: Optional[int] = None   # samples for the final loss report
    compute_final_loss: bool = True      # DPS returns best-so-far and skips this

    # --- strategies ---
    init_fn: Callable = init_randn
    unsqueeze_init: bool = True          # DPS does NOT unsqueeze; others do
    perturb_fn: Callable = perturb_beta_prod_ratio_sq
    perturb_square: bool = True          # whether std is r_t or r_t**2
    aggregate_fn: Callable = aggregate_logsumexp
    update_fn: Callable = update_ddim_minus_grad

    # --- step size ---
    # Called as step_size_fn(t, i) -> float/tensor.
    step_size_fn: Callable = lambda t, i: 1.0
    guidance_cutoff_t: Optional[int] = None   # zero guidance for t < cutoff (MNIST: 200)

    # --- timestep iteration ---
    # "range": range(diffusion_steps-1, 0, -1)   (DPS, LGD, LGD_latent)
    # "model": enumerate(model_uncond.timesteps) (LGD_CM)
    timestep_source: str = "model"

    # --- ddim step signature: legacy classes differ ---
    # "condition_x": model.sample_ddim_step(x, t, condition_x=None, device=, eta=0.0)
    # "indexed":     model.sample_ddim_step(x, t, i, device=, eta=0.0)
    ddim_step_style: str = "indexed"

    # --- conditional-sampler plumbing ---
    # legacy variants differ in whether the condition is repeated to
    # (n_cond, -1) before being passed to model_cond.sample
    repeat_condition: bool = False
    # some samplers return (samples, ...) and need slicing off the condition
    condition_on_slice: bool = False
    cond_output_transform: Optional[Callable] = None

    # --- target sampling ---
    # Called as target_generator(n, device) -> target samples.
    # NOTE: every legacy variant redraws the target INSIDE the MC loop, i.e.
    # n_mc * T times per run. Alg. 3 line 1 of the paper says it is drawn once
    # outside the outer loop. `resample_target_per_mc=True` reproduces legacy.
    target_generator: Optional[Callable] = None
    resample_target_per_mc: bool = True

    # --- bookkeeping ---
    device: str = "cuda"
    progress: bool = False
    seed: Optional[int] = None
    tag: str = ""              # free-form label recorded with results

    def describe(self) -> dict:
        """Flat, JSON-serialisable summary for the results store."""
        out = {}
        for k, v in self.__dict__.items():
            if callable(v):
                out[k] = getattr(v, "__name__", repr(v))
            elif isinstance(v, torch.Tensor):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out
