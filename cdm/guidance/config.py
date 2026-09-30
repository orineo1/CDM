"""Configuration for the unified guided-diffusion optimizer.

Every axis on which the historical optimizer variants differed is an explicit
field here. Strategy callables are used instead of string flags so that each
legacy variant is reproduced *exactly*, including its RNG call order.

See cdm/guidance/core.py for how each field is consumed.
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


class AdamGuidance:
    """Adam-style guidance update (AdamDPS-like): the guidance push uses bias-corrected first and
    second moments of the loss gradient accumulated across the reverse steps of one trajectory,
        m <- b1 m + (1 - b1) g,   v <- b2 v + (1 - b2) g^2,
        x_{t-1} <- ddim(x_t) - step * m_hat / (sqrt(v_hat) + eps),
    so the push is scale-free in the gradient and `step` is a displacement in x units.
    Moments are kept per trajectory slot (particles pass their index); reset() clears them."""

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.state = {}

    def reset(self):
        self.state = {}

    def __call__(self, x_t, x_t_minus_1, grad, step, slot=0, **kw):
        st = self.state.get(slot)
        if st is None:
            st = {"m": torch.zeros_like(grad), "v": torch.zeros_like(grad), "k": 0}
            self.state[slot] = st
        st["k"] += 1
        st["m"].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        st["v"].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        m_hat = st["m"] / (1 - self.beta1 ** st["k"])
        v_hat = st["v"] / (1 - self.beta2 ** st["k"])
        return x_t_minus_1.detach().clone() - step * m_hat / (v_hat.sqrt() + self.eps)

    __name__ = "AdamGuidance"


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
    # Pure prior sampling: take the DDIM step and skip the guidance block entirely
    # (no Tweedie perturbation, no conditional draws, no target draws, no gradient).
    # This is NOT the same as step_size_fn -> 0: the upstream
    # conditional-matching-paper simulations/src/Optimization.py::optimize_LGD
    # short-circuits at `if zeta == 0.0:` BEFORE the Monte-Carlo loop, so the
    # zeta=0 arm consumes no random numbers and its trajectory differs from a
    # zeta=0 run that still draws the samples and multiplies the gradient by 0.
    skip_guidance: bool = False

    # --- timestep iteration ---
    # "range": range(diffusion_steps-1, 0, -1)   (DPS, LGD, LGD_latent)
    # "model": enumerate(model_uncond.timesteps) (LGD_CM)
    timestep_source: str = "model"

    # --- ddim step: legacy variants differ in signature, and MNIST computes
    #     the step inline against a diffusers scheduler instead of calling the
    #     model's own method. ddim_step_fn, if set, overrides the style. ---
    # "condition_x": model.sample_ddim_step(x, t, condition_x=None, device=, eta=0.0)
    # "indexed":     model.sample_ddim_step(x, t, i, device=, eta=0.0)
    ddim_step_style: str = "indexed"
    # Called as ddim_step_fn(model_uncond, x_t, t, i) -> (x_prev, pred_x0)
    ddim_step_fn: Optional[Callable] = None
    eta: float = 0.0                     # DDIM stochasticity (0 = deterministic, 1 = DDPM); legacy variants use 0

    # --- explicit timestep list (timestep_source="list") ---
    timesteps_list: Optional[Any] = None
    # timestep_source="range": start the reverse process here instead of T-1 (partial trajectory
    # from a noised starting point, as in SDEdit); None keeps the full schedule
    range_t_start: Optional[int] = None
    # MNIST takes one unguided DDIM step after the loop and returns its pred_x0
    final_unguided_step: bool = False

    # extra kwargs forwarded to model_cond.sample (e.g. the consistency
    # multi-step schedule ts=[150., 50., 20., 10., 5., 1.] used by MNIST)
    cond_sample_kwargs: dict = field(default_factory=dict)

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

    # --- conditional-sample schedule over the reverse process ---
    # "constant": n_cond at every step; "linear": n_cond_min at the first step, n_cond at the last
    n_cond_schedule: str = "constant"
    n_cond_min: int = 50

    # --- particle population (optimize_guided_population) ---
    # R trajectories run in lockstep; after each step the particles are weighted by
    # exp(-resample_beta * loss) and resampled when the effective sample size drops below
    # ess_frac * R. resample_beta=None: independent trajectories (no resampling).
    n_particles: int = 1
    resample_beta: Optional[float] = None
    ess_frac: float = 0.5
    resample_every: int = 1              # only consider resampling every k steps
    weight_mode: str = "fresh"           # "fresh": w = exp(-beta L_t); "accumulated": telescoped over steps
    # schedule of the resampling strength over the reverse process (fraction of steps done, 0 -> 1):
    # "constant": resample_beta throughout; "linear": ramps 0 -> resample_beta (annealing: no resampling
    # while the Tweedie estimates are uninformative, full strength at the end)
    resample_schedule: str = "constant"

    # --- post-refinement: plain gradient steps on the plug-in loss from the returned x,
    #     no diffusion; optional prior term lambda * score_theta(x, t=1) ---
    refine_steps: int = 0
    refine_lr: float = 0.05
    refine_lambda: float = 0.0
    refine_optim: str = "adam"           # "adam" or "sgd"
    refine_n_cond: Optional[int] = None  # conditional samples per refinement step (default: n_cond)
    # gradient used by the refinement: "pathwise" = backpropagation through the sampler (default);
    # "zo" = zeroth-order central differences of the sample loss with common random numbers (the same
    # sampler and target draws on both sides), which also sees how the mixture of modes of P(Y|X=x)
    # shifts with x, a change pathwise gradients through samples largely miss
    refine_grad: str = "pathwise"
    zo_eps: float = 0.05                 # finite-difference step (whitened units)
    zo_dirs: int = 0                     # 0: all coordinate directions (2 d loss evaluations); M > 0: M random Gaussian directions

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
