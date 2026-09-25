"""MNIST -> GuidanceConfig adapter.

The MLGD-F loop itself is NOT re-implemented here. `guidance.core.optimize_guided`
runs it; this module only supplies the MNIST-shaped callables it takes:

  * `ddim_step_fn`    -- MNIST steps a diffusers `DDIMScheduler` inline rather
                         than calling a model method, which is exactly the case
                         `GuidanceConfig.ddim_step_fn` exists for.
  * `perturb_fn`      -- r_t = sqrt(1 - abar_t). `guidance.config.perturb_sqrt_beta_prod`
                         is the same formula but reads `model_uncond.scheduler`;
                         the MNIST UNet is a bare `UNet2DModel` wrapper with no
                         such attribute, so the closure below reads the scheduler
                         the experiment already holds.
  * `step_size_fn`    -- the zeta_t rules of Table 13.
  * `update_fn`       -- `guidance.config.update_ddim_minus_grad` plus the
                         [-1, 1] clamp the MNIST runs used (`--clamp`). There is
                         no `clamp` field on GuidanceConfig; the update callable
                         is the documented extension point for it.
  * `loss_function`   -- SWD from `losses.sliced.SupportAwareSlicedWassersteinLoss`.
                         `MNIST/run_mlgdf.py` carried a private copy of the same
                         computation; it is not ported.

Everything else -- log-mean-exp aggregation over the n_mc Tweedie perturbations,
the guidance cutoff, the explicit timestep list, the trailing unguided DDIM step,
the fresh final-loss evaluation -- is already a field of `GuidanceConfig`.
"""

import torch

from guidance.config import (GuidanceConfig, aggregate_logsumexp,
                             update_ddim_minus_grad)
from losses.circular import angles_to_circular, circular_to_angles
from losses.sliced import SupportAwareSlicedWassersteinLoss

# The iCT multi-step sampling schedule used everywhere in the MNIST experiment.
ICT_TS = [150.0, 50.0, 20.0, 10.0, 5.0, 1.0]

# zeta_t rules. "original" is the rule quoted in the paper's Table 13 caption;
# "double" is the doubled variant that caption assigns to the unimodal target.
STEP_SIZE_MODES = ("original", "dps", "half", "double",
                   "tripleLinear", "doubleLinear", "no_linear")


def compute_step_size(r_t, t, mode):
    """zeta_t as a function of r_t = sqrt(1 - abar_t) and the timestep."""
    if mode == "original":
        return r_t / (1 + r_t ** 2) + 5 * t / 1000
    if mode == "dps":
        return 1.0 / (r_t + 1e-8) * 0.1
    if mode == "half":
        return 0.5 * (r_t / (1 + r_t ** 2) + 5 * t / 1000)
    if mode == "double":
        return 2.0 * (r_t / (1 + r_t ** 2) + 5 * t / 1000)
    if mode == "tripleLinear":
        return r_t / (1 + r_t ** 2) + 15 * t / 1000
    if mode == "doubleLinear":
        return r_t / (1 + r_t ** 2) + 10 * t / 1000
    if mode == "no_linear":
        return r_t / (1 + r_t ** 2)
    raise ValueError(f"unknown step_size_mode: {mode}")


def make_ddim_step_fn(ddim, timesteps):
    """(model, x_t, t, i) -> (x_{t-1}, pred_x0) against a diffusers DDIMScheduler."""
    n = len(timesteps)

    def ddim_step_fn(model_uncond, x_t, t, i):
        residual = model_uncond(x_t, torch.tensor([t], device=x_t.device))
        alpha_t = ddim.alphas_cumprod[t]
        if i < n - 2:
            alpha_t_prev = ddim.alphas_cumprod[timesteps[i + 1]]
        else:
            alpha_t_prev = torch.tensor(1.0)
        beta_t = 1 - alpha_t
        pred_x0 = (x_t - beta_t ** 0.5 * residual) / alpha_t ** 0.5
        x_t_minus_1 = alpha_t_prev ** 0.5 * pred_x0 + (1 - alpha_t_prev) ** 0.5 * residual
        return x_t_minus_1, pred_x0

    return ddim_step_fn


def make_clamped_update(lo=-1.0, hi=1.0):
    """The shared DDIM-minus-gradient update, then clamp to [lo, hi]."""
    def update(x_t, x_t_minus_1, grad, step, **kw):
        return update_ddim_minus_grad(x_t, x_t_minus_1, grad, step, **kw).clamp(lo, hi)
    update.__name__ = f"update_ddim_minus_grad_clamped[{lo},{hi}]"
    return update


def make_swd_loss(device="cpu", n_projections=50):
    """The shared sliced-Wasserstein loss as a plain (X, Y) -> scalar callable."""
    swd = SupportAwareSlicedWassersteinLoss(device=device, n_projections=n_projections)

    def loss_function(pred_samples, target_samples):
        return swd.sliced_wasserstein_distance(
            pred_samples, target_samples, n_projections=n_projections)

    loss_function.__name__ = f"swd_{n_projections}proj"
    return loss_function


def _angle_roundtrip(samples):
    """(cos, sin) -> degrees -> (cos, sin), as the source loop did."""
    return angles_to_circular(circular_to_angles(samples))


def build_config(target, ddim, *, n_cond, n_mc, step_size_mode, device,
                 seed=None, clamp=True, final_n_cond_factor=4, tag=""):
    """Assemble the `GuidanceConfig` for one MNIST MLGD-F run.

    `ddim` is a diffusers `DDIMScheduler` with `set_timesteps` already called.
    """
    timesteps = ddim.timesteps

    def perturb_fn(model_uncond, t, i):
        return torch.sqrt(1 - ddim.alphas_cumprod[t])

    def step_size_fn(t, i):
        r_t = torch.sqrt(1 - ddim.alphas_cumprod[t])
        return compute_step_size(r_t, t, step_size_mode)

    def init_fn(nfeatures, dev):
        return torch.randn(1, 28, 28, device=dev, requires_grad=True)

    return GuidanceConfig(
        n_cond=n_cond,
        n_mc=n_mc,
        final_n_cond=final_n_cond_factor * n_cond,
        compute_final_loss=True,
        init_fn=init_fn,
        unsqueeze_init=True,                       # -> (1, 1, 28, 28)
        perturb_fn=perturb_fn,
        perturb_square=True,                       # x0 + r_t^2 * eps
        aggregate_fn=aggregate_logsumexp,
        update_fn=make_clamped_update() if clamp else update_ddim_minus_grad,
        step_size_fn=step_size_fn,
        guidance_cutoff_t=target.guidance_cutoff_t,
        timestep_source="list",
        timesteps_list=timesteps,
        ddim_step_fn=make_ddim_step_fn(ddim, timesteps),
        final_unguided_step=True,
        cond_sample_kwargs={"ts": list(ICT_TS)},
        repeat_condition=False,
        cond_output_transform=_angle_roundtrip,
        target_generator=target.generator(),
        resample_target_per_mc=True,
        device=device,
        seed=seed,
        tag=tag or f"mnist_{target.name}",
    )
