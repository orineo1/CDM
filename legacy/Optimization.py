
import importlib
from models import diffusion as Diffusion
from losses.functions import MMDLoss,RBF
from utils.dist import generate_mog_samples,generate_mog_samples_not_differentiable
from models.samplers import consistency as ConsistencyModels
from models.samplers import flow_matching as FlowMatching
from models.samplers.consistency import ConsistencyModel
from utils import dist as dist_utils
import torch.nn.functional as F
import torch
import torch.optim as optim_cond
from tqdm.notebook import tqdm

from guidance import (
    GuidanceConfig, optimize_guided,
    init_randn, init_zeros,
    perturb_none, perturb_beta_ratio, perturb_beta_prod_ratio_sq,
    perturb_sqrt_beta_prod,
    aggregate_logsumexp, aggregate_single,
)


def optimize_LGD_CM(model_uncond, model_cond, target_generator, loss_function,
                    step_size_vector, cond_output_transform=None,
                    nsamples=250, num_x_t=3, device="cuda", FLAG=False,
                    r_t_override=None, T=None, final_nsamples=250):
    """Promoted from PocThesis.ipynb cell 7.

    This was already the most general legacy variant -- it takes the target
    generator, loss, step-size schedule and output transform as arguments.
    It now delegates to guidance.core like the others.

    Note `step_size_vector` is indexed by the timestep `t`, not by the loop
    counter, matching the original.
    """
    if final_nsamples is None:
        final_nsamples = 4 * nsamples

    cfg = GuidanceConfig(
        n_cond=nsamples,
        n_mc=num_x_t,
        final_n_cond=final_nsamples,
        init_fn=init_randn,
        unsqueeze_init=True,
        perturb_fn=perturb_beta_prod_ratio_sq,
        perturb_square=True,                      # std = r_t**2
        aggregate_fn=aggregate_logsumexp,
        step_size_fn=lambda t, i: step_size_vector[t],
        timestep_source="model",
        ddim_step_style="indexed",
        repeat_condition=False,                   # x0_sample passed directly
        condition_on_slice=False,
        cond_output_transform=cond_output_transform,
        target_generator=target_generator,
        compute_final_loss=True,
        device=device,
        progress=FLAG,
        tag="legacy:optimize_LGD_CM",
    )
    out = optimize_guided(model_uncond, model_cond, loss_function, cfg)
    return out["x_final"], out["x_final"], out["final_loss"]


######### Diffusion
def optimize_DPS(model_uncond, model_cond, mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
                 nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda"):
    """DEPRECATED wrapper -- kept so existing notebooks keep working unchanged.

    Delegates to guidance.core.optimize_guided with the configuration that
    reproduces this function's historical behaviour. The frozen original is in
    Optimization_legacy.py and is what tests/test_equivalence.py compares to.
    """
    if isinstance(mog_means, torch.Tensor):
        mog_means = mog_means.to(device)
    if isinstance(mog_variances, torch.Tensor):
        mog_variances = mog_variances.to(device)
    if isinstance(weights, torch.Tensor):
        weights = weights.to(device)

    cfg = GuidanceConfig(
        n_cond=nsamples,
        n_mc=1,                                   # DPS has no MC loop
        init_fn=init_randn,
        unsqueeze_init=False,                     # DPS does not unsqueeze
        perturb_fn=perturb_none,                  # conditions on pred_x0 directly
        aggregate_fn=aggregate_single,
        step_size_fn=lambda t, i: model_uncond.betas[t].to(device),
        timestep_source="range",
        ddim_step_style="condition_x",
        repeat_condition=True,
        condition_on_slice=(not CM),
        target_generator=lambda n, dev: generate_mog_samples(
            n, mog_means, mog_variances, weights, tau=10.0).to(dev),
        compute_final_loss=False,                 # DPS returns best-so-far
        device=device,
        progress=True,
        tag="legacy:optimize_DPS",
    )
    out = optimize_guided(model_uncond, model_cond, MMDLoss(kernel=RBF()), cfg)
    return out["x_final"], out["best_x0"], out["best_loss"]


#### Optimize LGD

import math
import torch
from torch import optim
from tqdm import tqdm
# Optimize LGD using Monte Carlo-based guidance
def optimize_LGD(model_uncond, model_cond, mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
                 nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda", FLAG=False,
                 zeta=1.0):
    """DEPRECATED wrapper -- kept so existing notebooks keep working unchanged.

    Reproduces the historical behaviour exactly, including one quirk:
      * init is ZEROS, not N(0,I) (the paper's Alg. 1 says N(0,I); the newer
        optimize_LGD_CM carries a "CHANGED: zeros -> randn" comment). Fix this
        only AFTER tests/test_equivalence.py passes.

    `zeta` is the guidance strength of upstream
    conditional-matching-paper simulations/src/Optimization.py::optimize_LGD:
    the update is `x_{t-1} - zeta * grad`, and `zeta == 0.0` short-circuits the
    whole Monte-Carlo block (pure prior sampling, no random numbers consumed).
    The default 1.0 is the historical behaviour.
    """
    cfg = GuidanceConfig(
        n_cond=nsamples,
        n_mc=num_x_t,
        init_fn=init_zeros,                       # historical; see docstring
        unsqueeze_init=True,
        perturb_fn=perturb_beta_ratio,
        perturb_square=False,                     # std = r_t, not r_t**2
        aggregate_fn=aggregate_logsumexp,
        step_size_fn=lambda t, i: zeta,
        skip_guidance=(zeta == 0.0),
        timestep_source="range",
        ddim_step_style="condition_x",
        repeat_condition=True,
        condition_on_slice=(not CM),
        target_generator=lambda n, dev: generate_mog_samples_not_differentiable(
            n, mog_means, mog_variances, weights),
        compute_final_loss=True,
        device=device,
        progress=FLAG,
        tag="legacy:optimize_LGD",
    )
    out = optimize_guided(model_uncond, model_cond, MMDLoss(kernel=RBF()), cfg)
    return out["x_final"], out["x_final"], out["final_loss"]


def optimize_LGD_latent(model_uncond, model_cond, mog_means, mog_variances, weights,scaler2,encoder2, mu_list, Sigma_list, alpha,
               nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda", FLAG=False,factor_for_final_loss_calc=4):

  mmd_loss = MMDLoss(kernel=RBF())
  best_mmd_loss = float("inf")
  best_x0_sample = None

  # Initialize starting point
  # x_t = torch.zeros(model_uncond.nfeatures, device=device, requires_grad=True)
  x_t = torch.randn(model_uncond.nfeatures, device=device, requires_grad=True)

  x_t = x_t.unsqueeze(0)
  pbar = tqdm(range(model_uncond.diffusion_steps - 1, 0, -1)) if FLAG else range(model_uncond.diffusion_steps - 1, 0, -1)

  for t in pbar:
      x_t = x_t.detach().clone().requires_grad_(True)

      # optimizer = optim.AdamW([x_t], lr=.05, weight_decay=0.01)
      optimizer = optim.Adam([x_t], lr=0.1, betas=(0.5, 0.999), eps=1e-4)

      optimizer.zero_grad()

      # DDIM step (get x_{t-1} and predicted x0)
      x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)
      current_var = model_uncond.betas[t].to(device)
      r_t = current_var / torch.sqrt(1 + current_var ** 2)

      log_mean_exp_loss = torch.tensor([0])
      # Monte Carlo estimation of expected loss over x0 ~ N(pred_x0, r_t^2 * I)
      losses = []
      for j in range(num_x_t):
          x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
          condition = x0_sample.view(1, -1).repeat(nsamples, 1)
          target_samples, _, _ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
          if not CM:
              target_samples = target_samples[:, model_cond.condition_on:]

          # Generate MoG samples in encoded space
          mog_samples_raw = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)
          mog_samples_scaled = torch.tensor(scaler2.transform(mog_samples_raw.cpu().numpy()), device=device, dtype=torch.float32)
          mog_samples = encoder2(mog_samples_scaled)

          loss_val = mmd_loss(target_samples, mog_samples)
          losses.append(-loss_val)  # negative for log-mean-exp

          if loss_val.item() < best_mmd_loss:
              best_mmd_loss = loss_val.item()
              best_x0_sample = x0_sample.detach().clone()

      # Compute final log-mean-exp loss
      log_mean_exp_loss = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
      log_mean_exp_loss.backward()

      # Optimizer step
      optimizer.step()

      if FLAG:
          grad_norm = torch.norm(x_t.grad).item() if x_t.grad is not None else 0
          pbar.set_description(f"Step {t}| grad_norm:{grad_norm:.6f} | x_t:{x_t}|x0_sample:{x0_sample} | loss_val:{loss_val:.6f}|log_mean_exp_loss: {log_mean_exp_loss.item():.4f}")

  # compute the final loss - proxy for the real loss
  condition = x_t.view(1, -1).repeat(nsamples, 1)
  target_samples, _, _ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
  mog_samples_raw = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)
  mog_samples_scaled = torch.tensor(scaler2.transform(mog_samples_raw.cpu().numpy()), device=device, dtype=torch.float32)
  mog_samples = encoder2(mog_samples_scaled)
  final_loss =  mmd_loss(target_samples, mog_samples)

  # Final output
  x_t_final = x_t.detach().clone()
  torch.cuda.empty_cache() if device == "cuda" else None
  return x_t_final, x_t_final, best_mmd_loss, best_x0_sample,final_loss

######### FlowMatching

#### DFlow
def find_samples_to_kl(numb_points, mog_means, mog_variances, weights, device="cuda"):
    if isinstance(mog_means, (list, tuple)):
        mog_means_tensor = torch.stack(mog_means, dim=0)
    elif isinstance(mog_means, torch.Tensor):
        mog_means_tensor = mog_means.unsqueeze(0) if mog_means.dim() == 0 else mog_means
    else:
        raise TypeError("mog_means must be a list, tuple, or tensor")

    if mog_means_tensor.dim() > 2 and mog_means_tensor.shape[-1] == 1:
        mog_means_tensor = mog_means_tensor.squeeze(-1)

    mog_means_tensor = mog_means_tensor.to(device)
    mog_samples = generate_mog_samples(numb_points, mog_means, mog_variances, weights).to(device)

    if mog_samples.dim() < mog_means_tensor.dim():
        mog_means_tensor = mog_means_tensor.squeeze(-1)

    Target = torch.cat([mog_means_tensor, mog_samples], dim=0).to(torch.float32)
    return Target





def compute_loss(vf_y_cond_x, vf_X, x_1, samples, loss_method, mog_means, mog_variances, weights, device):
    if loss_method == "MMD":
        mmd_loss = MMDLoss(kernel=RBF())

        # Create conditioning tensor on device
        x_cond = x_1.reshape(-1).to(device) * torch.ones(
            samples.shape[0], vf_y_cond_x.condition_on, dtype=torch.float32, device=device
        )
        target_samples = vf_y_cond_x.sample(samples.shape[0], x_cond)
        target_samples = target_samples.to(device)
        samples = samples.to(device)

        loss = mmd_loss(target_samples, samples)

    else:
        raise ValueError(f"Unknown loss method: {loss_method}")

    return loss

def optimize_DFLOW(vf_y_cond_x, vf_X, device, mog_means, mog_variances, weights,
                   max_iter=100, FLAG=False, n_sample=50, loss_method="MMD", line_search_fn=None, REGULARIZE=False,
                    vf_x_cond_y=None):
    # Initialize x on the correct device with requires_grad
    x = torch.randn((1, vf_X.nfeatures), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([x], max_iter=max_iter, line_search_fn=line_search_fn)

    # Generate MoG samples on device
    samples = generate_mog_samples_not_differentiable(n_sample, mog_means, mog_variances, weights).to(device)

    best_x, best_loss = None, float('inf')

    def closure():
        optimizer.zero_grad()
        x_1 = vf_X.sample(num_sample=1, y=None, x_init=x)

        loss = compute_loss(vf_y_cond_x, vf_X, x_1, samples, loss_method,
                            mog_means, mog_variances, weights, device)

        nonlocal best_x, best_loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = x.detach().clone()

        if FLAG:
            print(f"x_0: {x.detach().cpu().numpy()} | x_1: {x_1} | loss: {loss.item():.6f}")

        if REGULARIZE:
            norm_x = x.norm(p=2)  # ‖x‖
            d = x.shape[1]

            R = (d - 1) * torch.log(norm_x + 1e-6) - 0.5 * norm_x ** 2
            loss = loss + -1. * R

        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)
    best_x1 = vf_X.sample(num_sample=1, y=None, x_init=best_x.to(device))
    # compute the final loss - proxy for the real loss
    with torch.no_grad():
        final_loss = compute_loss(vf_y_cond_x, vf_X, best_x1, samples, loss_method,
                                  mog_means, mog_variances, weights, device)

    del optimizer, samples, x, best_x
    torch.cuda.empty_cache() if device == "cuda" else None

    return best_x1, final_loss


#
#
# def optimize_DFLOW(vf_y_cond_x, vf_X, device, mog_means, mog_variances, weights,
#                    max_iter=100, FLAG=False, n_sample=50, loss_method="MMD",
#                    line_search_fn=None, REGULARIZE=False, crazy_condition=False, vf_x_cond_y=None,
#                    max_restarts=3, restart_loss_threshold=0.5):
#     for attempt in range(max_restarts):
#         # Initialize x with requires_grad=True
#         x = torch.randn((1, vf_X.nfeatures), dtype=torch.float32, device=device, requires_grad=True)
#         optimizer = torch.optim.LBFGS([x], max_iter=max_iter, line_search_fn=line_search_fn)
#
#         samples = generate_mog_samples(n_sample, mog_means, mog_variances, weights, tau=10.0).to(device)
#         best_x, best_loss = None, float('inf')
#
#         def closure():
#             optimizer.zero_grad()
#             if crazy_condition:
#                 condition = dist_utils.generate_mog_samples(1, mog_means, mog_variances, weights, tau=10.0).to(torch.float32).to(device)
#                 x_cond = condition.reshape(-1).to(device) * torch.ones(1, vf_x_cond_y.condition_on, dtype=torch.float32, device=device)
#                 x_1 = vf_x_cond_y.sample(1, x_cond)
#             else:
#                 x_1 = vf_X.sample(num_sample=1, y=None, x_init=x)
#
#             loss = compute_loss(vf_y_cond_x, vf_X, x_1, samples, loss_method,
#                                 mog_means, mog_variances, weights, device)
#
#             nonlocal best_x, best_loss
#             if loss.item() < best_loss:
#                 best_loss = loss.item()
#                 best_x = x.detach().clone()
#
#             if FLAG:
#                 print(f"Attempt {attempt+1} | x_0: {x.detach().cpu().numpy()} | x_1: {x_1} | loss: {loss.item():.6f}")
#             if REGULARIZE:
#                 norm_x = x.norm(p=2)
#                 d = x.shape[1]
#                 R = (d - 1) * torch.log(norm_x + 1e-6) - 0.5 * norm_x ** 2
#                 loss = loss - R  # Encourage Gaussian norm
#
#             loss.backward(retain_graph=True)
#             return loss
#
#         optimizer.step(closure)
#
#         # If success, break early
#         if best_loss <= restart_loss_threshold:
#             break
#         else:
#             if FLAG:
#                 print(f"Restarting (loss={best_loss:.4f} > {restart_loss_threshold})")
#
#         # Cleanup before retry
#         del optimizer, samples, x, best_x
#         torch.cuda.empty_cache() if device == "cuda" else None
#
#     # Final best sample
#     best_x1 = vf_X.sample(num_sample=1, y=None, x_init=best_x.to(device))
#     return best_x1, best_loss


def optimize_LGD_cfg(model_uncond, model_cond_cm, noise_scheduler,
                     mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
                     nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda",
                     FLAG=False, cfg_scale=2., lr=0.01, FLAG_X_0=False,
                     use_uniform=False):
    """Promoted from mnistExpThesisFinal.ipynb cell 26 (MNIST rotation task).

    Delegates to guidance.core. Reproduces the original exactly, including:

      * DDIM computed inline against a diffusers DDIMScheduler with 100
        inference steps, rather than via model.sample_ddim_step
      * perturbation std = r_t**2 with r_t = sqrt(1 - abar_t)  (note this is a
        third distinct formula; see guidance/config.py)
      * step size  r_t/(1 + r_t**2) + 5t/1000, forced to ZERO for t < 200
        (a guidance cutoff that appears in no algorithm box in the paper)
      * the conditional sampler run with the 6-step consistency schedule
        ts = [150, 50, 20, 10, 5, 1] -- NOT single-step
      * one unguided DDIM step after the loop, returning its pred_x0
      * final loss evaluated with 4x nsamples

    Two arguments are inert, exactly as in the original:
      cfg_scale -- never referenced in the body; there is no classifier-free
                   guidance here despite the function name
      FLAG_X_0  -- drove inline debug plotting; dropped in the promotion
    """
    from diffusers import DDIMScheduler

    from losses.circular import (
        circular_to_angles, angles_to_circular, create_mog_pdf_evaluator)
    from losses.sliced import SupportAwareSlicedWassersteinLoss
    from utils.dist import generate_mog_samples_not_differentiable

    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    ddim_scheduler.set_timesteps(num_inference_steps=100)
    timesteps = ddim_scheduler.timesteps

    if use_uniform:
        pdf_evaluator = lambda x: torch.ones_like(x) / 360.0
    else:
        pdf_evaluator = create_mog_pdf_evaluator(mog_means, mog_variances, weights)

    swd = SupportAwareSlicedWassersteinLoss(
        device=device, support_weight=20.0, pdf_threshold=1e-6, n_projections=50)

    def loss_function(pred_circular, target_circular):
        # The original calls the loss with four arguments; the angles are
        # recoverable from the circular representation (the conversion is an
        # involution), so this adapter is faithful.
        pred_angles = circular_to_angles(pred_circular)
        return swd(pred_circular, target_circular, pred_angles, pdf_evaluator)[0]

    def target_generator(n, dev):
        if use_uniform:
            return angles_to_circular(torch.rand(n, device=dev) * 360)
        ang = generate_mog_samples_not_differentiable(
            n, mog_means, mog_variances, weights).squeeze()
        return angles_to_circular(ang)

    def cond_transform(samples):
        return angles_to_circular(circular_to_angles(samples))

    def ddim_step_fn(model, x_t, t, i):
        residual = model(x_t, torch.tensor([t], device=device))
        alpha_prod_t = ddim_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (ddim_scheduler.alphas_cumprod[timesteps[i + 1]]
                             if i < len(timesteps) - 2 else torch.tensor(1.0))
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x_t - beta_prod_t ** 0.5 * residual) / alpha_prod_t ** 0.5
        x_prev = alpha_prod_t_prev ** 0.5 * pred_x0 + (1 - alpha_prod_t_prev) ** 0.5 * residual
        return x_prev, pred_x0

    def perturb_fn(model, t, i):
        return torch.sqrt(1 - ddim_scheduler.alphas_cumprod[t])

    def step_size_fn(t, i):
        r_t = torch.sqrt(1 - ddim_scheduler.alphas_cumprod[t])
        return r_t / (1 + r_t ** 2) + 5 * t / 1000

    cfg = GuidanceConfig(
        n_cond=nsamples,
        n_mc=num_x_t,
        final_n_cond=nsamples * 4,
        init_fn=lambda _nf, dev: torch.randn(1, 1, 28, 28, device=dev, requires_grad=True),
        unsqueeze_init=False,
        perturb_fn=perturb_fn,
        perturb_square=True,
        aggregate_fn=aggregate_logsumexp,
        step_size_fn=step_size_fn,
        guidance_cutoff_t=200,
        timestep_source="list",
        timesteps_list=timesteps,
        ddim_step_fn=ddim_step_fn,
        final_unguided_step=True,
        cond_sample_kwargs={"ts": [150.0, 50.0, 20.0, 10.0, 5.0, 1.0]},
        repeat_condition=False,
        condition_on_slice=False,
        cond_output_transform=cond_transform,
        target_generator=target_generator,
        compute_final_loss=True,
        device=device,
        progress=FLAG,
        tag="legacy:optimize_LGD_cfg",
    )
    out = optimize_guided(model_uncond, model_cond_cm, loss_function, cfg)
    return out["x_final"], out["x_final"], out["final_loss"]
