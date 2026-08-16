"""Unified guided-diffusion optimizer.

One implementation of the loop that was previously copy-pasted across
`optimize_DPS`, `optimize_LGD` (Optimization.py), `optimize_LGD_CM`
(PocThesis.ipynb cell 7) and `optimize_LGD_cfg` (mnistExpThesisFinal.ipynb
cell 26).

DESIGN RULE: this is a behaviour-preserving extraction. Every branch exists to
reproduce a legacy variant exactly, including the order in which random
numbers are consumed -- equivalence tests compare against Optimization_legacy.
Do not "fix" anything here until tests/test_equivalence.py passes; fixes go in
afterwards, as separate commits, so that any change in results is attributable.

Deliberately NOT unified here:
  * optimize_LGD_latent -- computes x_t_minus_1 and never uses it; it runs
    Adam on x_t rather than guided reverse diffusion. Different algorithm.
  * optimize_DFLOW      -- LBFGS over a flow-matching model. Different algorithm.
Both remain in Optimization.py unchanged.
"""

import math
from typing import Optional

import torch

from .config import GuidanceConfig


def _iter_timesteps(model_uncond, cfg):
    """Yield (i, t) pairs in the order the legacy variant used."""
    if cfg.timestep_source == "range":
        steps = range(model_uncond.diffusion_steps - 1, 0, -1)
        for i, t in enumerate(steps):
            yield i, t
    elif cfg.timestep_source == "model":
        for i, t in enumerate(model_uncond.timesteps):
            yield i, t
    elif cfg.timestep_source == "list":
        # MNIST iterates an explicit schedule and excludes the last step,
        # which it handles unguided after the loop.
        assert cfg.timesteps_list is not None, "timestep_source='list' needs timesteps_list"
        for i, t in enumerate(cfg.timesteps_list[:-1]):
            yield i, t
    else:
        raise ValueError(f"unknown timestep_source: {cfg.timestep_source}")


def _ddim_step(model_uncond, x_t, t, i, cfg):
    """Legacy variants expose three different ways of taking a DDIM step."""
    if cfg.ddim_step_fn is not None:
        # MNIST computes the step inline against a diffusers scheduler.
        return cfg.ddim_step_fn(model_uncond, x_t, t, i)
    if cfg.ddim_step_style == "condition_x":
        return model_uncond.sample_ddim_step(
            x_t, t, condition_x=None, device=cfg.device, eta=0.0
        )
    elif cfg.ddim_step_style == "indexed":
        return model_uncond.sample_ddim_step(
            x_t, t, i, device=cfg.device, eta=0.0
        )
    else:
        raise ValueError(f"unknown ddim_step_style: {cfg.ddim_step_style}")


def _draw_conditional(model_cond, x0_sample, cfg):
    """Draw n_cond samples from f_phi(x0_sample, .), matching legacy plumbing."""
    if cfg.repeat_condition:
        condition = x0_sample.view(1, -1).repeat(cfg.n_cond, 1).to(cfg.device)
    else:
        condition = x0_sample

    out = model_cond.sample(
        nsamples=cfg.n_cond, condition_x=condition, device=cfg.device,
        **cfg.cond_sample_kwargs
    )
    samples = out[0] if isinstance(out, tuple) else out

    if cfg.condition_on_slice:
        samples = samples[:, model_cond.condition_on:]
    if cfg.cond_output_transform is not None:
        samples = cfg.cond_output_transform(samples)
    return samples


def optimize_guided(model_uncond, model_cond, loss_function, cfg: GuidanceConfig):
    """Guided reverse diffusion with a black-box loss on the induced conditional.

    Args:
        model_uncond:  pretrained score/diffusion model for P(X)
        model_cond:    differentiable conditional sampler f_phi for P(Y|X)
        loss_function: callable(pred_samples, target_samples) -> scalar tensor
        cfg:           GuidanceConfig

    Returns:
        dict with keys: x_final, final_loss, best_loss, best_x0, n_steps
        (wrappers in Optimization.py unpack this into the legacy tuple shapes)
    """
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        if cfg.device == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)

    assert cfg.target_generator is not None, "cfg.target_generator must be set"

    device = cfg.device
    # MNIST's UNet has no .nfeatures; its init_fn ignores the argument.
    x_t = cfg.init_fn(getattr(model_uncond, "nfeatures", None), device)
    if cfg.unsqueeze_init:
        x_t = x_t.unsqueeze(0)

    best_loss = float("inf")
    best_x0 = None
    n_steps = 0

    iterator = _iter_timesteps(model_uncond, cfg)
    if cfg.progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(list(iterator))
        except ImportError:
            pass

    for i, t in iterator:
        x_t = x_t.detach().clone().requires_grad_(True)

        x_t_minus_1, pred_x0 = _ddim_step(model_uncond, x_t, t, i, cfg)

        r_t = cfg.perturb_fn(model_uncond, t, i)
        neg_losses = []

        n_inner = 1 if r_t is None else cfg.n_mc
        for _ in range(n_inner):
            if r_t is None:
                x0_sample = pred_x0
            else:
                scale = r_t ** 2 if cfg.perturb_square else r_t
                x0_sample = pred_x0 + scale * torch.randn_like(pred_x0)

            pred_samples = _draw_conditional(model_cond, x0_sample, cfg)
            target_samples = cfg.target_generator(cfg.n_cond, device)

            loss_val = loss_function(pred_samples, target_samples)
            neg_losses.append(-loss_val)

            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                best_x0 = x0_sample.detach().clone()

        agg_loss = cfg.aggregate_fn(neg_losses)
        grad = torch.autograd.grad(agg_loss, x_t, retain_graph=True)[0]

        step = cfg.step_size_fn(t, i)
        if cfg.guidance_cutoff_t is not None and t < cfg.guidance_cutoff_t:
            step = 0.0

        with torch.no_grad():
            x_t = cfg.update_fn(x_t, x_t_minus_1, grad, step)
        n_steps += 1

    # ---- optional final unguided step (MNIST takes one, using its
    #      pred_x0 as the returned sample) ----
    if cfg.final_unguided_step:
        with torch.no_grad():
            last_t = cfg.timesteps_list[-1]
            _, pred_x0 = _ddim_step(model_uncond, x_t, last_t,
                                    len(cfg.timesteps_list) - 1, cfg)
            x_t = pred_x0

    # ---- final loss, evaluated fresh at the returned x ----
    final_loss = None
    if cfg.compute_final_loss:
        final_n = cfg.final_n_cond or cfg.n_cond
        with torch.no_grad():
            eval_cfg = _EvalView(cfg, final_n)
            pred_samples = _draw_conditional(model_cond, x_t, eval_cfg)
            target_samples = cfg.target_generator(final_n, device)
            final_loss = loss_function(pred_samples, target_samples)

    x_final = x_t.detach().clone()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "x_final": x_final,
        "final_loss": final_loss.detach() if final_loss is not None else None,
        "best_loss": best_loss,
        "best_x0": best_x0,
        "n_steps": n_steps,
    }


class _EvalView:
    """Thin view over a GuidanceConfig with n_cond overridden, so the final
    evaluation can use a different sample count without mutating cfg."""

    def __init__(self, cfg: GuidanceConfig, n_cond: int):
        self._cfg = cfg
        self.n_cond = n_cond

    def __getattr__(self, name):
        return getattr(self._cfg, name)
