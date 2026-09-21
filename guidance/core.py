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

import time
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
            x_t, t, condition_x=None, device=cfg.device, eta=cfg.eta
        )
    elif cfg.ddim_step_style == "indexed":
        return model_uncond.sample_ddim_step(
            x_t, t, i, device=cfg.device, eta=cfg.eta
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


def _guided_step(model_uncond, model_cond, loss_function, cfg, x_t, t, i, best):
    """One guided reverse-diffusion step for one state x_t.

    Tweedie estimate, MC perturbations, inner loss (Algorithm 2), gradient with respect to
    x_t, DDIM step minus the guidance push. `best` is a dict {loss, x0} updated in place with
    the lowest inner loss seen so far. Consumes random numbers in the legacy order."""
    device = cfg.device
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

        if loss_val.item() < best["loss"]:
            best["loss"] = loss_val.item()
            best["x0"] = x0_sample.detach().clone()

    agg_loss = cfg.aggregate_fn(neg_losses)
    grad = torch.autograd.grad(agg_loss, x_t, retain_graph=True)[0]

    step = cfg.step_size_fn(t, i)
    if cfg.guidance_cutoff_t is not None and t < cfg.guidance_cutoff_t:
        step = 0.0

    with torch.no_grad():
        x_next = cfg.update_fn(x_t, x_t_minus_1, grad, step)
    return x_next, {"loss": float(agg_loss.detach()), "grad_norm": float(grad.norm()), "step": float(step)}


def _final_loss(model_cond, loss_function, cfg, x):
    """The loss reported for a returned x: fresh conditional draws (final_n_cond) at x itself."""
    final_n = cfg.final_n_cond or cfg.n_cond
    with torch.no_grad():
        eval_cfg = _EvalView(cfg, final_n)
        pred_samples = _draw_conditional(model_cond, x, eval_cfg)
        target_samples = cfg.target_generator(final_n, cfg.device)
        return loss_function(pred_samples, target_samples).detach()


def _prior_score(model_uncond, x, t=None):
    """Approximate grad log P(x) from the noise predictor at a small noise level:
    score_t(x) = -eps_theta(x, t) / sqrt(1 - abar_t), with x taken as x_t (abar_t ~ 1).
    t defaults to a tenth of the schedule: at t = 1 the network's relative error is worst
    (scripts/check_models.py), while at t ~ T/10 it is ~1e-2 and abar_t is still ~1."""
    if t is None:
        t = max(1, int(round(0.1 * int(model_uncond.diffusion_steps))))
    t_batch = torch.full([x.shape[0], 1], t, device=x.device)
    eps = model_uncond(x, t_batch, None)
    return -eps / torch.sqrt(1 - model_uncond.baralphas[t])


def refine(model_uncond, model_cond, loss_function, cfg, x0):
    """Post-refinement: plain gradient descent on the plug-in loss L_hat(x) (Algorithm 2 called
    at the clean x, gradient with respect to x, no diffusion), optionally regularised by the
    prior: x <- x - lr * (grad L_hat(x) - lambda * score_theta(x)). Returns (x, trace)."""
    x = x0.detach().clone()
    trace = []
    t_start = time.time()
    for k in range(cfg.refine_steps):
        x = x.detach().clone().requires_grad_(True)
        pred_samples = _draw_conditional(model_cond, x, cfg)
        target_samples = cfg.target_generator(cfg.n_cond, cfg.device)
        loss_val = loss_function(pred_samples, target_samples)
        grad = torch.autograd.grad(loss_val, x)[0]
        prior_norm = 0.0
        if cfg.refine_lambda > 0:
            with torch.no_grad():
                score = _prior_score(model_uncond, x.detach())
            prior_norm = float(score.norm())
            grad = grad - cfg.refine_lambda * score
        with torch.no_grad():
            x = x - cfg.refine_lr * grad
        trace.append({"k": k, "loss": float(loss_val.detach()), "grad_norm": float(grad.norm()),
                      "prior_score_norm": prior_norm, "elapsed_s": time.time() - t_start})
    return x.detach(), trace


def _seed(cfg):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        if cfg.device == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)


def _init_state(model_uncond, cfg):
    # MNIST's UNet has no .nfeatures; its init_fn ignores the argument.
    x_t = cfg.init_fn(getattr(model_uncond, "nfeatures", None), cfg.device)
    if cfg.unsqueeze_init:
        x_t = x_t.unsqueeze(0)
    return x_t


def _make_iterator(model_uncond, cfg):
    iterator = _iter_timesteps(model_uncond, cfg)
    if cfg.progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(list(iterator))
        except ImportError:
            pass
    return iterator


def optimize_guided(model_uncond, model_cond, loss_function, cfg: GuidanceConfig):
    """Guided reverse diffusion with a black-box loss on the induced conditional.

    Args:
        model_uncond:  pretrained score/diffusion model for P(X)
        model_cond:    differentiable conditional sampler f_phi for P(Y|X)
        loss_function: callable(pred_samples, target_samples) -> scalar tensor
        cfg:           GuidanceConfig

    Returns:
        dict with keys: x_final, final_loss, best_loss, best_x0, n_steps, time_s,
        trace (per-step list of {i, t, loss, grad_norm, step, elapsed_s});
        with cfg.refine_steps > 0 also refine = {trace, x_before, final_loss_before}
        (wrappers in Optimization.py unpack this into the legacy tuple shapes)
    """
    _seed(cfg)
    assert cfg.target_generator is not None, "cfg.target_generator must be set"
    device = cfg.device
    x_t = _init_state(model_uncond, cfg)

    best = {"loss": float("inf"), "x0": None}
    n_steps = 0
    trace = []                      # per-step diagnostics, returned with the result
    t_start = time.time()

    for i, t in _make_iterator(model_uncond, cfg):
        x_t, info = _guided_step(model_uncond, model_cond, loss_function, cfg, x_t, t, i, best)
        n_steps += 1
        trace.append({"i": i, "t": int(t) if not torch.is_tensor(t) else float(t),
                      **info, "elapsed_s": time.time() - t_start})

    # ---- optional final unguided step (MNIST takes one, using its
    #      pred_x0 as the returned sample) ----
    if cfg.final_unguided_step:
        with torch.no_grad():
            last_t = cfg.timesteps_list[-1]
            _, pred_x0 = _ddim_step(model_uncond, x_t, last_t,
                                    len(cfg.timesteps_list) - 1, cfg)
            x_t = pred_x0

    # ---- optional post-refinement from the returned x ----
    refine_info = None
    if cfg.refine_steps > 0:
        loss_before = _final_loss(model_cond, loss_function, cfg, x_t) if cfg.compute_final_loss else None
        x_before = x_t.detach().clone()
        x_t, rtrace = refine(model_uncond, model_cond, loss_function, cfg, x_t)
        refine_info = {"trace": rtrace, "x_before": x_before,
                       "final_loss_before": None if loss_before is None else float(loss_before)}

    # ---- final loss, evaluated fresh at the returned x ----
    final_loss = _final_loss(model_cond, loss_function, cfg, x_t) if cfg.compute_final_loss else None

    x_final = x_t.detach().clone()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "x_final": x_final,
        "final_loss": final_loss,
        "best_loss": best["loss"],
        "best_x0": best["x0"],
        "n_steps": n_steps,
        "time_s": time.time() - t_start,
        "trace": trace,             # list of {i, t, loss, grad_norm, step, elapsed_s}
        "refine": refine_info,
    }


def _systematic_resample(w):
    """Systematic resampling: ancestor indices for weights w (sums to 1)."""
    R = w.numel()
    u = (torch.rand(1, device=w.device) + torch.arange(R, device=w.device)) / R
    cum = torch.cumsum(w, 0)
    cum[-1] = 1.0
    return torch.searchsorted(cum, u).clamp_(max=R - 1)


def optimize_guided_population(model_uncond, model_cond, loss_function, cfg: GuidanceConfig):
    """Particle version of optimize_guided: R = cfg.n_particles trajectories advance in lockstep
    (each step is exactly one _guided_step per particle, so the cost equals R restarts); after
    every step the particles are weighted by exp(-resample_beta * loss) and, when the effective
    sample size falls below ess_frac * R, resampled (systematic), so trajectories that committed
    to a poor basin are replaced by copies of better ones while the noise is still high enough
    for the copies to diversify. resample_beta=None gives R independent restarts run together.

    Returns dict with: x_final (best particle by final loss), final_loss, best_particle,
    x_particles (R, d), final_losses [R], n_steps, time_s, trace (per step: i, t, loss [R],
    grad_norm [R], ess, resampled, ancestors), n_resamples, resample_steps, and, with
    cfg.refine_steps > 0, refine = {traces [R], x_before (R, d), final_losses_before [R]}.
    """
    _seed(cfg)
    assert cfg.target_generator is not None, "cfg.target_generator must be set"
    R = int(cfg.n_particles)
    device = cfg.device
    xs = [_init_state(model_uncond, cfg) for _ in range(R)]
    bests = [{"loss": float("inf"), "x0": None} for _ in range(R)]
    logw = torch.zeros(R)
    prev_loss = None
    n_steps, n_resamples = 0, 0
    trace, resample_steps = [], []
    t_start = time.time()

    for i, t in _make_iterator(model_uncond, cfg):
        losses, gnorms, step = [], [], 0.0
        for r in range(R):
            xs[r], info = _guided_step(model_uncond, model_cond, loss_function, cfg, xs[r], t, i, bests[r])
            losses.append(info["loss"]); gnorms.append(info["grad_norm"]); step = info["step"]
        L = torch.tensor(losses)
        ess, resampled, ancestors = float(R), False, None
        if cfg.resample_beta is not None:
            if cfg.weight_mode == "fresh":
                logw = -cfg.resample_beta * L
            else:   # accumulated: telescoped potential ratio exp(-beta (L_t - L_{t+1})), reset on resampling
                logw = logw - cfg.resample_beta * (L - (prev_loss if prev_loss is not None else 0.0))
            w = torch.softmax(logw, 0)
            ess = float(1.0 / (w * w).sum())
            if (i + 1) % cfg.resample_every == 0 and ess < cfg.ess_frac * R:
                idx = _systematic_resample(w)
                xs = [xs[j].detach().clone() for j in idx.tolist()]
                bests = [dict(bests[j]) for j in idx.tolist()]
                L = L[idx]
                logw = torch.zeros(R)
                resampled, ancestors = True, idx.tolist()
                n_resamples += 1; resample_steps.append(i)
        prev_loss = L
        n_steps += 1
        trace.append({"i": i, "t": int(t) if not torch.is_tensor(t) else float(t), "loss": losses,
                      "grad_norm": gnorms, "step": step, "ess": ess, "resampled": resampled,
                      "ancestors": ancestors, "elapsed_s": time.time() - t_start})

    if cfg.final_unguided_step:
        with torch.no_grad():
            last_t = cfg.timesteps_list[-1]
            xs = [_ddim_step(model_uncond, x, last_t, len(cfg.timesteps_list) - 1, cfg)[1] for x in xs]

    refine_info = None
    if cfg.refine_steps > 0:
        before = [float(_final_loss(model_cond, loss_function, cfg, x)) for x in xs] if cfg.compute_final_loss else None
        x_before = torch.cat([x.detach() for x in xs], 0)
        traces = []
        for r in range(R):
            xs[r], rtrace = refine(model_uncond, model_cond, loss_function, cfg, xs[r])
            traces.append(rtrace)
        refine_info = {"traces": traces, "x_before": x_before, "final_losses_before": before}

    final_losses = [float(_final_loss(model_cond, loss_function, cfg, x)) for x in xs] if cfg.compute_final_loss else None
    best_r = int(min(range(R), key=lambda r: final_losses[r])) if final_losses is not None else 0
    x_particles = torch.cat([x.detach() for x in xs], 0)
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "x_final": xs[best_r].detach().clone(),
        "final_loss": None if final_losses is None else torch.tensor(final_losses[best_r]),
        "best_particle": best_r,
        "x_particles": x_particles,
        "final_losses": final_losses,
        "best_loss": bests[best_r]["loss"],
        "best_x0": bests[best_r]["x0"],
        "n_steps": n_steps,
        "time_s": time.time() - t_start,
        "trace": trace,
        "n_resamples": n_resamples,
        "resample_steps": resample_steps,
        "refine": refine_info,
    }


class _EvalView:
    """Thin view over a GuidanceConfig with n_cond overridden, so the final
    evaluation can use a different sample count without mutating cfg."""

    def __init__(self, cfg: GuidanceConfig, n_cond: int):
        self._cfg = cfg
        self.n_cond = n_cond

    def __getattr__(self, name):
        return getattr(self._cfg, name)
