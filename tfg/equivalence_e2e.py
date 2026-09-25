"""EXP-012 prerequisite: end-to-end harness equivalence checks.

The new wrapper must be shown to reproduce the legacy CDM implementation before
any confirmatory numbers are quoted from it.

Chain of evidence
-----------------
  A1  ``legacy_replica`` == ``Optimization.optimize_LGD``   (final x, final loss)
  A2  new wrapper in legacy-compat mode == ``legacy_replica`` (full trajectory)
  => new wrapper == legacy implementation.

A1 compares only final outputs because ``optimize_LGD`` exposes no trajectory and
the legacy file must stay untouched. A2 compares every step, so the chain covers
the whole trajectory.

Legacy conventions reproduced exactly in ``legacy_replica`` (Optimization.py:88-146):
  * ``x_t = zeros``; loop ``t = T-1 .. 1``; DDIM ``eta = 0``
  * ``r_t = beta_t / sqrt(1 + beta_t^2)``; ``x0_sample = pred_x0 + r_t * randn_like``
  * ``num_x_t`` Monte-Carlo draws per step
  * target samples **redrawn inside the j loop** at the same count as ``nsamples``
  * ``MMDLoss(RBF())`` with the ADAPTIVE bandwidth
  * ``log_mean_exp = -logsumexp(stack(-losses)) + log(num_x_t)``; backward; ``x = x_prev - grad``
  * ``model_cond.sample`` called with its DEFAULT ``ts`` (the 15-entry ladder), not
    the paper's 6-entry one.

RNG order matters: both paths must consume the global generator identically.
"""

import math
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tfg._compat import ensure_legacy_importable      # noqa: E402

ensure_legacy_importable()

import dist_utils                            # noqa: E402
from LossFunctions import MMDLoss, RBF       # noqa: E402


def legacy_replica(model_uncond, model_cond, params, nsamples=250, num_x_t=3,
                   bandwidth=None, device="cpu", trace=None, ts=None):
    """Faithful re-implementation of ``optimize_LGD`` (CM=True path)."""
    mmd = MMDLoss(kernel=RBF(bandwidth=bandwidth, device="cpu"), device="cpu")
    tgt_means = params["target_means"].float()
    tgt_vars = torch.stack([v.float() for v in params["target_variances"]])
    tgt_w = params["target_weights"].float()

    x_t = torch.zeros(model_uncond.nfeatures, device=device).unsqueeze(0)
    for t in range(model_uncond.diffusion_steps - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)
        x_prev, pred_x0 = model_uncond.sample_ddim_step(
            x_t, t, condition_x=None, device=device, eta=0.0)
        cur_var = model_uncond.betas[t].to(device)
        r_t = cur_var / torch.sqrt(1 + cur_var ** 2)

        losses = []
        for _ in range(num_x_t):
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            cond = x0_sample.view(1, -1).repeat(nsamples, 1)
            kw = {} if ts is None else {"ts": ts}
            y, _, _ = model_cond.sample(nsamples=nsamples, condition_x=cond,
                                        device=device, **kw)
            mog = dist_utils.generate_mog_samples_not_differentiable(
                nsamples, tgt_means, tgt_vars, tgt_w)
            losses.append(-mmd(y, mog))

        lme = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
        lme.backward()
        grad = x_t.grad.clone()
        with torch.no_grad():
            x_t = x_prev.detach().clone() - grad
        if trace is not None:
            trace.append({"t": t, "x": float(x_t.reshape(-1)[0]),
                          "grad": float(grad.reshape(-1)[0]),
                          "loss": float(lme.detach())})

    # final proxy loss, as legacy does
    cond = x_t.view(1, -1).repeat(nsamples, 1)
    kw = {} if ts is None else {"ts": ts}
    with torch.no_grad():
        y, _, _ = model_cond.sample(nsamples=nsamples, condition_x=cond,
                                    device=device, **kw)
        mog = dist_utils.generate_mog_samples_not_differentiable(
            nsamples, tgt_means, tgt_vars, tgt_w)
        final = float(mmd(y, mog))
    return float(x_t.detach().reshape(-1)[0]), final


def run_legacy(model_uncond, model_cond, params, nsamples=250, num_x_t=3,
               device="cpu"):
    """Call the untouched repository implementation."""
    import Optimization
    tgt_vars = [v.float() for v in params["target_variances"]]
    out = Optimization.optimize_LGD(
        model_uncond, model_cond,
        params["target_means"].float(), tgt_vars, params["target_weights"].float(),
        [m.float() for m in params["mu_list"]],
        [s.float() for s in params["Sigma_list"]], params["alpha"].float(),
        nsamples=nsamples, num_x_t=num_x_t, loss="MMD", CM=True,
        device=device, FLAG=False)
    x_final, _, final_loss = out
    return float(x_final.reshape(-1)[0]), float(final_loss)
