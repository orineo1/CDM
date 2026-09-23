"""
Correcting the bias of loss-guided sampling with twisted sequential Monte Carlo, on the toy setting
of Figure 1 (exact loss, 1-D x, DDPM steps with eta = 1).

Guided reverse diffusion is a proposal: at reverse step t it draws
    x_{t-1} ~ q(. | x_t) = N(mu_p(x_t) - s_t g(x_t), sigma_t^2)
where mu_p is the unguided DDPM mean, g = grad_{x_t} L(x0_hat(x_t)) the loss gradient at the Tweedie
estimate and s_t the guidance step. Its terminal law is not the tilted target
Q_beta(x0) ~ P(x0) exp(-beta L(x0)) (Figure 1(d)). Twisted SMC (Wu et al., 2023) repairs this with
importance weights: with twisting potentials G_t(x_t) = exp(-beta L(x0_hat(x_t))) and the unguided
transition p(. | x_t) = N(mu_p, sigma_t^2), every particle carries
    log w  +=  log p(x_{t-1}|x_t) - log q(x_{t-1}|x_t) + log G_{t-1}(x_{t-1}) - log G_t(x_t),
and the population is resampled when its effective sample size drops below a fraction of N. The
weighted terminal population targets P(x0) G_0(x0) = Q_beta(x0). Everything here is exact except
the trained score network, so the figure isolates the sampler's bias and its correction.

    python scripts/fig_smc_correction.py --setting hard --n-particles 2000 --device cuda

Writes results/fig_smc_<setting>.png (per beta: Q_beta, prior, plain guided histogram, corrected
histogram) and results/fig_smc_<setting>.json (TV between each histogram and Q_beta, ESS trace,
number of resampling events).
"""
import argparse
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import fig1_toy_example as f1                                       # noqa: E402
from guidance.core import _systematic_resample                      # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", default="hard", choices=["hard", "bimodal", "paper", "mog2d"])
    p.add_argument("--ckpt-dir", default="checkpoints/toy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 2.0, 8.0, 16.0])
    p.add_argument("--n-particles", type=int, default=2000)
    p.add_argument("--ess-frac", type=float, default=0.5)
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    return p.parse_args()


def guided_smc(st, model, beta, N, args, corrected):
    """One population of N trajectories under the guided DDPM proposal (step s_t = beta * beta_t, eta = 1).
    corrected=False: plain loss-guided sampling (Figure 1(d)); corrected=True: twisted-SMC weights and
    resampling. Returns final x0 samples, the ESS trace and the number of resampling events."""
    dev = args.device
    G = tuple(g.detach().to(dev) for g in f1.cond_params(st, st.x_star[0].to(dev)))
    ab = model.baralphas.to(dev)
    T = int(model.diffusion_steps)
    x = torch.randn(N, 1, device=dev)
    logw = torch.zeros(N, device=dev)
    ess_trace, n_res = [], 0
    prev_logG = None

    def loss_and_grad(x_t, t):
        x_t = x_t.detach().requires_grad_(True)
        eps = model(x_t, torch.full([N, 1], t, device=dev), None)
        x0 = (x_t - torch.sqrt(1 - ab[t]) * eps) / torch.sqrt(ab[t])
        L = f1.l2_1d(f1.cond_params(st, x0.view(-1)), G)                     # (N,)
        g = torch.autograd.grad(L.sum(), x_t)[0]
        return eps.detach(), x0.detach(), L.detach(), g.detach()

    for t in range(T - 1, 0, -1):
        eps, x0, L, g = loss_and_grad(x, t)
        ab_prev = ab[t - 1]
        sigma = torch.sqrt((1 - ab_prev) / (1 - ab[t])) * torch.sqrt(1 - ab[t] / ab_prev)   # eta = 1
        mu_p = torch.sqrt(ab_prev) * x0 + torch.sqrt(torch.clamp(1 - ab_prev - sigma ** 2, min=0)) * eps
        s_t = beta * float(model.betas[t])
        mu_q = mu_p - s_t * g
        noise = torch.randn_like(x)
        x_new = mu_q + sigma * noise
        if corrected and beta > 0:
            logG_t = -beta * L
            # log p(x_new | x) - log q(x_new | x): same variance, means mu_p and mu_q
            log_pq = ((x_new - mu_q) ** 2 - (x_new - mu_p) ** 2).sum(1) / (2 * sigma ** 2)
            # G_{t-1} at the new state needs the loss at its Tweedie estimate: computed at the next step;
            # accumulate the transition ratio now and the potential ratio when the next loss is known
            logw = logw + log_pq
            # potentials telescope: initial weight G_{T-1}(x_{T-1}), then G_t(x_t) / G_{t+1}(x_{t+1}) once the
            # loss at the new state is known (one iteration later), closed by G_0(x_0) after the loop
            logw = logw + (logG_t if prev_logG is None else logG_t - prev_logG)
            prev_logG = logG_t
            w = torch.softmax(logw, 0)
            ess = float(1.0 / (w * w).sum())
            ess_trace.append(ess)
            if ess < args.ess_frac * N:
                idx = _systematic_resample(w.cpu()).to(dev)
                x_new = x_new[idx]
                prev_logG = prev_logG[idx]
                logw = torch.zeros(N, device=dev)
                n_res += 1
        x = x_new
    # terminal potential: the loss at the final x0 itself closes the telescoping product
    if corrected and beta > 0:
        L0 = f1.l2_1d(f1.cond_params(st, x.view(-1)), G).detach()
        logw = logw + (-beta * L0 - prev_logG)
        w = torch.softmax(logw, 0)
        idx = _systematic_resample(w.cpu()).to(dev)
        x = x[idx]
        n_res += 1
    return x.view(-1).cpu().numpy(), ess_trace, n_res


def tv_hist_vs_curve(samples, x_grid, q, lo, hi, bins):
    """TV between the histogram of samples and the analytic density on the same bins."""
    h, edges = np.histogram(samples, bins=bins, range=(lo, hi), density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    qb = np.interp(centres, x_grid, q)
    width = edges[1] - edges[0]
    return float(0.5 * np.abs(h - qb).sum() * width)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    st = f1.make_setting(args.setting)
    a = types.SimpleNamespace(device=args.device, ckpt_dir=args.ckpt_dir, seed=args.seed, nunits=128, nblocks=3,
                              epochs=0, batch_size=1024, uncond="load", cond="load", cm="load", retrain=False)
    model, _, _, _ = f1.get_models(st, a)
    model.eval()
    x_grid, px, Q = f1.analytic_curves(st, args.betas)
    lo, hi = st.x_range
    out = {"setting": args.setting, "n_particles": args.n_particles, "betas": args.betas, "per_beta": {}}
    runs = {}
    for beta in args.betas:
        t0 = time.time()
        plain, _, _ = guided_smc(st, model, beta, args.n_particles, args, corrected=False)
        corr, ess, n_res = guided_smc(st, model, beta, args.n_particles, args, corrected=True)
        tv_plain = tv_hist_vs_curve(plain, x_grid, Q[beta], lo, hi, args.bins)
        tv_corr = tv_hist_vs_curve(corr, x_grid, Q[beta], lo, hi, args.bins)
        runs[beta] = (plain, corr)
        out["per_beta"][str(beta)] = {"tv_plain_vs_Q": tv_plain, "tv_corrected_vs_Q": tv_corr, "n_resample": n_res,
                                      "ess_min": (min(ess) if ess else None), "ess_final": (ess[-1] if ess else None),
                                      "time_s": time.time() - t0}
        print(f"[smc] beta={beta:g}: TV(hist, Q_beta) plain {tv_plain:.3f} -> corrected {tv_corr:.3f} | "
              f"{n_res} resamplings, min ESS {min(ess) if ess else float('nan'):.0f} | {time.time()-t0:.0f}s")
    stem = args.out or os.path.join("results", f"fig_smc_{args.setting}")
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    with open(stem + ".json", "w") as f:
        json.dump(out, f, indent=2)
    np.savez(stem + ".npz", **{f"plain_{b}": runs[b][0] for b in args.betas}, **{f"corr_{b}": runs[b][1] for b in args.betas})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_star = float(st.x_star[0])
    fig, axes = plt.subplots(1, len(args.betas), figsize=(4.2 * len(args.betas), 3.6), sharey=True)
    ymax = 1.1 * max(max(Q[b].max(), px.max()) for b in args.betas)
    for i, beta in enumerate(args.betas):
        ax = axes[i]
        plain, corr = runs[beta]
        ax.hist(plain, bins=args.bins, range=(lo, hi), density=True, color="steelblue", alpha=0.45, label="guided samples (LGD)")
        ax.hist(corr, bins=args.bins, range=(lo, hi), density=True, histtype="step", color="tab:green", lw=1.8,
                label="twisted-SMC corrected")
        ax.plot(x_grid, Q[beta], color="tab:purple", lw=2, label=r"$\mathcal{Q}_\beta(x)$", zorder=3)
        ax.plot(x_grid, px, color="tab:orange", ls="-.", lw=1.8, label=r"prior $\mathcal{P}(x)$", zorder=4)
        ax.axvline(x_star, color="black", ls="--", lw=1.5, zorder=5)
        r = out["per_beta"][str(beta)]
        ax.set_title(rf"$\beta = {beta:g}$:  TV to $\mathcal{{Q}}_\beta$  {r['tv_plain_vs_Q']:.2f} $\to$ {r['tv_corrected_vs_Q']:.2f}", fontsize=11)
        ax.set_xlim(lo, hi); ax.set_ylim(0, ymax); ax.set_xlabel("x"); ax.grid(alpha=0.15)
        if i == 0:
            ax.set_ylabel("density"); ax.annotate(r"$x^*$", (x_star, ymax), xytext=(-4, -4), textcoords="offset points", ha="right", va="top")
        if i == 1:
            h, l = ax.get_legend_handles_labels()
            order = [l.index(k) for k in (r"prior $\mathcal{P}(x)$", r"$\mathcal{Q}_\beta(x)$", "guided samples (LGD)", "twisted-SMC corrected")]
            ax.legend([h[j] for j in order], [l[j] for j in order], fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(stem + ".png", dpi=200)
    print(f"[figure] wrote {stem}.png and {stem}.json")


if __name__ == "__main__":
    main()
