"""
Reproduce Figure 1 of the paper (the toy example) from the refactored library.

Setting (paper App. B.8): joint P(X,Y) is a 2-component Gaussian mixture in R^2
with dim X = dim Y = 1, means (-3, 2) and (3, -2), shared covariance
[[0.5, 0.15], [0.15, 0.4]], equal weights; true optimum x* = -3 and target
G(Y) = P(Y | X = x*).

Panels:
  top left    joint scatter, the true x* and the x-hat found by MLGD and MLGD-F
  top right   induced conditionals P(Y | X = x-hat) against the target G(Y)
  bottom      CDMS beta-sweep on the prior alone: histogram of guided samples
              against the analytic tilted density Q_beta(x) ~ P(x) exp(-beta L(x)),
              with L the exact L2 distance between P(Y|X=x) and G; the prior
              P(x) is drawn in every panel so the move from prior to Q_beta is visible

Everything runs through guidance.optimize_guided:
  * MLGD   = legacy optimize_LGD config with the conditional *diffusion* sampler
  * MLGD-F = the same config with the conditional *consistency* sampler (CM=True)
  * beta-sweep = the same outer loop with n_cond = n_mc = 1, an identity "sampler"
    (so the loss is evaluated at the Tweedie point itself), the analytic L2-GMM
    loss, step size beta * beta_t (the DPS convention, so the total guidance over
    the trajectory is about beta * grad L), and a stochastic DDIM step (cfg.eta = 1),
    i.e. the DPS-style sampler of the paper's Figure 1 bottom row.

The three toy models (unconditional DDPM on X, conditional DDPM on Y|X,
conditional iCT on Y|X) are trained here on the fly (no HF checkpoints exist
for the toy) and cached under --ckpt-dir; pass --retrain to redo them.

Differences from the original notebook (conditional-matching-paper/simulations/
notebooks/toy_example_with_beta_sweep.ipynb), all deliberate:
  * one shared unconditional model (T = 100, all steps) for every panel; the
    notebook used a separate T = 999 DDPM with 20 DDIM steps for the sweep
  * the mixture weights of P(Y|X=x) are differentiated in x in the sweep loss
    (the notebook detached them)
  * notation follows the paper: G(Y) for the target, x* for the truth, x-hat for
    the recovered input, Q_beta(x) on the bottom y-axis

Usage:
    python oz/fig1_toy_example.py                       # full: ~10-20 min on CPU
    python oz/fig1_toy_example.py --quick               # smoke: ~1-2 min
    python oz/fig1_toy_example.py --init zeros          # the legacy initialisation
Outputs: --out PNG and a JSON next to it with every number shown.
"""
import argparse
import json
import math
import os
import sys
import time
from functools import partial

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from guidance import GuidanceConfig, optimize_guided                       # noqa: E402
from guidance.config import (init_randn, init_zeros, perturb_beta_ratio,    # noqa: E402
                             perturb_none, aggregate_logsumexp, aggregate_single)
from priors.diffusion import DiffusionModel                                 # noqa: E402
from samplers.consistency import ConsistencyModeliCT                        # noqa: E402
from losses.functions import MMDLoss, RBF                                    # noqa: E402
from utils.dist import (generate_mog_samples_not_differentiable,            # noqa: E402
                        compute_conditionals, compute_alpha,
                        filter_and_normalize, gmm_l2_distance)

# ----------------------------------------------------------------------------
# The toy joint (paper App. B.8)
# ----------------------------------------------------------------------------
MU_LIST = [torch.tensor([-3., 2.]), torch.tensor([3., -2.])]
SIGMA_LIST = [torch.tensor([[0.5, 0.15], [0.15, 0.4]])] * 2
ALPHA = torch.tensor([0.5, 0.5])
X_STAR = torch.tensor([-3.])
CONDITION_ON = 1          # dim(x) = 1, dim(y) = 1
DIFFUSION_STEPS = 100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", default="checkpoints/toy")
    p.add_argument("--out", default="results/fig1_toy_example.png")
    for name, what in [("uncond", "score network for P(X)"), ("cond", "conditional diffusion sampler for Y|X (MLGD)"),
                       ("cm", "conditional consistency sampler for Y|X (MLGD-F)")]:
        p.add_argument(f"--{name}", choices=["auto", "train", "load"], default="auto",
                       help=f"{what}: auto = load from --ckpt-dir if present else train; "
                            f"train = always (re)train and overwrite; load = require the checkpoint")
    p.add_argument("--retrain", action="store_true", help="shorthand for --uncond train --cond train --cm train")
    # training (notebook: 3000 epochs, batch 1024, 3 blocks x 128 units)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--nblocks", type=int, default=3)
    p.add_argument("--nunits", type=int, default=128)
    # optimisation (notebook: n_cond 250, n_mc 1, one attempt each)
    p.add_argument("--n-cond", type=int, default=250)
    p.add_argument("--n-mc", type=int, default=1)
    p.add_argument("--init", choices=["randn", "zeros"], default="randn",
                   help="x_T initialisation; the paper says N(0,I), the legacy code used zeros")
    # beta sweep (notebook: 200 samples per beta, betas 0/0.5/1/1.5)
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    p.add_argument("--n-beta-samples", type=int, default=200)
    p.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    a = p.parse_args()
    if a.quick:
        a.epochs, a.n_beta_samples, a.n_cond = 300, 30, 100
    if a.retrain:
        a.uncond = a.cond = a.cm = "train"
    return a


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Exact 1D quantities, differentiable in x (used by the sweep and the analytic curves)
# ----------------------------------------------------------------------------
def cond_params(x):
    """P(Y | X = x) for the toy joint: means, variances, weights of a 1D mixture,
    all differentiable in the scalar tensor x."""
    means, variances, logw = [], [], []
    for mu, S, a in zip(MU_LIST, SIGMA_LIST, ALPHA):
        mu, S = mu.to(x), S.to(x)
        sxx, sxy, syy = S[0, 0], S[0, 1], S[1, 1]
        means.append(mu[1] + sxy / sxx * (x - mu[0]))
        variances.append(syy - sxy * sxy / sxx)
        logw.append(torch.log(a.to(x)) - 0.5 * torch.log(2 * math.pi * sxx) - (x - mu[0]) ** 2 / (2 * sxx))
    return torch.stack(means), torch.stack(variances), torch.softmax(torch.stack(logw), 0)


def _inner(m1, v1, w1, m2, v2, w2):
    s = v1[:, None] + v2[None, :]
    d = m1[:, None] - m2[None, :]
    return (w1[:, None] * w2[None, :] * torch.exp(-d * d / (2 * s)) / torch.sqrt(2 * math.pi * s)).sum()


def l2_1d(p, q):
    """Exact squared L2 distance between two 1D Gaussian mixtures (means, vars, weights)."""
    return _inner(*p, *p) - 2 * _inner(*p, *q) + _inner(*q, *q)


def prior_density(x):
    """Marginal P(x) of the toy joint, for a tensor of x values."""
    out = torch.zeros_like(x)
    for mu, S, a in zip(MU_LIST, SIGMA_LIST, ALPHA):
        out = out + a * torch.exp(-(x - mu[0]) ** 2 / (2 * S[0, 0])) / math.sqrt(2 * math.pi * S[0, 0])
    return out


def mixture_density_1d(y, params):
    m, v, w = params
    return sum(wk * torch.exp(-(y - mk) ** 2 / (2 * vk)) / math.sqrt(2 * math.pi * vk) for mk, vk, wk in zip(m, v, w))


# ----------------------------------------------------------------------------
# Models: train once, cache
# ----------------------------------------------------------------------------
def get_models(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    dev = args.device

    def path(name):
        return os.path.join(args.ckpt_dir, f"toy_{name}_seed{args.seed}.pt")

    gen_joint = partial(generate_mog_samples_not_differentiable, means=MU_LIST, variances=SIGMA_LIST, weights=ALPHA)
    gen_x = partial(generate_mog_samples_not_differentiable, means=MU_LIST, variances=SIGMA_LIST, weights=ALPHA,
                    kernel_func=lambda X: X[:, :CONDITION_ON])

    uncond = DiffusionModel(nfeatures=CONDITION_ON, nunits=args.nunits, condition=False,
                            diffusion_steps=DIFFUSION_STEPS, nblocks=args.nblocks)
    cond = DiffusionModel(nfeatures=2, nunits=args.nunits, condition=True, condition_on=CONDITION_ON,
                          diffusion_steps=DIFFUSION_STEPS, nblocks=args.nblocks)
    cm = ConsistencyModeliCT(nfeatures=2 - CONDITION_ON, condition_on=CONDITION_ON, nunits=args.nunits, depth=args.nblocks)

    timings = {}
    for name, mode, model, train in [
        ("Diffusion_uncond", args.uncond, uncond, lambda: uncond.train_model(None, data_generator=gen_x, nepochs=args.epochs,
                                                                batch_size=args.batch_size, condition_on=CONDITION_ON, device=dev)),
        ("Diffusion_cond", args.cond, cond, lambda: cond.train_model(None, data_generator=gen_joint, nepochs=args.epochs,
                                                          batch_size=args.batch_size, condition_on=CONDITION_ON, device=dev)),
        ("CM", args.cm, cm, lambda: cm.train_model(X=None, data_generator=gen_joint, nepochs=args.epochs, batch_size=args.batch_size,
                                          device=dev, condition=CONDITION_ON, use_improved_training=True)),
    ]:
        have = os.path.exists(path(name))
        if mode == "load" and not have:
            sys.exit(f"[models] --{name.lower()} load: checkpoint not found: {path(name)}")
        if have and mode != "train":
            model.load_state_dict(torch.load(path(name), map_location=dev))
            print(f"[models] loaded {path(name)}")
        else:
            set_seed(args.seed)
            t0 = time.time()
            train()
            timings[name] = time.time() - t0
            torch.save(model.state_dict(), path(name))
            print(f"[models] trained {name} in {timings[name]:.0f}s -> {path(name)}")
        model.to(dev).eval()
    return uncond, cond, cm, timings


# ----------------------------------------------------------------------------
# MLGD / MLGD-F through the library (legacy optimize_LGD configuration)
# ----------------------------------------------------------------------------
def target_mixture():
    """G(Y) = P(Y | X = x*) in the (means, variances, weights) layout the samplers expect."""
    mu_t, S_t = compute_conditionals(MU_LIST, SIGMA_LIST, X_STAR)
    w_t = compute_alpha(MU_LIST, SIGMA_LIST, ALPHA, X_STAR)
    return filter_and_normalize(mu_t, S_t, w_t, threshold=0.001)


def run_mlgd(uncond, model_cond, use_cm, args, run_seed):
    mog_means, mog_vars, w = target_mixture()
    cfg = GuidanceConfig(
        n_cond=args.n_cond,
        n_mc=args.n_mc,
        init_fn=init_randn if args.init == "randn" else init_zeros,
        unsqueeze_init=True,
        perturb_fn=perturb_beta_ratio,
        perturb_square=False,
        aggregate_fn=aggregate_logsumexp,
        step_size_fn=lambda t, i: 1.0,
        timestep_source="range",
        ddim_step_style="condition_x",
        repeat_condition=True,
        condition_on_slice=(not use_cm),
        target_generator=lambda n, dev: generate_mog_samples_not_differentiable(n, mog_means, mog_vars, w).to(dev),
        compute_final_loss=True,
        device=args.device,
        seed=run_seed,
        tag="fig1_toy:" + ("MLGD-F" if use_cm else "MLGD"),
    )
    t0 = time.time()
    res = optimize_guided(uncond, model_cond, MMDLoss(kernel=RBF(device=args.device)), cfg)
    elapsed = time.time() - t0
    x_hat = res["x_final"].detach().float().view(-1)[0].cpu()
    mu_p, S_p = compute_conditionals(MU_LIST, SIGMA_LIST, x_hat.view(1))
    w_p = compute_alpha(MU_LIST, SIGMA_LIST, ALPHA, x_hat.view(1))
    l2_gmm = float(gmm_l2_distance(mu_p, S_p, w_p, mog_means, mog_vars, w))
    return {"x_hat": float(x_hat), "l2_gmm": l2_gmm, "l2_x": float(abs(x_hat - X_STAR[0])),
            "final_loss": float(res["final_loss"]), "time_s": elapsed}


# ----------------------------------------------------------------------------
# beta sweep: the same outer loop, analytic loss at the Tweedie point, stochastic DDIM
# ----------------------------------------------------------------------------
class _IdentitySampler:
    """Stands in for f_phi: returns the conditioning point itself, so the loss
    sees x0_hat directly (n_cond = 1)."""
    def sample(self, nsamples, condition_x=None, device=None, **kw):
        return condition_x


def run_beta_sweep(uncond, args):
    G = cond_params(X_STAR[0].to(args.device))
    G = tuple(g.detach() for g in G)

    def analytic_loss(pred_samples, target_samples):
        x0 = pred_samples.view(-1)[0]
        return l2_1d(cond_params(x0), G)

    samples = {}
    for beta in args.betas:
        cfg = GuidanceConfig(
            n_cond=1, n_mc=1,
            init_fn=init_randn, unsqueeze_init=True,
            perturb_fn=perturb_none, aggregate_fn=aggregate_single,
            # DPS convention: guidance per step scales with the noise increment beta_t,
            # so the total guidance over the trajectory is ~ beta * grad L (sum_t beta_t ~ 1)
            step_size_fn=lambda t, i, b=beta: b * float(uncond.betas[t]),
            timestep_source="range", ddim_step_style="condition_x",
            repeat_condition=False, condition_on_slice=False,
            target_generator=lambda n, dev: torch.zeros(0, device=dev),
            compute_final_loss=False,
            eta=1.0,                       # stochastic step: this is a sampler, not an optimiser
            device=args.device,
            tag=f"fig1_toy:beta={beta}",
        )
        xs = []
        t0 = time.time()
        for i in range(args.n_beta_samples):
            cfg.seed = args.seed * 1000 + i
            res = optimize_guided(uncond, _IdentitySampler(), analytic_loss, cfg)
            xs.append(float(res["x_final"].view(-1)[0]))
        samples[beta] = np.array(xs)
        print(f"[sweep] beta={beta}: {len(xs)} samples, mean {np.mean(xs):+.3f}, std {np.std(xs):.3f}, {time.time()-t0:.0f}s")
    return samples


def analytic_curves(args):
    x_grid = torch.linspace(-6, 6, 481)
    G = tuple(g.detach() for g in cond_params(X_STAR[0]))
    px = prior_density(x_grid)
    L = torch.tensor([float(l2_1d(cond_params(x), G)) for x in x_grid])
    Q = {}
    for beta in args.betas:
        q = px * torch.exp(-beta * L)
        Q[beta] = (q / torch.trapezoid(q, x_grid)).numpy()
    return x_grid.numpy(), px.numpy(), L.numpy(), Q


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def make_figure(out_png, mlgd, mlgdf, sweep, curves, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    x_grid, px, L, Q = curves
    c_data, c_true, c_mlgd, c_mlgdf, c_q = "steelblue", "black", "tomato", "seagreen", "darkorange"
    fs_title, fs_leg = 15, 10

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.28, hspace=0.42)

    # top left: joint
    ax = fig.add_subplot(gs[0, 0:2])
    set_seed(args.seed)
    xy = generate_mog_samples_not_differentiable(5000, MU_LIST, SIGMA_LIST, ALPHA).cpu().numpy()
    ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.3, color=c_data)
    ax.axvline(float(X_STAR[0]), color=c_true, ls="--", lw=2.2, label=r"true $x^\star$")
    ax.axvline(mlgd["x_hat"], color=c_mlgd, lw=2, alpha=0.9, label=r"MLGD $\hat{x}$")
    ax.axvline(mlgdf["x_hat"], color=c_mlgdf, lw=2, alpha=0.9, label=r"MLGD-F $\hat{x}$")
    ax.set_xlim(-6, 6); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(r"joint $\mathcal{P}(X, Y)$", fontsize=fs_title)
    ax.legend(fontsize=fs_leg, loc="upper right"); ax.grid(alpha=0.2)

    # top right: conditionals vs target
    ax = fig.add_subplot(gs[0, 2:4])
    y = torch.linspace(-2, 6, 500)
    G = cond_params(X_STAR[0])
    ax.plot(y, mixture_density_1d(y, G).detach(), color=c_true, ls="--", lw=2, label=r"target $\mathcal{G}(Y)$")
    ax.fill_between(y, mixture_density_1d(y, G).detach(), color=c_true, alpha=0.08)
    for r, c, name in [(mlgd, c_mlgd, "MLGD"), (mlgdf, c_mlgdf, "MLGD-F")]:
        p = cond_params(torch.tensor(r["x_hat"]))
        ax.plot(y, mixture_density_1d(y, p).detach(), color=c, lw=2,
                label=rf"{name}: $\mathcal{{P}}(Y\mid X=\hat{{x}})$, $L^2={r['l2_gmm']:.4f}$")
    ax.set_xlabel("y"); ax.set_title(r"induced conditional at $\hat{x}$ vs. target", fontsize=fs_title)
    ax.legend(fontsize=fs_leg); ax.grid(alpha=0.2)

    # bottom: beta sweep
    for i, beta in enumerate(args.betas[:4]):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(x_grid, px, color="grey", ls=":", lw=1.5, label=r"prior $\mathcal{P}(x)$")
        ax.plot(x_grid, Q[beta], color=c_q, lw=2, label=r"analytic $\mathcal{Q}_\beta(x)$")
        ax.fill_between(x_grid, Q[beta], color=c_q, alpha=0.15)
        ax.hist(sweep[beta], bins=36, range=(-6, 6), density=True, color=c_data, alpha=0.55,
                edgecolor="white", label="guided samples")
        ax.axvline(float(X_STAR[0]), color=c_true, ls="--", lw=1.2)
        ax.set_xlim(-6, 6); ax.set_ylim(0, 1.0); ax.set_xlabel("x")
        ax.set_title(rf"$\beta = {beta}$", fontsize=fs_title); ax.grid(alpha=0.15)
        if i == 0:
            ax.set_ylabel(r"$\mathcal{Q}_\beta(x)$")
            ax.legend(fontsize=fs_leg - 1, loc="upper right")

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[figure] wrote {out_png}")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[setup] device={args.device} seed={args.seed} quick={args.quick}")

    uncond, cond, cm, train_times = get_models(args)

    mlgd = run_mlgd(uncond, cond, use_cm=False, args=args, run_seed=args.seed)
    print(f"[MLGD]   x_hat={mlgd['x_hat']:+.4f}  L2 GMM={mlgd['l2_gmm']:.5f}  |x_hat-x*|={mlgd['l2_x']:.4f}  {mlgd['time_s']:.1f}s")
    mlgdf = run_mlgd(uncond, cm, use_cm=True, args=args, run_seed=args.seed)
    print(f"[MLGD-F] x_hat={mlgdf['x_hat']:+.4f}  L2 GMM={mlgdf['l2_gmm']:.5f}  |x_hat-x*|={mlgdf['l2_x']:.4f}  {mlgdf['time_s']:.1f}s")

    sweep = run_beta_sweep(uncond, args)
    curves = analytic_curves(args)
    make_figure(args.out, mlgd, mlgdf, sweep, curves, args)

    summary = {
        "args": vars(args), "train_times_s": train_times, "MLGD": mlgd, "MLGD-F": mlgdf,
        "beta_sweep": {str(b): {"mean": float(np.mean(s)), "std": float(np.std(s)), "n": int(len(s))} for b, s in sweep.items()},
    }
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] summary -> {os.path.splitext(args.out)[0] + '.json'}")


if __name__ == "__main__":
    main()
