"""
Figure 1 of the paper (the toy example), from the refactored library, with
simulation and plotting separated.

    python scripts/fig1_toy_example.py                 # simulate if no cache, then plot
    python scripts/fig1_toy_example.py --rerun-sim     # force a new simulation
    python scripts/fig1_toy_example.py --quick         # 1-2 min smoke test
    python scripts/fig1_toy_example.py --setting paper # the paper's 2-component toy

The simulation writes <out>_sim.json (all numbers) and <out>_sweep.npz (the
guided samples). Re-running without --rerun-sim only re-draws the figure from
those files, so layout changes never trigger a new run.

Settings (--setting):
  bimodal   default. 3-component joint in R^2, dim X = dim Y = 1: components
            (-3, 2), (-3, -2), (3, 0), shared covariance, equal weights. At the
            true optimum x* = -3 the conditional P(Y | X = x*) is a two-mode
            mixture in y; at x = +3 it is unimodal. The target G(Y) is bimodal,
            so matching it forces the optimiser to find x*.
  paper     the paper's App. B.8 toy: components (-3, 2), (3, -2), x* = -3;
            here G is a single Gaussian.
  hard      four well-separated x-clusters with different induced shapes on y,
            placed so that distance in x is unrelated to distributional fit:
            three modes at x = -6 (over-covering, the best wrong match, far from
            x*); one mode at x = -2 (misses a mode, the worst match, adjacent to
            x*); the target's two modes at x* = +2; one broad bump at x = +6.
            Equal prior mass per cluster; four basins with a known loss ranking.
  mog2d     the paper's 2D setting (11 components, SimulationParameters/
            mog_2d_full.txt, x* = -5, bimodal target at y = +-5).

Panels:
  top left    joint scatter, the true x* and the x-hat found by MLGD and MLGD-F
  top right   induced conditionals P(Y | X = x-hat) against the target G(Y), on y
  bottom      CDMS beta-sweep on X: histogram of guided samples against the
              analytic tilted density Q_beta(x) ~ P(x) exp(-beta L(x)), with L
              the exact L2 distance between P(Y|X=x) and G, and the prior P(x)
              in every panel. beta -> 0 gives the prior, beta -> inf a spike at x*.

Everything runs through guidance.optimize_guided:
  * MLGD   = legacy optimize_LGD config with the conditional *diffusion* sampler
  * MLGD-F = the same config with the conditional *consistency* sampler (CM=True)
  * beta-sweep = the same outer loop with n_cond = n_mc = 1, an identity "sampler"
    (so the loss is evaluated at the Tweedie point itself), the analytic L2-GMM
    loss, step size beta * beta_t (the DPS convention, so the total guidance over
    the trajectory is about beta * grad L), and a stochastic DDIM step
    (cfg.eta = 1): the DPS-style sampler of the paper's Figure 1 bottom row.

The three networks (unconditional DDPM on X, conditional DDPM on Y|X, conditional
iCT on Y|X; 3 blocks x 128 units, 3000 epochs, batch 1024, as in the paper) are
trained on the fly and cached under --ckpt-dir/<setting>/; --uncond/--cond/--cm
{auto,train,load} control each one, --retrain retrains all three.

Differences from the original notebook (conditional-matching-paper/simulations/
notebooks/toy_example_with_beta_sweep.ipynb), all deliberate: one shared
unconditional model (T = 100, all steps) for every panel instead of a separate
T = 999 DDPM with 20 DDIM steps for the sweep; the mixture weights of P(Y|X=x)
are differentiated in x in the sweep loss; notation follows the paper (G(Y) for
the target, x* for the truth, x-hat for the recovered input, Q_beta(x) below).

CPU or GPU, auto-detected; --device overrides. Full run ~5 min on CPU with cached
models, ~9 min including training.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
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

CONDITION_ON = 1          # dim(x) = 1, dim(y) = 1 in every setting here
DIFFUSION_STEPS = 100
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
@dataclass
class Setting:
    name: str
    mu_list: list
    Sigma_list: list
    alpha: torch.Tensor
    x_star: torch.Tensor

    @property
    def x_range(self):
        xs = torch.stack([m[0] for m in self.mu_list])
        sd = max(float(S[0, 0].sqrt()) for S in self.Sigma_list)
        return float(xs.min() - 3 * sd), float(xs.max() + 3 * sd)

    @property
    def y_range(self):
        ys = torch.stack([m[1] for m in self.mu_list])
        sd = max(float(S[1, 1].sqrt()) for S in self.Sigma_list)
        return float(ys.min() - 3 * sd), float(ys.max() + 3 * sd)


def make_setting(name):
    S = torch.tensor([[0.5, 0.15], [0.15, 0.4]])
    if name == "paper":
        return Setting(name, [torch.tensor([-3., 2.]), torch.tensor([3., -2.])], [S, S],
                       torch.tensor([0.5, 0.5]), torch.tensor([-3.]))
    if name == "bimodal":
        return Setting(name, [torch.tensor([-3., 2.]), torch.tensor([-3., -2.]), torch.tensor([3., 0.])],
                       [S, S, S], torch.tensor([1 / 3, 1 / 3, 1 / 3]), torch.tensor([-3.]))
    if name == "hard":
        # four x-clusters, 4 apart with x-sd 0.5, each inducing a different shape on y.
        # Placed so that closeness in x says nothing about closeness of P(Y|x) to G:
        #   x=-6: +2, 0, -2 (over-covering: the best wrong match, far from x*)
        #   x=-2: one mode at +2 (misses a mode: the worst match, right next to x*)
        #   x=+2: modes at +2, -2 (the target, x*)
        #   x=+6: one broad bump at 0 (blur)
        # every cluster carries 1/4 of the prior mass, split equally over its components
        clusters = [(-6., [2., 0., -2.], 0.4), (-2., [2.], 0.4), (2., [2., -2.], 0.4), (6., [0.], 3.0)]
        mu, Sig, w = [], [], []
        for cx, ys, vy in clusters:
            for cy in ys:
                mu.append(torch.tensor([cx, cy])); Sig.append(torch.tensor([[0.25, 0.0], [0.0, vy]])); w.append(1 / (4 * len(ys)))
        return Setting(name, mu, Sig, torch.tensor(w), torch.tensor([2.]))
    if name == "mog2d":
        loc = {}
        with open(os.path.join(REPO, "SimulationParameters", "mog_2d_full.txt")) as f:
            exec(f.read(), {"torch": torch}, loc)
        return Setting(name, [m.float() for m in loc["mu_list"]], [s.float() for s in loc["Sigma_list"]],
                       loc["alpha"].float(), torch.tensor([-5.]))
    raise ValueError(name)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", choices=["bimodal", "hard", "paper", "mog2d"], default="bimodal")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", default="checkpoints/toy")
    p.add_argument("--out", default=None, help="figure path; default results/fig1_<setting>.png")
    p.add_argument("--rerun-sim", action="store_true",
                   help="run the simulation again even if <out>_sim.json exists (default: plot from cache)")
    for name, what in [("uncond", "score network for P(X)"), ("cond", "conditional diffusion sampler for Y|X (MLGD)"),
                       ("cm", "conditional consistency sampler for Y|X (MLGD-F)")]:
        p.add_argument(f"--{name}", choices=["auto", "train", "load"], default="auto",
                       help=f"{what}: auto = load from --ckpt-dir if present else train; "
                            f"train = always (re)train and overwrite; load = require the checkpoint")
    p.add_argument("--retrain", action="store_true", help="shorthand for --uncond train --cond train --cm train")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--nblocks", type=int, default=3)
    p.add_argument("--nunits", type=int, default=128)
    p.add_argument("--n-cond", type=int, default=250)
    p.add_argument("--n-mc", type=int, default=1)
    p.add_argument("--n-restarts", type=int, default=10,
                   help="independent restarts per method; the best final loss is shown, all endpoints are saved")
    p.add_argument("--init", choices=["randn", "zeros"], default="randn",
                   help="x_T initialisation; the paper says N(0,I), the legacy code used zeros")
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 2.0, 8.0])
    p.add_argument("--n-beta-samples", type=int, default=200, help="guided trajectories per beta, run as one batch")
    p.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    a = p.parse_args()
    if a.out is None:
        a.out = f"results/fig1_{a.setting}.png"
    if a.quick:
        a.epochs, a.n_beta_samples, a.n_cond, a.n_restarts = 300, 50, 100, 2
    if a.retrain:
        a.uncond = a.cond = a.cm = "train"
    return a


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Exact 1D quantities, differentiable in x
# ----------------------------------------------------------------------------
def cond_params(st, x):
    """P(Y | X = x): means, variances, weights of a 1D mixture, differentiable in x."""
    means, variances, logw = [], [], []
    for mu, S, a in zip(st.mu_list, st.Sigma_list, st.alpha):
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
    """Exact squared L2 distance between two 1D Gaussian mixtures."""
    return _inner(*p, *p) - 2 * _inner(*p, *q) + _inner(*q, *q)


def prior_density(st, x):
    out = torch.zeros_like(x)
    for mu, S, a in zip(st.mu_list, st.Sigma_list, st.alpha):
        out = out + a * torch.exp(-(x - mu[0]) ** 2 / (2 * S[0, 0])) / math.sqrt(2 * math.pi * S[0, 0])
    return out


def mixture_density_1d(y, params):
    m, v, w = params
    return sum(wk * torch.exp(-(y - mk) ** 2 / (2 * vk)) / math.sqrt(2 * math.pi * vk) for mk, vk, wk in zip(m, v, w))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def get_models(st, args):
    ckpt_dir = os.path.join(args.ckpt_dir, st.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    dev = args.device

    def path(name):
        return os.path.join(ckpt_dir, f"{name}_seed{args.seed}.pt")

    gen_joint = partial(generate_mog_samples_not_differentiable, means=st.mu_list, variances=st.Sigma_list, weights=st.alpha)
    gen_x = partial(generate_mog_samples_not_differentiable, means=st.mu_list, variances=st.Sigma_list, weights=st.alpha,
                    kernel_func=lambda X: X[:, :CONDITION_ON])

    uncond = DiffusionModel(nfeatures=CONDITION_ON, nunits=args.nunits, condition=False,
                            diffusion_steps=DIFFUSION_STEPS, nblocks=args.nblocks)
    cond = DiffusionModel(nfeatures=2, nunits=args.nunits, condition=True, condition_on=CONDITION_ON,
                          diffusion_steps=DIFFUSION_STEPS, nblocks=args.nblocks)
    cm = ConsistencyModeliCT(nfeatures=2 - CONDITION_ON, condition_on=CONDITION_ON, nunits=args.nunits, depth=args.nblocks)

    timings = {}
    for name, mode, model, train in [
        ("Diffusion_uncond", args.uncond, uncond, lambda: uncond.train_model(
            None, data_generator=gen_x, nepochs=args.epochs, batch_size=args.batch_size, condition_on=CONDITION_ON, device=dev)),
        ("Diffusion_cond", args.cond, cond, lambda: cond.train_model(
            None, data_generator=gen_joint, nepochs=args.epochs, batch_size=args.batch_size, condition_on=CONDITION_ON, device=dev)),
        ("CM", args.cm, cm, lambda: cm.train_model(
            X=None, data_generator=gen_joint, nepochs=args.epochs, batch_size=args.batch_size,
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
# MLGD / MLGD-F (legacy optimize_LGD configuration)
# ----------------------------------------------------------------------------
def target_mixture(st):
    mu_t, S_t = compute_conditionals(st.mu_list, st.Sigma_list, st.x_star)
    w_t = compute_alpha(st.mu_list, st.Sigma_list, st.alpha, st.x_star)
    return filter_and_normalize(mu_t, S_t, w_t, threshold=0.001)


def run_mlgd(st, uncond, model_cond, use_cm, args, seed):
    mog_means, mog_vars, w = target_mixture(st)
    cfg = GuidanceConfig(
        n_cond=args.n_cond, n_mc=args.n_mc,
        init_fn=init_randn if args.init == "randn" else init_zeros, unsqueeze_init=True,
        perturb_fn=perturb_beta_ratio, perturb_square=False, aggregate_fn=aggregate_logsumexp,
        step_size_fn=lambda t, i: 1.0,
        timestep_source="range", ddim_step_style="condition_x",
        repeat_condition=True, condition_on_slice=(not use_cm),
        target_generator=lambda n, dev: generate_mog_samples_not_differentiable(n, mog_means, mog_vars, w).to(dev),
        compute_final_loss=True, device=args.device, seed=seed,
        tag=f"fig1_{st.name}:" + ("MLGD-F" if use_cm else "MLGD"),
    )
    t0 = time.time()
    res = optimize_guided(uncond, model_cond, MMDLoss(kernel=RBF(device=args.device)), cfg)
    elapsed = time.time() - t0
    x_hat = res["x_final"].detach().float().view(-1)[0].cpu()
    mu_p, S_p = compute_conditionals(st.mu_list, st.Sigma_list, x_hat.view(1))
    w_p = compute_alpha(st.mu_list, st.Sigma_list, st.alpha, x_hat.view(1))
    return {"x_hat": float(x_hat), "l2_gmm": float(gmm_l2_distance(mu_p, S_p, w_p, mog_means, mog_vars, w)),
            "l2_x": float(abs(x_hat - st.x_star[0])), "final_loss": float(res["final_loss"]), "time_s": elapsed,
            "n_steps": res["n_steps"], "best_loss": res["best_loss"], "trace": res["trace"], "config": cfg.describe(),
            "seed": seed}


def run_mlgd_restarts(st, uncond, model_cond, use_cm, args):
    """R independent restarts; the record of the best one (by the algorithm's own final
    loss, as in the paper's protocol) plus every restart's outcome."""
    name = "MLGD-F" if use_cm else "MLGD"
    runs = [run_mlgd(st, uncond, model_cond, use_cm, args, seed=args.seed * 100 + r) for r in range(args.n_restarts)]
    best = min(runs, key=lambda r: r["final_loss"])
    basin = 0.5 * min(abs(float(m[0]) - float(st.x_star[0])) for m in st.mu_list if abs(float(m[0]) - float(st.x_star[0])) > 1e-6) \
        if len({float(m[0]) for m in st.mu_list}) > 1 else 1.5
    hits = sum(abs(r["x_hat"] - float(st.x_star[0])) < basin for r in runs)
    out = dict(best)
    out.update({"n_restarts": len(runs), "hits_xstar_basin": int(hits), "basin_halfwidth": float(basin),
                "restarts": [{k: r[k] for k in ("seed", "x_hat", "final_loss", "l2_gmm", "l2_x", "time_s", "n_steps")} for r in runs],
                "traces": [r["trace"] for r in runs]})
    out.pop("trace", None)
    print(f"[{name}] best of {len(runs)}: x_hat={best['x_hat']:+.4f}  L2 GMM={best['l2_gmm']:.5f}  |x_hat-x*|={best['l2_x']:.4f}  "
          f"| {hits}/{len(runs)} restarts in the x* basin | {sum(r['time_s'] for r in runs):.0f}s total")
    return out


# ----------------------------------------------------------------------------
# beta sweep
# ----------------------------------------------------------------------------
class _IdentitySampler:
    """Stands in for f_phi: returns the conditioning point itself (n_cond = 1)."""
    def sample(self, nsamples, condition_x=None, device=None, **kw):
        return condition_x


def run_beta_sweep(st, uncond, args):
    G = tuple(g.detach() for g in cond_params(st, st.x_star[0].to(args.device)))

    def analytic_loss(pred_samples, target_samples):
        # one row per trajectory; rows are independent, so the gradient of the
        # sum is the per-trajectory gradient
        return torch.stack([l2_1d(cond_params(st, x0), G) for x0 in pred_samples.view(-1)]).sum()

    samples, diag = {}, {}
    for beta in args.betas:
        B = args.n_beta_samples
        cfg = GuidanceConfig(
            n_cond=1, n_mc=1, init_fn=lambda nf, dev, B=B: torch.randn(B, nf, device=dev), unsqueeze_init=False,
            perturb_fn=perturb_none, aggregate_fn=aggregate_single,
            step_size_fn=lambda t, i, b=beta: b * float(uncond.betas[t]),   # DPS convention
            timestep_source="range", ddim_step_style="condition_x",
            repeat_condition=False, condition_on_slice=False,
            target_generator=lambda n, dev: torch.zeros(0, device=dev),
            compute_final_loss=False, eta=1.0, device=args.device, tag=f"fig1_{st.name}:beta={beta}",
        )
        t0 = time.time()
        cfg.seed = args.seed * 1000 + int(round(10 * beta))
        res = optimize_guided(uncond, _IdentitySampler(), analytic_loss, cfg)
        xs = res["x_final"].view(-1).cpu().numpy()
        samples[beta] = xs
        # the batched loss is a sum over the B trajectories: store the per-trajectory mean per step
        diag[beta] = {"n_samples": int(B), "n_steps": int(res["n_steps"]), "time_s": time.time() - t0,
                      "loss_per_step_mean": [s["loss"] / B for s in res["trace"]],
                      "grad_norm_per_step": [s["grad_norm"] for s in res["trace"]],
                      "step_per_step": [s["step"] for s in res["trace"]], "config": cfg.describe()}
        print(f"[sweep] beta={beta}: {len(xs)} samples, mean {np.mean(xs):+.3f}, std {np.std(xs):.3f}, {time.time()-t0:.0f}s")
    return samples, diag


# ----------------------------------------------------------------------------
# Simulation cache
# ----------------------------------------------------------------------------
def sim_paths(out):
    stem = os.path.splitext(out)[0]
    return stem + "_sim.json", stem + "_sweep.npz"


def simulate(st, args):
    uncond, cond, cm, train_times = get_models(st, args)
    mlgd = run_mlgd_restarts(st, uncond, cond, use_cm=False, args=args)
    mlgdf = run_mlgd_restarts(st, uncond, cm, use_cm=True, args=args)
    sweep, diag = run_beta_sweep(st, uncond, args)
    sim = {"setting": st.name, "args": vars(args), "train_times_s": train_times, "MLGD": mlgd, "MLGD-F": mlgdf,
           "betas": list(args.betas),
           "beta_sweep": {str(b): {"mean": float(np.mean(s)), "std": float(np.std(s)), "n": int(len(s)), **diag[b]}
                          for b, s in sweep.items()}}
    pj, pn = sim_paths(args.out)
    with open(pj, "w") as f:
        json.dump(sim, f, indent=2)
    np.savez(pn, betas=np.array(args.betas), **{f"beta_{b}": s for b, s in sweep.items()})
    print(f"[sim] wrote {pj} and {pn}")
    return sim, sweep


def load_sim(args):
    pj, pn = sim_paths(args.out)
    if not (os.path.exists(pj) and os.path.exists(pn)):
        return None
    sim = json.load(open(pj))
    z = np.load(pn)
    sweep = {float(b): z[f"beta_{b}"] for b in z["betas"]}
    print(f"[sim] loaded cached simulation from {pj} (use --rerun-sim to recompute)")
    return sim, sweep


# ----------------------------------------------------------------------------
# Figure (reads only the cached simulation + analytic curves)
# ----------------------------------------------------------------------------
def analytic_curves(st, betas):
    lo, hi = st.x_range
    x_grid = torch.linspace(lo, hi, 601)
    G = tuple(g.detach() for g in cond_params(st, st.x_star[0]))
    px = prior_density(st, x_grid)
    L = torch.tensor([float(l2_1d(cond_params(st, x), G)) for x in x_grid])
    Q = {}
    for beta in betas:
        q = px * torch.exp(-beta * L)
        Q[beta] = (q / torch.trapezoid(q, x_grid)).numpy()
    return x_grid.numpy(), px.numpy(), Q


def make_figure(st, sim, sweep, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    betas = [float(b) for b in sim["betas"]][:4]
    mlgd, mlgdf = sim["MLGD"], sim["MLGD-F"]
    x_grid, px, Q = analytic_curves(st, betas)
    xlo, xhi = st.x_range
    # never let a recovered x-hat fall outside the frame
    allx = [mlgd["x_hat"], mlgdf["x_hat"]] + [q["x_hat"] for r in (mlgd, mlgdf) for q in r.get("restarts", [])]
    xlo = min(xlo, min(allx) - 0.6)
    xhi = max(xhi, max(allx) + 0.6)
    c_data, c_true, c_mlgd, c_mlgdf, c_q, c_prior = "steelblue", "black", "tab:red", "tab:green", "tab:purple", "tab:orange"
    fs_title, fs_leg = 15, 9.5
    x_star = float(st.x_star[0])

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.28, hspace=0.75)

    # top left: joint
    ax = fig.add_subplot(gs[0, 0:2])
    set_seed(0)
    xy = generate_mog_samples_not_differentiable(5000, st.mu_list, st.Sigma_list, st.alpha).cpu().numpy()
    ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.3, color=c_data)
    # different widths so both stay visible when the two solutions (nearly) coincide
    def lab(name, r):
        if r.get("n_restarts", 1) > 1:
            return rf"{name} $\hat{{x}}$ (best of {r['n_restarts']}; {r['hits_xstar_basin']}/{r['n_restarts']} in the $x^\star$ basin)"
        return rf"{name} $\hat{{x}}$"
    ax.axvline(mlgd["x_hat"], color=c_mlgd, lw=4, alpha=0.6, label=lab("MLGD", mlgd), zorder=3)
    ax.axvline(mlgdf["x_hat"], color=c_mlgdf, lw=1.6, alpha=1.0, label=lab("MLGD-F", mlgdf), zorder=4)
    ax.axvline(x_star, color=c_true, ls="--", lw=2.2, label=r"true $x^\star$", zorder=5)   # dashed drawn last
    # every restart's endpoint as a tick at the bottom of the panel
    y0, y1 = ax.get_ylim()
    for r, c, off in [(mlgd, c_mlgd, 0.0), (mlgdf, c_mlgdf, 0.035)]:
        xs = [q["x_hat"] for q in r.get("restarts", [])]
        if xs:
            ax.scatter(xs, [y0 + off * (y1 - y0)] * len(xs), marker="|", s=120, color=c, lw=1.5, zorder=5, clip_on=False)
    ax.set_xlim(xlo, xhi); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(r"joint $\mathcal{P}(X, Y)$", fontsize=fs_title)
    ax.legend(fontsize=fs_leg - 1, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False); ax.grid(alpha=0.2)

    # top right: conditionals vs target (on y)
    ax = fig.add_subplot(gs[0, 2:4])
    ylo, yhi = st.y_range
    y = torch.linspace(ylo, yhi, 600)
    G = cond_params(st, st.x_star[0])
    gy = mixture_density_1d(y, G).detach()
    ax.fill_between(y, gy, color=c_true, alpha=0.08, zorder=1)
    ax.plot(y, gy, color=c_true, lw=2.2, label=r"target $\mathcal{G}(Y)$", zorder=2)
    for r, c, lw, a, z, name in [(mlgd, c_mlgd, 4, 0.6, 3, "MLGD"), (mlgdf, c_mlgdf, 1.6, 1.0, 4, "MLGD-F")]:
        p = cond_params(st, torch.tensor(r["x_hat"]))
        ax.plot(y, mixture_density_1d(y, p).detach(), color=c, lw=lw, alpha=a, zorder=z,
                label=f"{name}: " + r"$\mathcal{P}(Y\mid X=\hat{x})$" + rf", $L^2={max(r['l2_gmm'], 0.0):.4f}$")
    ax.set_xlim(ylo, yhi); ax.set_xlabel("y")
    ax.set_title(r"induced conditional at $\hat{x}$ vs. target, on $y$", fontsize=fs_title)
    ax.legend(fontsize=fs_leg, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False); ax.grid(alpha=0.2)

    # bottom: beta sweep (on x); one shared y-limit from the analytic curves
    ymax = 1.08 * max(max(Q[b].max(), px.max()) for b in betas)
    for i, beta in enumerate(betas):
        ax = fig.add_subplot(gs[1, i])
        ax.hist(sweep[beta], bins=40, range=(xlo, xhi), density=True, color=c_data, alpha=0.5,
                edgecolor="white", linewidth=0.3, label="guided samples")
        ax.plot(x_grid, Q[beta], color=c_q, lw=2.0, label=r"analytic $\mathcal{Q}_\beta(x)$", zorder=3)
        ax.fill_between(x_grid, Q[beta], color=c_q, alpha=0.12, zorder=2)
        ax.axvline(x_star, color=c_true, ls="--", lw=1.8, label=r"true $x^\star$", zorder=4)
        # the prior is drawn last, dashed and dark, so it stays visible where it coincides with Q_beta
        ax.plot(x_grid, px, color=c_prior, ls="-.", lw=2.2, label=r"prior $\mathcal{P}(x)$", zorder=5)
        ax.set_xlim(xlo, xhi); ax.set_ylim(0, ymax); ax.set_xlabel("x")
        ax.set_title(rf"$\beta = {beta:g}$", fontsize=fs_title); ax.grid(alpha=0.15)
        if i == 0:
            ax.set_ylabel(r"$\mathcal{Q}_\beta(x)$")
        if i == len(betas) - 1:
            # the last panel (largest beta) has the most empty space: mass has left the decoys
            ax.legend(fontsize=fs_leg - 1, loc="upper right", frameon=False)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[figure] wrote {out_png}")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    st = make_setting(args.setting)
    print(f"[setup] setting={st.name} device={args.device} seed={args.seed} quick={args.quick}")

    cached = None if args.rerun_sim else load_sim(args)
    sim, sweep = cached if cached is not None else simulate(st, args)
    make_figure(st, sim, sweep, args.out)


if __name__ == "__main__":
    main()
