"""
Figure 1 of the paper (the toy example), from the refactored library, with
simulation and plotting separated.

    python experiments/synthetic/pipeline/fig1_toy_example.py                 # simulate if no cache, then plot
    python experiments/synthetic/pipeline/fig1_toy_example.py --rerun-sim     # force a new simulation
    python experiments/synthetic/pipeline/fig1_toy_example.py --quick         # 1-2 min smoke test
    python experiments/synthetic/pipeline/fig1_toy_example.py --setting paper # the paper's 2-component toy

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
  mog2d     the paper's 2D setting (11 components, experiments/synthetic/configs/mog/
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


from cdm.guidance import GuidanceConfig, optimize_guided, optimize_guided_population   # noqa: E402
from cdm.guidance.config import (init_randn, init_zeros, perturb_beta_ratio,    # noqa: E402
                             perturb_none, aggregate_logsumexp, aggregate_single)
from cdm.models.diffusion import DiffusionModel                                 # noqa: E402
from cdm.models.samplers.consistency import ConsistencyModeliCT                 # noqa: E402
from cdm.losses.functions import MMDLoss, RBF                                    # noqa: E402
from cdm.utils.dist import (generate_mog_samples_not_differentiable,            # noqa: E402
                        compute_conditionals, compute_alpha,
                        filter_and_normalize, gmm_l2_distance)

CONDITION_ON = 1          # dim(x) = 1, dim(y) = 1 in every setting here
DIFFUSION_STEPS = 100
CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "mog")


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
        # every cluster carries 1/4 of the prior mass, split equally over its components.
        # y-variances differ per cluster so the decoy conditionals do not overlap the target's curve:
        # narrow three-mode decoy, broad single-mode decoy, the target itself, and a very broad bump.
        # Each component is tilted (correlation rho between X and Y): with a diagonal covariance
        # P(Y|x) would be the same for every x inside a cluster, so the loss would be exactly flat
        # over the whole x* cluster and x* would be a set, not a point. With rho != 0 the modes of
        # P(Y|x) slide linearly in x, the loss is a bowl with a unique minimum at x*, and
        # |x_hat - x*| is meaningful. The target's two modes tilt in opposite directions (a "V").
        clusters = [(-6., [2., 0., -2.], 0.15, [0.5, 0.5, 0.5]),
                    (-2., [2.], 0.9, [-0.7]),
                    (2., [2., -2.], 0.4, [0.4, -0.4]),   # |rho| 0.7 made the x* basin <1 wide: no restart found it
                    (6., [0.], 3.0, [-0.5])]
        mu, Sig, w = [], [], []
        for cx, ys, vy, rhos in clusters:
            for cy, rho in zip(ys, rhos):
                cxy = rho * math.sqrt(0.25 * vy)
                mu.append(torch.tensor([cx, cy])); Sig.append(torch.tensor([[0.25, cxy], [cxy, vy]])); w.append(1 / (4 * len(ys)))
        return Setting(name, mu, Sig, torch.tensor(w), torch.tensor([2.]))
    if name == "mog2d":
        loc = {}
        with open(os.path.join(CONFIGS, "mog_2d_full.txt")) as f:
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
    p.add_argument("--rerun-sweep", action="store_true",
                   help="recompute only the beta sweep (and the loss curves) from the cached run; methods are kept")
    p.add_argument("--smc-npz", default=None,
                   help="panel (d) from fig_smc_correction.py output (e.g. results/fig_smc_hard.npz): plain guided "
                        "samples plus the twisted-SMC corrected population and the TV of each")
    p.add_argument("--layout", choices=["lr", "tb", "wide2", "wide3"], default="lr",
                   help="lr: left column (joint, conditionals, loss) + right 2x2 beta sweep; tb: top 1x3 + bottom 1x4")
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
    p.add_argument("--methods", nargs="+", default=["MLGD", "MLGD-F", "PMLGD", "PMLGD-F"],
                   choices=["MLGD", "MLGD-F", "PMLGD", "PMLGD-F"],
                   help="MLGD/MLGD-F: --n-restarts independent trajectories, best final loss wins; "
                        "PMLGD/PMLGD-F: the same budget as one resampled particle population")
    p.add_argument("--n-particles", type=int, default=None, help="population size (default: --n-restarts, same budget)")
    p.add_argument("--resample-beta", type=float, default=5.0, help="particle weights exp(-beta*loss); 0 disables")
    p.add_argument("--ess-frac", type=float, default=0.5)
    p.add_argument("--resample-every", type=int, default=1)
    p.add_argument("--weight-mode", choices=["fresh", "accumulated"], default="fresh")
    p.add_argument("--refine-steps", type=int, default=0, help="post-refinement gradient steps, applied to every method")
    p.add_argument("--refine-lr", type=float, default=0.05)
    p.add_argument("--refine-lambda", type=float, default=0.0)
    p.add_argument("--n-restarts", type=int, default=10,
                   help="independent restarts per method; the best final loss is shown, all endpoints are saved")
    p.add_argument("--init", choices=["randn", "zeros"], default="randn",
                   help="x_T initialisation; the paper says N(0,I), the legacy code used zeros")
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 2.0, 8.0, 16.0])
    p.add_argument("--panels", default="abcd",
                   help="panels to draw: a joint, b loss landscape, c conditionals, d beta sweep; letters are relabeled "
                        "consecutively (e.g. 'abd' draws a, b and the sweep as (c))")
    p.add_argument("--betas-plot", type=float, nargs="+", default=None,
                   help="betas shown in the sweep panel (a subset of the cached --betas); default: all, at most 4")
    p.add_argument("--figsize", type=float, nargs=2, default=None, help="figure width and height in inches")
    p.add_argument("--font-scale", type=float, default=1.0, help="multiplies every font size in the figure")
    p.add_argument("--compact", action="store_true",
                   help="short titles and labels, legends below the panels: for a small figure with large fonts")
    p.add_argument("--n-beta-samples", type=int, default=2000, help="guided trajectories per beta, run as one batch")
    p.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    p.add_argument("--traj", default=None,
                   help="wide3: JSON of guided trajectories for panel (e): {method: {x0: [restarts][steps], selected: r}}")
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
    """P(Y | X = x): means, variances, weights of a 1D mixture, differentiable in x.

    x may be a scalar or a batch of any shape; the mixture index is the last dimension."""
    x = torch.as_tensor(x)
    means, variances, logw = [], [], []
    for mu, S, a in zip(st.mu_list, st.Sigma_list, st.alpha):
        mu, S = mu.to(x), S.to(x)
        sxx, sxy, syy = S[0, 0], S[0, 1], S[1, 1]
        means.append(mu[1] + sxy / sxx * (x - mu[0]))
        variances.append((syy - sxy * sxy / sxx).expand_as(x))
        logw.append(torch.log(a.to(x)) - 0.5 * torch.log(2 * math.pi * sxx) - (x - mu[0]) ** 2 / (2 * sxx))
    return torch.stack(means, -1), torch.stack(variances, -1), torch.softmax(torch.stack(logw, -1), -1)


def _inner(m1, v1, w1, m2, v2, w2):
    s = v1[..., :, None] + v2[..., None, :]
    d = m1[..., :, None] - m2[..., None, :]
    return (w1[..., :, None] * w2[..., None, :] * torch.exp(-d * d / (2 * s)) / torch.sqrt(2 * math.pi * s)).sum((-2, -1))


def l2_1d(p, q):
    """Exact squared L2 distance between two 1D Gaussian mixtures (batched over leading dims)."""
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


def _basin_halfwidth(st):
    others = [abs(float(m[0]) - float(st.x_star[0])) for m in st.mu_list if abs(float(m[0]) - float(st.x_star[0])) > 1e-6]
    return 0.5 * min(others) if others else 1.5


def _eval_x(st, x_hat, mog):
    mog_means, mog_vars, w = mog
    mu_p, S_p = compute_conditionals(st.mu_list, st.Sigma_list, x_hat.view(1))
    w_p = compute_alpha(st.mu_list, st.Sigma_list, st.alpha, x_hat.view(1))
    return {"x_hat": float(x_hat), "l2_gmm": float(gmm_l2_distance(mu_p, S_p, w_p, mog_means, mog_vars, w)),
            "l2_x": float(abs(x_hat - st.x_star[0]))}


def run_particles(st, uncond, model_cond, use_cm, args):
    """One resampled population with the restart budget: n_particles = n_restarts trajectories."""
    name = "PMLGD-F" if use_cm else "PMLGD"
    mog = target_mixture(st)
    R = args.n_particles or args.n_restarts
    cfg = _mlgd_cfg(st, model_cond, use_cm, args, seed=args.seed * 100, mog=mog)
    cfg.n_particles = R
    cfg.resample_beta = None if args.resample_beta <= 0 else args.resample_beta
    cfg.ess_frac, cfg.resample_every, cfg.weight_mode = args.ess_frac, args.resample_every, args.weight_mode
    cfg.tag = f"fig1_{st.name}:{name}"
    t0 = time.time()
    res = optimize_guided_population(uncond, model_cond, MMDLoss(kernel=RBF(device=args.device)), cfg)
    elapsed = time.time() - t0
    xs = res["x_particles"].detach().float().view(-1).cpu()
    particles = [{**_eval_x(st, x, mog), "final_loss": float(l)} for x, l in zip(xs, res["final_losses"])]
    basin = _basin_halfwidth(st)
    hits = sum(abs(q["x_hat"] - float(st.x_star[0])) < basin for q in particles)
    best = particles[res["best_particle"]]
    out = {**best, "time_s": elapsed, "n_steps": res["n_steps"], "best_loss": res["best_loss"], "config": cfg.describe(),
           "seed": cfg.seed, "n_restarts": R, "n_particles": R, "hits_xstar_basin": int(hits), "basin_halfwidth": float(basin),
           "n_resamples": res["n_resamples"], "resample_steps": res["resample_steps"],
           "restarts": particles, "population_trace": res["trace"],
           "ess_per_step": [s["ess"] for s in res["trace"]]}
    if res.get("refine") is not None:
        rf = res["refine"]
        out["refine"] = {"traces": rf["traces"], "x_before": rf["x_before"].view(-1).tolist(),
                         "final_losses_before": rf["final_losses_before"],
                         "hits_before": int(sum(abs(float(x) - float(st.x_star[0])) < basin for x in rf["x_before"].view(-1)))}
    print(f"[{name}] best of {R} particles: x_hat={best['x_hat']:+.4f}  L2 GMM={best['l2_gmm']:.5f}  |x_hat-x*|={best['l2_x']:.4f}  "
          f"| {hits}/{R} particles in the x* basin | {res['n_resamples']} resamplings"
          + (f" (steps {res['resample_steps'][:6]}{'...' if len(res['resample_steps']) > 6 else ''})" if res["n_resamples"] else "")
          + (f" | refine: {out['refine']['hits_before']}/{R} in basin before" if "refine" in out else "")
          + f" | {elapsed:.0f}s")
    return out


def _mlgd_cfg(st, model_cond, use_cm, args, seed, mog):
    mog_means, mog_vars, w = mog
    return GuidanceConfig(
        n_cond=args.n_cond, n_mc=args.n_mc,
        init_fn=init_randn if args.init == "randn" else init_zeros, unsqueeze_init=True,
        perturb_fn=perturb_beta_ratio, perturb_square=False, aggregate_fn=aggregate_logsumexp,
        step_size_fn=lambda t, i: 1.0,
        timestep_source="range", ddim_step_style="condition_x",
        repeat_condition=True, condition_on_slice=(not use_cm),
        target_generator=lambda n, dev: generate_mog_samples_not_differentiable(n, mog_means, mog_vars, w).to(dev),
        compute_final_loss=True, device=args.device, seed=seed,
        refine_steps=args.refine_steps, refine_lr=args.refine_lr, refine_lambda=args.refine_lambda,
        tag=f"fig1_{st.name}:" + ("MLGD-F" if use_cm else "MLGD"),
    )


def run_mlgd(st, uncond, model_cond, use_cm, args, seed):
    mog = target_mixture(st)
    cfg = _mlgd_cfg(st, model_cond, use_cm, args, seed, mog)
    t0 = time.time()
    res = optimize_guided(uncond, model_cond, MMDLoss(kernel=RBF(device=args.device)), cfg)
    elapsed = time.time() - t0
    x_hat = res["x_final"].detach().float().view(-1)[0].cpu()
    out = {**_eval_x(st, x_hat, mog), "final_loss": float(res["final_loss"]), "time_s": elapsed,
           "n_steps": res["n_steps"], "best_loss": res["best_loss"], "trace": res["trace"], "config": cfg.describe(),
           "seed": seed}
    if res.get("refine") is not None:
        rf = res["refine"]
        out["refine"] = {"trace": rf["trace"], "x_before": float(rf["x_before"].view(-1)[0]),
                         "final_loss_before": rf["final_loss_before"]}
    return out


def empirical_loss_curve(st, model_cond, use_cm, args, n_grid=41, n_seeds=3):
    """The plug-in loss the optimiser actually sees, on a grid of x: MMD between n_cond draws of the
    trained sampler at x and n_cond draws of the target, repeated over seeds. Unlike the population
    loss L(x) it carries estimator noise and sampler error."""
    mog_means, mog_vars, w = target_mixture(st)
    loss_fn = MMDLoss(kernel=RBF(device=args.device))
    xlo, xhi = st.x_range
    grid = np.linspace(xlo, xhi, n_grid)
    vals = np.zeros((n_seeds, n_grid))
    t0 = time.time()
    with torch.no_grad():
        for si in range(n_seeds):
            set_seed(args.seed * 100 + si)
            for gi, x in enumerate(grid):
                cond = torch.full((args.n_cond, CONDITION_ON), float(x), device=args.device)
                out = model_cond.sample(nsamples=args.n_cond, condition_x=cond, device=args.device)
                y = out[0] if isinstance(out, tuple) else out
                if not use_cm:
                    y = y[:, model_cond.condition_on:]
                tgt = generate_mog_samples_not_differentiable(args.n_cond, mog_means, mog_vars, w).to(args.device)
                vals[si, gi] = float(loss_fn(y, tgt))
    print(f"[loss-curve] {'MLGD-F' if use_cm else 'MLGD'}: {n_grid} x-values x {n_seeds} seeds, n_cond={args.n_cond}, {time.time()-t0:.0f}s")
    return {"x": grid.tolist(), "values": vals.tolist(), "n_cond": int(args.n_cond), "n_seeds": n_seeds,
            "time_s": time.time() - t0}


def run_mlgd_restarts(st, uncond, model_cond, use_cm, args):
    """R independent restarts; the record of the best one (by the algorithm's own final
    loss, as in the paper's protocol) plus every restart's outcome."""
    name = "MLGD-F" if use_cm else "MLGD"
    runs = [run_mlgd(st, uncond, model_cond, use_cm, args, seed=args.seed * 100 + r) for r in range(args.n_restarts)]
    best = min(runs, key=lambda r: r["final_loss"])
    basin = _basin_halfwidth(st)
    hits = sum(abs(r["x_hat"] - float(st.x_star[0])) < basin for r in runs)
    out = dict(best)
    out.update({"n_restarts": len(runs), "hits_xstar_basin": int(hits), "basin_halfwidth": float(basin),
                "restarts": [{k: r[k] for k in ("seed", "x_hat", "final_loss", "l2_gmm", "l2_x", "time_s", "n_steps", "refine") if k in r}
                             for r in runs],
                "traces": [r["trace"] for r in runs]})
    out.pop("trace", None)
    if "refine" in best:
        out["hits_before_refine"] = int(sum(abs(r["refine"]["x_before"] - float(st.x_star[0])) < basin for r in runs))
    print(f"[{name}] best of {len(runs)}: x_hat={best['x_hat']:+.4f}  L2 GMM={best['l2_gmm']:.5f}  |x_hat-x*|={best['l2_x']:.4f}  "
          f"| {hits}/{len(runs)} restarts in the x* basin"
          + (f" ({out['hits_before_refine']}/{len(runs)} before refinement)" if "hits_before_refine" in out else "")
          + f" | {sum(r['time_s'] for r in runs):.0f}s total")
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
        # one row per trajectory, evaluated for the whole batch at once; rows are
        # independent, so the gradient of the sum is the per-trajectory gradient
        return l2_1d(cond_params(st, pred_samples.view(-1)), G).sum()

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
    results = {}
    for name in args.methods:
        use_cm = name.endswith("-F")
        model = cm if use_cm else cond
        results[name] = run_particles(st, uncond, model, use_cm, args) if name.startswith("P") \
            else run_mlgd_restarts(st, uncond, model, use_cm, args)
    sim = {"setting": st.name, "args": vars(args), "train_times_s": train_times, **results, "methods": list(args.methods)}
    return _sweep_and_save(st, uncond, cond, cm, args, sim)


def _sweep_and_save(st, uncond, cond, cm, args, sim):
    sweep, diag = run_beta_sweep(st, uncond, args)
    curves = {"MLGD": empirical_loss_curve(st, cond, False, args), "MLGD-F": empirical_loss_curve(st, cm, True, args)}
    sim.update({"empirical_loss": curves, "betas": list(args.betas),
                "beta_sweep": {str(b): {"mean": float(np.mean(s)), "std": float(np.std(s)), "n": int(len(s)), **diag[b]}
                               for b, s in sweep.items()}})
    pj, pn = sim_paths(args.out)
    with open(pj, "w") as f:
        json.dump(sim, f, indent=2)
    np.savez(pn, betas=np.array(args.betas), **{f"beta_{b}": s for b, s in sweep.items()})
    print(f"[sim] wrote {pj} and {pn}")
    return sim, sweep


def rerun_sweep(st, args):
    """Keep the cached method results, recompute the beta sweep and the loss curves."""
    cached = load_sim(args)
    assert cached is not None, "--rerun-sweep needs a cached simulation"
    sim, _ = cached
    uncond, cond, cm, _ = get_models(st, args)
    return _sweep_and_save(st, uncond, cond, cm, args, sim)


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


def load_smc(path, betas):
    """Plain and twisted-SMC-corrected guided samples saved by fig_smc_correction.py (<stem>.npz + <stem>.json)."""
    d = np.load(path)
    smc = {"plain": {b: d[f"plain_{float(b)}"] for b in betas}, "corr": {b: d[f"corr_{float(b)}"] for b in betas}}
    js = os.path.splitext(path)[0] + ".json"
    if os.path.exists(js):
        with open(js) as f:
            per = json.load(f)["per_beta"]
        smc["tv"] = {k: (v["tv_plain_vs_Q"], v["tv_corrected_vs_Q"]) for k, v in per.items()}
    return smc


def _kde(samples, grid, bw=None):
    """Gaussian kernel density of 1-D samples on grid (Silverman bandwidth unless given)."""
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if bw is None:
        bw = 0.9 * min(x.std(), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.34) * len(x) ** (-0.2)
    dens = np.zeros_like(grid, dtype=float)
    for c in np.array_split(x, max(1, len(x) // 20000)):
        dens += np.exp(-0.5 * ((grid[:, None] - c[None, :]) / bw) ** 2).sum(1)
    return dens / (len(x) * bw * np.sqrt(2 * np.pi))


def draw_trajectories(ax, traj, x_star, xlo, xhi, fs_leg, style):
    """Panel (e): the clean estimate x0_hat(x_t) of every restart along the reverse process (step on the
    x-axis, noise -> end), per method; the restart each method returns is drawn on top."""
    for name in ("MLGD-F", "MLGD"):
        if name not in traj:
            continue
        X = np.asarray(traj[name]["x0"])                 # [restarts, steps]
        steps = np.arange(X.shape[1])
        for r in range(X.shape[0]):
            ax.plot(steps, X[r], color=style[name]["color"], lw=0.5, alpha=0.35, zorder=2)
        sel = traj[name].get("selected")
        if sel is not None:
            ax.plot(steps, X[sel], color=style[name]["color"], lw=1.4, alpha=1.0, zorder=3)
    ax.axhline(x_star, color="black", ls="--", lw=0.9, zorder=4)
    ax.set_ylim(xlo, xhi); ax.set_xlim(0, X.shape[1] - 1)
    ax.set_xlabel("reverse step"); ax.set_ylabel(r"$\hat{x}_0(x_t)$")


def make_figure(st, sim, sweep, out_png, layout="lr", smc=None, panels="abcd", betas_plot=None, figsize=None,
                font_scale=1.0, compact=False, traj=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    betas = [float(b) for b in sim["betas"]][:4]
    if layout == "wide3" and betas_plot:            # the betas of the (dense) SMC file, not of the cached sweep
        betas = [float(b) for b in betas_plot]
    elif betas_plot:
        betas = [b for b in betas if any(abs(b - float(q)) < 1e-9 for q in betas_plot)]
    show_cond = "c" in panels
    plt.rcParams.update({"font.size": 10 * font_scale, "axes.labelsize": 11 * font_scale,
                         "xtick.labelsize": 9 * font_scale, "ytick.labelsize": 9 * font_scale})
    methods = [m for m in sim.get("methods", ["MLGD", "MLGD-F"]) if m in sim]
    # one style per method, fixed across figures: wide translucent lines for the diffusion sampler,
    # thin lines on top for the consistency sampler; particle variants in blue tones
    STYLE = {"MLGD":    dict(color="tab:red",     lw=4.0, alpha=0.6, z=3),
             "MLGD-F":  dict(color="tab:green",   lw=1.6, alpha=1.0, z=4),
             "PMLGD":   dict(color="navy",        lw=4.0, alpha=0.5, z=3),
             "PMLGD-F": dict(color="darkmagenta", lw=1.6, alpha=1.0, z=4)}
    res = {m: sim[m] for m in methods}
    x_grid, px, Q = analytic_curves(st, betas)
    xlo, xhi = st.x_range
    allx = [r["x_hat"] for r in res.values()] + [q["x_hat"] for r in res.values() for q in r.get("restarts", [])]
    xlo, xhi = min(xlo, min(allx) - 0.6), max(xhi, max(allx) + 0.6)
    # palette: x* black dashed, G black dashed, Q_beta purple, prior orange dash-dot; dashed curves are drawn last
    c_data, c_true, c_q, c_prior = "steelblue", "black", "tab:purple", "tab:orange"
    c_mlgd, c_mlgdf = STYLE["MLGD"]["color"], STYLE["MLGD-F"]["color"]

    def vline(ax, m, label=None):
        st_ = STYLE[m]
        if layout == "wide3":   # a triangle on the bottom edge of the panel instead of a full-height line
            ax.plot([res[m]["x_hat"]], [0.0], ls="none", marker="^", ms=7, color=st_["color"], mec="white", mew=0.5,
                    transform=ax.get_xaxis_transform(), clip_on=False, zorder=8, label=label)
            return
        ax.axvline(res[m]["x_hat"], color=st_["color"], lw=st_["lw"], alpha=st_["alpha"], zorder=st_["z"], label=label)
    fs_title, fs_leg = 13 * font_scale, 9 * font_scale
    x_star = float(st.x_star[0])

    # layout "lr": left column = joint / loss landscape (same x-axis) / conditionals on y;
    #              right column = the four beta panels stacked, sharing the x-axis so basins line up
    # layout "tb": top row = joint / loss / conditionals, bottom row = 1x4 beta sweep
    tv_panel, ax_tv = False, None
    if layout == "lr":
        fig = plt.figure(figsize=figsize or ((14, 12) if show_cond else (14, 8)))
        outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.55 if compact else 0.33)
        if show_cond:
            left = outer[0].subgridspec(2, 1, height_ratios=[2.1, 1], hspace=0.45)
            left_top = left[0].subgridspec(2, 1, height_ratios=[1.2, 1], hspace=0.08)   # joint + loss share x
        else:
            left_top = outer[0].subgridspec(2, 1, height_ratios=[1.2, 1], hspace=0.08)
        right = outer[1].subgridspec(len(betas), 1, hspace=0.12)
        ax_joint = fig.add_subplot(left_top[0])
        ax_loss = fig.add_subplot(left_top[1], sharex=ax_joint)
        ax_cond = fig.add_subplot(left[1]) if show_cond else None
        top_axes = [ax_joint, ax_loss, ax_cond]
        sweep_axes = []
        for i in range(len(betas)):
            sweep_axes.append(fig.add_subplot(right[i], sharex=sweep_axes[0] if sweep_axes else None))
        sweep_ylabel = lambda i: True
        sweep_xlabel = lambda i: i == len(betas) - 1
    elif layout == "wide3":
        # full width, one 2 x 3 grid so that (a)-(d) have the same size: joint over loss (same x) |
        # conditionals over trajectories | the betas stacked over both rows; one legend under (a)-(d), one under (e)
        fig = plt.figure(figsize=figsize or (6.75, 3.15))     # legends inside (d) and (e): no legend rows below
        outer = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1, 1],
                                  wspace=0.42, hspace=0.42, left=0.07, right=0.99, top=0.933, bottom=0.11)
        left_col = outer[:, 0].subgridspec(2, 1, hspace=0.06)    # (a) and (b) share x and touch
        ax_joint = fig.add_subplot(left_col[0])
        ax_loss = fig.add_subplot(left_col[1], sharex=ax_joint)
        mid_col = outer[:, 1].subgridspec(2, 1, hspace=0.3)      # (c), (d): only (c)'s x-axis between them
        ax_cond = fig.add_subplot(mid_col[0])
        ax_traj = fig.add_subplot(mid_col[1])
        top_axes = [ax_joint, ax_loss, ax_cond]
        tv_panel = len(betas) == 1            # (e) the densities at one beta, (f) TV to Q_beta against beta
        right = outer[:, 2].subgridspec(len(betas), 1, hspace=0.12) if not tv_panel else None
        sweep_axes = [fig.add_subplot(outer[0, 2])] if tv_panel else []
        ax_tv = fig.add_subplot(outer[1, 2]) if tv_panel else None
        for i in range(0 if tv_panel else len(betas)):
            sweep_axes.append(fig.add_subplot(right[i], sharex=sweep_axes[0] if sweep_axes else None,
                                              sharey=sweep_axes[0] if sweep_axes else None))
        sweep_ylabel = lambda i: i == len(betas) // 2
        sweep_xlabel = lambda i: i == len(betas) - 1
    elif layout == "wide2":
        # one row, full page width: joint over loss (same x-axis) | conditionals on y | 2 x 2 beta sweep;
        # one legend row for the whole figure
        fig = plt.figure(figsize=figsize or (6.75, 2.75))
        outer = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.12, 0.92, 1.16], wspace=0.36,
                                  left=0.065, right=0.995, top=0.92, bottom=0.25)
        left_top = outer[0].subgridspec(2, 1, height_ratios=[1.15, 1], hspace=0.08)
        ax_joint = fig.add_subplot(left_top[0])
        ax_loss = fig.add_subplot(left_top[1], sharex=ax_joint)
        ax_cond = fig.add_subplot(outer[1])
        top_axes = [ax_joint, ax_loss, ax_cond]
        grid = outer[2].subgridspec(2, 2, hspace=0.1, wspace=0.08)
        sweep_axes = []
        for i in range(len(betas)):
            sweep_axes.append(fig.add_subplot(grid[i // 2, i % 2], sharex=sweep_axes[0] if sweep_axes else None,
                                              sharey=sweep_axes[0] if sweep_axes else None))
        sweep_ylabel = lambda i: i % 2 == 0
        sweep_xlabel = lambda i: i >= 2
    else:
        fig = plt.figure(figsize=(16, 7.5))
        outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.15, 1], hspace=0.7)
        top = outer[0].subgridspec(1, 3, wspace=0.42)
        bot = outer[1].subgridspec(1, 4, wspace=0.28)
        top_axes = [fig.add_subplot(top[i]) for i in range(3)]
        sweep_axes = [fig.add_subplot(bot[i]) for i in range(4)]
        sweep_ylabel = lambda i: i == 0
        sweep_xlabel = lambda i: True
    ax_joint, ax_loss, ax_cond = top_axes
    shown = [a_ for a_ in [ax_joint, ax_loss, ax_cond] if a_ is not None]
    for letter, ax_ in zip("abc", shown):   # inside the top-left corner (empty in every panel)
        ax_.text(0.012, 0.965, f"({letter})", transform=ax_.transAxes, fontsize=fs_title, fontweight="bold",
                 ha="left", va="top", zorder=10)
    sweep_letter = "(e)" if layout == "wide3" else f"({'abcd'[len(shown)]})"
    W2 = layout in ("wide2", "wide3")
    W3 = layout == "wide3"
    if not compact and not W2:
        sweep_axes[0].text(-0.08, 1.04, sweep_letter, transform=sweep_axes[0].transAxes, fontsize=fs_title,
                           fontweight="bold", ha="right", va="bottom")

    def lab(name, r):
        return rf"{name} $\hat{{x}}$"

    # ---- left, top: the joint ------------------------------------------------
    ax = ax_joint
    set_seed(0)
    xy = generate_mog_samples_not_differentiable(5000, st.mu_list, st.Sigma_list, st.alpha).cpu().numpy()
    ax.scatter(xy[:, 0], xy[:, 1], s=7, alpha=0.3, color=c_data)
    for m in methods:
        vline(ax, m, lab(m, res[m]))
    if W3:   # x* as a black star on the bottom edge, with the returned x-hat (triangles) on top of it
        ax.plot([x_star], [0.0], ls="none", marker="*", ms=12, color=c_true, mec="white", mew=0.5,
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=7)
        ax.annotate(r"$x^*$", (x_star, 0.0), xycoords=ax.get_xaxis_transform(), xytext=(5, 4),
                    textcoords="offset points", ha="left", va="bottom", fontsize=fs_title - 1, zorder=6)
    else:
        ax.axvline(x_star, color=c_true, ls="--", lw=2, zorder=5)
    y0, y1 = ax.get_ylim()
    if not W3:
        ax.annotate(r"$x^*$", (x_star, y1), xytext=(-4, -4), textcoords="offset points",
                    ha="right", va="top", fontsize=fs_title - 1, zorder=6)
    ax.set_xlim(xlo, xhi); ax.set_ylabel("y")
    ax.set_title(r"joint $\mathcal{P}(X, Y)$, recovered $\hat{x}$" if (compact or W2) else
                 r"joint $\mathcal{P}(X, Y)$ and the recovered $\hat{x}$", fontsize=fs_title); ax.grid(alpha=0.2)
    if layout in ("lr", "wide2", "wide3"):
        ax.tick_params(labelbottom=False)   # x-axis numbers only once, on the loss panel below
    else:
        ax.set_xlabel("x")
        ax.legend(fontsize=fs_leg - 1, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    # ---- left, middle: induced conditionals on y ------------------------------
    ax = ax_cond
    if ax is None:
        ax = fig.add_subplot(111); ax.set_visible(False)   # panel (c) not drawn: draw into a hidden axis
    ylo, yhi = max(st.y_range[0], -5.0), min(st.y_range[1], 5.0)
    y = torch.linspace(ylo, yhi, 600)
    G = cond_params(st, st.x_star[0])
    gy = mixture_density_1d(y, G).detach()
    others = sorted({round(float(m[0]), 3) for m in st.mu_list} - {round(x_star, 3)})
    c_decoy = ["tab:brown", "tab:pink", "tab:cyan", "tab:olive"]
    for k, cx in enumerate(others):   # what a wrong x induces: dotted, own colour, labelled at its peak
        pk = cond_params(st, torch.tensor(cx))
        dk = mixture_density_1d(y, pk).detach().numpy()
        ax.plot(y, dk, color=c_decoy[k % len(c_decoy)], ls=":", lw=2.2, zorder=1,
                label=rf"$\mathcal{{P}}(Y\mid X={cx:g})$")
        j = int(np.argmax(dk))
        # label placement alternates so labels do not sit on other curves: above, right of, left of the peak
        place = [dict(xytext=(0, 4), ha="center", va="bottom"), dict(xytext=(26, -16), ha="left", va="top"),
                 dict(xytext=(-16, 2), ha="right", va="bottom")][k % 3]
        if layout in ("wide2", "wide3"):
            place = dict(xytext=(0, 1), ha="center", va="bottom") if k != 1 else dict(xytext=(-4, -2), ha="right", va="top")
        anchor = (float(y[j]), float(dk[j]))
        if layout == "wide3" and abs(cx - 6) < 1e-6:   # left of its peak, between the green and brown modes
            anchor, place = (-0.95, 0.31), dict(xytext=(0, 0), ha="center", va="bottom")
        ax.annotate(rf"$x={cx:g}$", anchor, textcoords="offset points",
                    fontsize=fs_leg - (1 if layout == "wide3" else 0), color=c_decoy[k % len(c_decoy)], **place)
    ax.fill_between(y, gy, color=c_true, alpha=0.08, zorder=1)
    ax.plot([], [], color=c_true, ls="--", lw=2, label=r"target $\mathcal{G}(Y)$")   # legend slot before the methods
    # the recovered conditionals often coincide (every method that found x* draws the same curve), so
    # they are drawn as nested outlines: decreasing width in a fixed order, each on top of the previous
    widths = {"MLGD": 1.8 if layout == "wide3" else 6.0, "PMLGD": 4.2, "PMLGD-F": 2.6,
              "MLGD-F": 1.8 if layout == "wide3" else 1.4}
    for z, name in enumerate([m for m in ("MLGD", "PMLGD", "PMLGD-F", "MLGD-F") if m in methods]):
        r, st_ = res[name], STYLE[name]
        p = cond_params(st, torch.tensor(r["x_hat"]))
        ax.plot(y, mixture_density_1d(y, p).detach(), color=st_["color"], lw=widths[name], alpha=0.9, zorder=2 + z,
                label=f"{name}: " + r"$\mathcal{P}(Y\mid X=\hat{x})$" + rf", $L^2={max(r['l2_gmm'], 0.0):.4f}$")
    # the target is drawn last (dashed, like x*) so it stays visible where a recovered x-hat matches it exactly
    if layout == "wide3":   # solid, same width as the methods; labelled on its left flank
        ax.plot(y, gy, color=c_true, lw=1.8, zorder=5)
        ax.annotate(r"$\mathcal{G}$ ($x=2$)", (-2.62, 0.2), xytext=(-2, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=fs_leg - 1, color=c_true)
    else:
        ax.plot(y, gy, color=c_true, ls="--", lw=2, zorder=5)
    ax.set_xlim(ylo, yhi); ax.set_ylim(0, min(0.61, 1.1 * ax.get_ylim()[1] / 1.05)); ax.set_ylabel("density")
    ax.set_xlabel("y", labelpad=0 if layout == "wide3" else None)
    ax.set_title(r"$\mathcal{P}(Y\mid X=\hat{x})$ vs. the target $\mathcal{G}$" if W2 else
                 r"induced conditional $\mathcal{P}(Y\mid X=x)$ vs. the target, on $y$", fontsize=fs_title)
    if not W2:
        ax.legend(fontsize=fs_leg - 1, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3 if len(methods) <= 2 else 2,
                  frameon=False)
    ax.grid(alpha=0.2)

    # ---- left, bottom: the loss landscape L(x) --------------------------------
    ax = ax_loss
    Gd = tuple(g.detach() for g in G)
    L = l2_1d(cond_params(st, torch.as_tensor(x_grid, dtype=torch.float32)), Gd).numpy()
    if W3:    # one axis: the population loss and the sample loss of each sampler, each scaled to max 1
        sc = L[(x_grid > xlo) & (x_grid < xhi)].max()
        ax.plot(x_grid, L / sc, color="0.25", lw=2, label=r"$\mathcal{L}(x)$", zorder=2)
        for name, c in [("MLGD", c_mlgd), ("MLGD-F", c_mlgdf)]:
            if name in sim.get("empirical_loss", {}):
                e = sim["empirical_loss"][name]
                mu_e = np.array(e["values"]).mean(0)
                ax.plot(np.array(e["x"]), mu_e / mu_e.max(), color=c, lw=1.2, alpha=0.9, zorder=2,
                        label=rf"$\hat{{\mathcal{{L}}}}(x)$, {name}")
        ax.legend(fontsize=fs_leg - 1, loc="upper right", frameon=False, handlelength=1.2, borderaxespad=0.2,
                  labelspacing=0.2)
    else:
        ax.plot(x_grid, L, color="0.25", lw=2, label=r"$\mathcal{L}(x)$" if (compact or W2) else r"population $\mathcal{L}(x)$", zorder=2)
    if "empirical_loss" in sim and not W3:   # the sample loss the optimiser sees (MMD), mean +- sd over seeds, own scale
        ax2 = ax.twinx()
        for name, c in [("MLGD", c_mlgd), ("MLGD-F", c_mlgdf)]:
            if name not in sim["empirical_loss"]:
                continue
            e = sim["empirical_loss"][name]
            xe, v = np.array(e["x"]), np.array(e["values"])
            mu_e, sd_e = v.mean(0), v.std(0)
            ax2.plot(xe, mu_e, color=c, lw=1.2, alpha=0.9, zorder=2)   # mean over seeds
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel(r"$\hat{\mathcal{L}}(x)$" if (compact or W2) else r"sample loss $\hat{\mathcal{L}}(x)$ (MMD)", fontsize=fs_leg)
        ax2.tick_params(labelsize=fs_leg)
        n_c = sim["empirical_loss"]["MLGD"]["n_cond"]
        ax.plot([], [], color="0.5", lw=1.2, label=r"sample $\hat{\mathcal{L}}(x)$ (right axis)" if (compact or W2) else
                rf"$\hat{{\mathcal{{L}}}}(x)$ (MMD, $n_{{\rm cond}}={n_c}$; right axis)")
    if not W3:   # in wide3 the x-hat and x* markers are drawn once, on the edge between (a) and (b)
        for m in methods:
            vline(ax, m, lab(m, res[m]))
        ax.axvline(x_star, color=c_true, ls="--", lw=2, zorder=5)
    ax.set_xlim(xlo, xhi); ax.set_xlabel("x")
    ax.set_ylim(0, 1.08 if W3 else 1.15 * L[(x_grid > xlo) & (x_grid < xhi)].max())
    ax.set_ylabel("loss (max 1)" if W3 else r"$\mathcal{L}(x)$" if (compact or W2) else
                  r"$\mathcal{L}(x) = \mathcal{D}\left(\mathcal{P}(Y\mid X=x),\,\mathcal{G}\right)$")
    if layout == "lr":
        ax.legend(fontsize=fs_leg - 1, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    elif not W2:
        ax.set_title(r"loss landscape: minimised only at $x^\star$", fontsize=fs_title)
    ax.grid(alpha=0.2)

    # ---- middle, bottom (wide3): the guided trajectories --------------------------
    if W3:
        if traj:
            draw_trajectories(ax_traj, traj, x_star, xlo, xhi, fs_leg, STYLE)
            ax_traj.set_ylim(-5.0, xhi)        # no trajectory goes below -3
        ax_traj.text(0.012, 0.965, "(d)", transform=ax_traj.transAxes, fontsize=fs_title, fontweight="bold",
                     ha="left", va="top", zorder=10)
        ax_traj.grid(alpha=0.2)

    # ---- right: the CDMS beta-sweep, 2 x 2 -------------------------------------
    ymax = 1.1 * max(max(Q[b].max(), px.max()) for b in betas)
    if tv_panel and smc:   # one beta: the sample curves may peak above Q_beta
        ymax = max(ymax, 1.08 * max(_kde(smc[k][betas[0]], x_grid, bw=0.12).max() for k in ("plain", "corr")))
    smc_tv = (smc or {}).get("tv", {})
    for i, beta in enumerate(betas):
        ax = sweep_axes[i]
        plain = smc["plain"][beta] if smc else sweep[beta]
        if W3:    # both sample sets as smoothed empirical densities, drawn alike
            dg = _kde(plain, x_grid, bw=0.12)      # fixed small bandwidth: the default oversmooths the separate modes
            ax.plot(x_grid, dg, color=c_data, lw=2.0, ls="--" if tv_panel else "-", label="guided samples", zorder=6)
            if smc:
                ax.plot(x_grid, _kde(smc["corr"][beta], x_grid, bw=0.12), color="black", lw=1.8,
                        ls="--" if tv_panel else "-", label="twisted-SMC corrected", zorder=7)
        else:
            ax.hist(plain, bins=80, range=(xlo, xhi), density=True, color=c_data, alpha=0.5,
                    edgecolor="white", linewidth=0.3, label="guided samples")
        if smc and not W3:   # twisted-SMC corrected population of fig_smc_correction.py, same setting and betas
            ax.hist(smc["corr"][beta], bins=80, range=(xlo, xhi), density=True, histtype="step",
                    color="black", linewidth=1.3, label="twisted-SMC corrected", zorder=4)
        if smc and str(float(beta)) in smc_tv:
                tp, tc = smc_tv[str(float(beta))]
                ax.text(0.97 if tv_panel else 0.03 if (compact or W2) else 0.98,
                        0.62 if tv_panel else (0.74 if W2 else 0.62) if (compact or W2) else 0.9,
                        rf"TV {tp:.2f}$\to${tc:.2f}", transform=ax.transAxes,
                        ha="right" if tv_panel else "left" if (compact or W2) else "right", va="top", fontsize=fs_leg - (1 if W2 else 0))
        ax.plot(x_grid, Q[beta], color=c_q, lw=2.0, label=r"$\mathcal{Q}_\beta(x)$", zorder=3)
        ax.fill_between(x_grid, Q[beta], color=c_q, alpha=0.12, zorder=2)
        if W3:   # x* as the black star on the x-axis, as in (a)
            ax.plot([x_star], [0.0], ls="none", marker="*", ms=10, color=c_true, mec="white", mew=0.5,
                    transform=ax.get_xaxis_transform(), clip_on=False, zorder=9)
        else:
            ax.axvline(x_star, color=c_true, ls="--", lw=1.8, zorder=4)
        if i == 0 and not W2:
            ax.annotate(r"$x^*$", (x_star, ymax), xytext=(-4, -4), textcoords="offset points", ha="right", va="top",
                        fontsize=fs_title - 1)
        ax.plot(x_grid, px, color=c_prior, ls="-" if tv_panel else "-.", lw=1.8 if tv_panel else 2.2,
                label=r"prior $\mathcal{P}(x)$", zorder=3 if tv_panel else 5)   # dashed sample curves on top
        ax.set_xlim(xlo, xhi); ax.set_ylim(0, ymax); ax.grid(alpha=0.15)
        if W2:
            if not tv_panel:
                ax.text(0.03, 0.93, rf"$\beta = {beta:g}$", transform=ax.transAxes, fontsize=fs_leg + 1, va="top")
            if W3:
                ax.set_xticks([-5, 0, 5])
            if i == 0:
                ax.set_title(r"$\bf{" + sweep_letter + r"}$ CDMS samples" + (rf", $\beta = {beta:g}$" if tv_panel else ""),
                             fontsize=fs_title, loc="left")
            if not sweep_xlabel(i):
                ax.tick_params(labelbottom=False)
            if not sweep_ylabel(i) and not W3:
                ax.tick_params(labelleft=False)
        elif layout == "lr":
            ax.text(0.02, 0.9, rf"$\beta = {beta:g}$", transform=ax.transAxes, fontsize=fs_title, va="top")
            if i == 0:
                ax.set_title(r"$\bf{" + sweep_letter + r"}$ CDMS samples, increasing $\beta$" if compact else
                             r"CDMS samples for increasing guidance strength $\beta$", fontsize=fs_title)
            if not sweep_xlabel(i):
                ax.tick_params(labelbottom=False)
        else:
            ax.set_title(rf"CDMS samples, $\beta = {beta:g}$", fontsize=fs_title)
        if sweep_ylabel(i) and (not compact or i == len(betas) // 2) and (not W2 or W3):
            ax.set_ylabel("density")
        if sweep_xlabel(i):
            ax.set_xlabel("x")
        if not W2 and i == (0 if (smc or layout != "lr") else len(betas) - 1):   # legend where the panel has room
            h, l = ax.get_legend_handles_labels()
            keys = [r"prior $\mathcal{P}(x)$", r"$\mathcal{Q}_\beta(x)$", "guided samples"]
            if smc:
                keys.append("twisted-SMC corrected")
            order = [l.index(k) for k in keys]
            if compact:
                short = {r"prior $\mathcal{P}(x)$": "prior", "guided samples": "guided",
                         "twisted-SMC corrected": "SMC-corrected"}
                sweep_axes[-1].legend([h[j] for j in order], [short.get(l[j], l[j]) for j in order],
                                      fontsize=fs_leg - 1, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2,
                                      frameon=False)
            else:
                ax.legend([h[j] for j in order], [l[j] for j in order], fontsize=fs_leg - 1,
                          loc="upper right" if not smc else "upper left",
                          bbox_to_anchor=None if not smc else (0.0, 0.78), frameon=False)

    if tv_panel and smc_tv:   # (f): TV to Q_beta of the guided and the corrected samples, for every beta of the run
        bs = sorted(float(k) for k in smc_tv)
        pos = np.arange(len(bs))                        # betas 0, 1, 2, 4, 8, 16 at equal spacing
        ax_tv.plot(pos, [smc_tv[str(b)][0] for b in bs], color=c_data, ls="--", lw=2.0, marker="o", ms=4,
                   label="guided samples")
        ax_tv.plot(pos, [smc_tv[str(b)][1] for b in bs], color="black", ls="--", lw=1.8, marker="o", ms=4,
                   label="twisted-SMC corrected")
        ax_tv.set_xticks(pos); ax_tv.set_xticklabels([f"{b:g}" for b in bs])
        ax_tv.set_xlabel(r"$\beta$", labelpad=0); ax_tv.set_ylabel(r"TV to $\mathcal{Q}_\beta$")
        ax_tv.set_ylim(0, 1.12 * max(v[0] for v in smc_tv.values())); ax_tv.grid(alpha=0.2)
        ax_tv.text(0.012, 0.965, "(f)", transform=ax_tv.transAxes, fontsize=fs_title, fontweight="bold",
                   ha="left", va="top", zorder=10)
        ax_tv.text(0.45, 0.5, "guided", transform=ax_tv.transAxes, color=c_data, ha="center", fontsize=fs_leg)
        ax_tv.text(0.45, 0.1, "corrected", transform=ax_tv.transAxes, color="black", ha="center", fontsize=fs_leg)
    if W2:   # thinner lines and smaller dots for the print size, then one legend for all panels
        for a_ in fig.axes:
            for ln in a_.lines:
                ln.set_linewidth(ln.get_linewidth() * 0.5)
            for coll in a_.collections:
                if hasattr(coll, "get_sizes") and len(coll.get_sizes()):
                    coll.set_sizes(coll.get_sizes() * 0.15)
                coll.set_linewidth(np.asarray(coll.get_linewidth()) * 0.5)
            for p_ in a_.patches:
                p_.set_linewidth(p_.get_linewidth() * 0.6)
        rename = {lab("MLGD", None): "MLGD", lab("MLGD-F", None): "MLGD-F",
                  r"target $\mathcal{G}(Y)$": r"target $\mathcal{G}$, $x^*$"}
        if W3:    # the returned x-hat (a, b), its conditional (c) and its trajectories (d) share the colour
            rename = {r"target $\mathcal{G}(Y)$": r"$x^*$ and its target $\mathcal{G}$"}
        hs, ls_ = [], []
        for a_ in [ax_joint, ax_loss, ax_cond, sweep_axes[0]] + [a_ for a_ in fig.axes if a_ not in (ax_joint, ax_loss, ax_cond)]:
            for h_, l_ in zip(*a_.get_legend_handles_labels()):
                l_ = rename.get(l_, l_)
                if l_ in ls_ or l_.startswith("_") or l_.startswith(r"$\mathcal{P}(Y\mid X=") or ": " in l_:
                    continue
                hs.append(h_); ls_.append(l_)
        if W3:
            sweep_keys = {"guided samples", "twisted-SMC corrected", r"$\mathcal{Q}_\beta(x)$", r"prior $\mathcal{P}(x)$"}
            left = [(h_, l_) for h_, l_ in zip(hs, ls_) if l_ not in sweep_keys and not l_.startswith(r"$\hat")
                    and l_ != r"$\mathcal{L}(x)$"]
            right_ = sorted([(h_, l_) for h_, l_ in zip(hs, ls_) if l_ in sweep_keys],
                            key=lambda t: ["guided samples", "twisted-SMC corrected", r"$\mathcal{Q}_\beta(x)$",
                                           r"prior $\mathcal{P}(x)$"].index(t[1]))
            from matplotlib.lines import Line2D
            left = [(Line2D([], [], color=STYLE[m]["color"], lw=1.4, marker="^", ms=6, mec="white", mew=0.4),
                     rf"{m}, returned $\hat{{x}}$") for m in methods]
            left.append((Line2D([], [], color=c_true, ls="--", lw=1.0, marker="*", ms=8, mec="white", mew=0.4),
                         r"$x^*$ and its target $\mathcal{G}$"))
            left.insert(0, (Line2D([], [], ls="none"), r"$\bf{(a)}$–$\bf{(d)}$:"))   # the legend serves all four panels
            pa, pc = ax_joint.get_position(), ax_traj.get_position()
            pe = sweep_axes[-1].get_position()
            # (a)-(d): in the empty lower part of (d); (e): in the empty upper part of its first panel
            left = left[1:]
            ax_traj.legend(*zip(*left), loc="lower right", ncol=1, fontsize=fs_leg - 1, frameon=False,
                           handlelength=1.8,
                           labelspacing=0.25, borderaxespad=0.2)
            short = {"guided samples": "guided", "twisted-SMC corrected": "SMC-corrected",
                     r"prior $\mathcal{P}(x)$": r"prior $\mathcal{P}$"}
            sweep_axes[0].legend([h_ for h_, _ in right_], [short.get(l_, l_) for _, l_ in right_],
                                 loc="upper left" if tv_panel else "upper right",
                                 ncol=1, fontsize=fs_leg - 1, frameon=True, framealpha=1.0, facecolor="white",
                                 edgecolor="none", handlelength=1.3, columnspacing=0.8, labelspacing=0.25,
                                 borderaxespad=0.15, borderpad=0.2).set_zorder(20)   # opaque: x* runs underneath
        else:
            fig.legend(hs, ls_, loc="lower center", ncol=(len(ls_) + 1) // 2, fontsize=fs_leg - 1, frameon=False,
                       bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=1.6)
    fig.savefig(out_png, dpi=300 if W2 else 200, bbox_inches=None if W2 else "tight")
    print(f"[figure] wrote {out_png}")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    st = make_setting(args.setting)
    print(f"[setup] setting={st.name} device={args.device} seed={args.seed} quick={args.quick}")

    if args.rerun_sweep:   # takes precedence over --rerun-sim: the methods are kept from the cache
        sim, sweep = rerun_sweep(st, args)
    else:
        cached = None if args.rerun_sim else load_sim(args)
        sim, sweep = cached if cached is not None else simulate(st, args)
    smc_betas = args.betas_plot if (args.layout == "wide3" and args.betas_plot) else sorted(sweep)
    smc = load_smc(args.smc_npz, smc_betas) if args.smc_npz else None
    traj = None
    if args.traj:   # {"MLGD": {"x0": [restarts][steps], "selected": r}, "MLGD-F": ...}
        with open(args.traj) as f:
            traj = json.load(f)
    make_figure(st, sim, sweep, args.out, layout=args.layout, smc=smc, panels=args.panels,
                betas_plot=args.betas_plot, figsize=args.figsize, font_scale=args.font_scale,
                compact=args.compact, traj=traj)


if __name__ == "__main__":
    main()
