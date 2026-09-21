"""
Check the trained networks against the analytic mixture, independently of the x* search.

Three measurements, each answering one question:

  1. score     Is the prior's score network right, and at which noise levels? The noised
               mixture at level t is again a mixture (means sqrt(abar_t) mu_k, covariances
               abar_t Sigma_k + (1 - abar_t) I), so its score is exact. Reported: the relative
               Fisher divergence E||s_theta - s||^2 / E||s||^2 per t, with x_t drawn from the
               true noised prior.
  2. prior     Do unguided samples from the prior look like P(X)? Sliced Wasserstein between
               model samples and exact draws (utils.evaluation.evaluate_unconditional_model),
               plus the fraction of samples per mixture component against the mixture weights.
  3. sampler   Does the conditional sampler reproduce P(Y | X = x)? Sliced Wasserstein averaged
               over random x ~ P(X) (utils.evaluation.evaluate_conditional_model), and the same
               comparison at chosen points: x*, the other cluster centres, random draws, and,
               when dim(X) = 1, a grid over x. dim(Y) = 1 in every synthetic setting, so the
               per-point error is the exact 1D Wasserstein-1 between sorted samples.

Inputs, either the pipeline checkpoints
    python scripts/check_models.py --prior runs/mog4d/models/model_unconditional.pt \
        --sampler runs/mog4d/models/model_conditional.pt --data runs/mog4d/data.pt
or a Figure-1 setting whose networks live under checkpoints/toy/<setting>/
    python scripts/check_models.py --setting hard

Writes <out>.json with every number and <out>.png with three panels (score error per t,
unguided samples against P(x), conditional error over x with x* marked).
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

from utils.dist import (generate_mog_samples_not_differentiable, compute_conditionals, compute_alpha,   # noqa: E402
                        filter_and_normalize)
from utils.evaluation import evaluate_unconditional_model, evaluate_conditional_model                 # noqa: E402
from losses.functions import sliced_wasserstein_distance                                             # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("inputs (pipeline checkpoints)")
    g.add_argument("--prior", help="unconditional diffusion checkpoint from scripts/train_models.py")
    g.add_argument("--sampler", nargs="*", default=[], help="conditional checkpoints (diffusion and/or cm)")
    g.add_argument("--data", help="data file from scripts/generate_data.py (mixture, x*, condition_on)")
    g = p.add_argument_group("inputs (Figure-1 setting)")
    g.add_argument("--setting", choices=["hard", "bimodal", "paper", "mog2d"])
    g.add_argument("--ckpt-dir", default="checkpoints/toy")
    g.add_argument("--seed", type=int, default=42, help="checkpoint seed of the Figure-1 setting")
    g = p.add_argument_group("budget")
    g.add_argument("--n-samples", type=int, default=4000, help="samples per comparison")
    g.add_argument("--n-random-x", type=int, default=100, help="random x for the averaged conditional error")
    g.add_argument("--n-seeds", type=int, default=3, help="repeats at the chosen points")
    g.add_argument("--n-t", type=int, default=12, help="noise levels for the score check")
    g.add_argument("--n-projections", type=int, default=200)
    g.add_argument("--grid", type=int, default=41, help="x-grid size for the conditional curve (dim(X) = 1 only)")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--out", default=None, help="stem for .json/.png (default results/check_<name>)")
    return p.parse_args()


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------
def load_inputs(args):
    """Returns (mixture dict, models dict name -> (model, kind), name). kind is 'diffusion' or 'cm'."""
    dev = args.device
    if args.setting:
        import fig1_toy_example as f1
        st = f1.make_setting(args.setting)
        a = types.SimpleNamespace(device=dev, ckpt_dir=args.ckpt_dir, seed=args.seed, nunits=128, nblocks=3,
                                  epochs=0, batch_size=1024, uncond="load", cond="load", cm="load", retrain=False)
        uncond, cond, cm, _ = f1.get_models(st, a)
        mix = {"mu_list": st.mu_list, "Sigma_list": st.Sigma_list, "alpha": st.alpha, "x_star": st.x_star,
               "condition_on": f1.CONDITION_ON}
        return mix, {"prior": (uncond, "diffusion"), "cond_diffusion": (cond, "diffusion"), "cond_cm": (cm, "cm")}, args.setting
    assert args.data, "--data is required with pipeline checkpoints"
    from run_optimization import load_model
    d = torch.load(args.data, map_location="cpu", weights_only=False)
    mix = {"mu_list": [m.float() for m in d["mu_list"]], "Sigma_list": [S.float() for S in d["Sigma_list"]],
           "alpha": d["alpha"].float(), "x_star": d.get("x_star"), "condition_on": int(d["condition_on"])}
    if mix["x_star"] is None:
        mix["x_star"] = recover_x_star(mix)
    models = {}
    if args.prior:
        models["prior"] = load_model(args.prior, dev)
    for i, path in enumerate(args.sampler):
        m, arch = load_model(path, dev)
        models[f"cond_{arch}" if f"cond_{arch}" not in models else f"cond_{arch}{i}"] = (m, arch)
    name = os.path.splitext(os.path.basename(args.data))[0]
    return mix, models, name


def recover_x_star(mix):
    """The synthetic construction gives two components the same x-part of their mean, and that
    shared vector is x*. Parameter files omit it; recover it from the fingerprint."""
    c, mus = mix["condition_on"], mix["mu_list"]
    for i in range(len(mus)):
        for j in range(i + 1, len(mus)):
            if torch.allclose(mus[i][:c], mus[j][:c]):
                print(f"[inputs] x* recovered from components {i} and {j} sharing the x-part")
                return mus[i][:c].clone()
    print("[inputs] x* unknown (no two components share an x-part)")
    return None


def x_marginal(mix):
    c = mix["condition_on"]
    return [m[:c] for m in mix["mu_list"]], [S[:c, :c] for S in mix["Sigma_list"]], mix["alpha"]


# ----------------------------------------------------------------------------
# 1. score network against the exact noised-mixture score
# ----------------------------------------------------------------------------
def mixture_score(x, means, covs, weights):
    """grad_x log sum_k w_k N(x; m_k, C_k), via responsibilities."""
    logp, grads = [], []
    for m, C, w in zip(means, covs, weights):
        dist = torch.distributions.MultivariateNormal(m, covariance_matrix=C)
        logp.append(torch.log(w) + dist.log_prob(x))
        grads.append(-torch.linalg.solve(C, (x - m).T).T)
    r = torch.softmax(torch.stack(logp, -1), -1)
    return (r[..., None] * torch.stack(grads, -2)).sum(-2)


@torch.no_grad()
def check_score(model, mix, args):
    mus, Sigs, alpha = x_marginal(mix)
    d = mus[0].numel()
    dev = args.device
    T = int(model.diffusion_steps)
    ts = sorted({int(round(v)) for v in np.linspace(1, T - 1, args.n_t)})
    out = []
    for t in ts:
        ab = float(model.baralphas[t])
        means = [math.sqrt(ab) * m for m in mus]
        covs = [ab * S + (1 - ab) * torch.eye(d) for S in Sigs]
        x_t = generate_mog_samples_not_differentiable(args.n_samples, means, covs, alpha).float()
        true = mixture_score(x_t, means, covs, alpha)
        eps = model(x_t.to(dev), torch.full([x_t.shape[0], 1], t, device=dev), None).cpu()
        pred = -eps / math.sqrt(1 - ab)
        num, den = ((pred - true) ** 2).sum(1).mean(), (true ** 2).sum(1).mean()
        out.append({"t": t, "abar": ab, "rel_fisher": float(num / den), "fisher": float(num),
                    "score_norm2": float(den)})
    print("[score] relative Fisher divergence per t: " + ", ".join(f"t={o['t']}:{o['rel_fisher']:.3f}" for o in out))
    return out


# ----------------------------------------------------------------------------
# 2. unguided samples against P(X)
# ----------------------------------------------------------------------------
@torch.no_grad()
def check_prior(model, mix, args):
    mus, Sigs, alpha = x_marginal(mix)
    c = mix["condition_on"]
    swd_raw, swd_norm, _, _ = evaluate_unconditional_model(
        model, mix["mu_list"], mix["Sigma_list"], mix["alpha"], c, args.device,
        nsamples=args.n_samples, num_projections=args.n_projections, model_type="diffusion", num_evaluations=args.n_seeds)
    xs = model.sample(nsamples=args.n_samples, condition_x=None, device=args.device)[0].detach().cpu().float()
    # responsibilities under the exact x-marginal: which component each sample belongs to
    logp = torch.stack([torch.log(w) + torch.distributions.MultivariateNormal(m, covariance_matrix=S).log_prob(xs)
                        for m, S, w in zip(mus, Sigs, alpha)], -1)
    frac = torch.softmax(logp, -1).mean(0)
    out = {"swd_raw": float(swd_raw), "swd_norm": float(swd_norm), "n_samples": int(args.n_samples),
           "component_mass_model": frac.tolist(), "component_mass_true": alpha.tolist(),
           "samples_x": xs.numpy() if c == 1 else None}
    print(f"[prior] SWD raw {swd_raw:.4f} norm {swd_norm:.4f} | component mass model {np.round(frac.numpy(), 3)} "
          f"vs true {np.round(alpha.numpy(), 3)}")
    return out


# ----------------------------------------------------------------------------
# 3. conditional sampler against P(Y | X = x)
# ----------------------------------------------------------------------------
@torch.no_grad()
def draw_cond(model, kind, x, n, dev):
    cond = x.view(1, -1).repeat(n, 1).float().to(dev)
    y = model.sample(nsamples=n, condition_x=cond, device=dev)[0].detach().cpu().float()
    if kind == "diffusion":
        y = y[:, model.condition_on:]
    return y


def exact_cond(mix, x, n):
    m, S = compute_conditionals(mix["mu_list"], mix["Sigma_list"], x.view(-1, 1))
    w = compute_alpha(mix["mu_list"], mix["Sigma_list"], mix["alpha"], x.view(-1))
    m, S, w = filter_and_normalize(m, S, w, threshold=1e-6)
    return generate_mog_samples_not_differentiable(n, m, S, w).float(), (m, S, w)


def w1_1d(a, b):
    a, b = torch.sort(a.view(-1))[0], torch.sort(b.view(-1))[0]
    return float((a - b).abs().mean())


@torch.no_grad()
def cond_error_at(model, kind, mix, x, args):
    """Sampler against the exact conditional at one x: W1 (dim(Y) = 1) or normalised SWD, over seeds."""
    vals = []
    for s in range(args.n_seeds):
        torch.manual_seed(1000 * args.seed + s)
        y = draw_cond(model, kind, x, args.n_samples, args.device)
        y_true, (m, S, w) = exact_cond(mix, x, args.n_samples)
        if y.shape[1] == 1:
            vals.append(w1_1d(y, y_true))
        else:
            vals.append(sliced_wasserstein_distance(y, y_true, num_projections=args.n_projections, p=2,
                                                    mu_list_sliced=m, Sigma_list_sliced=S, alpha=w)[1])
    return float(np.mean(vals)), float(np.std(vals))


@torch.no_grad()
def check_sampler(model, kind, mix, args):
    c = mix["condition_on"]
    avg, std, _ = evaluate_conditional_model(
        model, mix["mu_list"], mix["Sigma_list"], mix["alpha"], c, args.device,
        num_evaluations=args.n_random_x, nsamples=args.n_samples, num_projections=args.n_projections,
        extract_generated_only=(kind == "diffusion"), model_type=kind)
    out = {"swd_norm_random_x_mean": float(avg), "swd_norm_random_x_std": float(std), "n_random_x": args.n_random_x,
           "metric_at_points": "W1" if mix["mu_list"][0].numel() - c == 1 else "SWD_norm", "points": {}}
    pts = {}
    if mix["x_star"] is not None:
        pts["x*"] = mix["x_star"].view(-1).float()
    centres = []
    for m in mix["mu_list"]:
        xc = m[:c].float()
        if not any(torch.allclose(xc, o) for o in centres):
            centres.append(xc)
    for xc in centres:
        if mix["x_star"] is None or not torch.allclose(xc, pts["x*"]):
            label = np.round(xc.numpy(), 2).tolist() if c > 1 else round(float(xc), 2)
            pts[f"centre {label}"] = xc
    torch.manual_seed(args.seed)
    xr = generate_mog_samples_not_differentiable(3, mix["mu_list"], mix["Sigma_list"], mix["alpha"])[:, :c].float()
    for k in range(3):
        pts[f"random x {k}"] = xr[k]
    for name, x in pts.items():
        mean, sd = cond_error_at(model, kind, mix, x, args)
        out["points"][name] = {"x": x.tolist(), "mean": mean, "std": sd}
    print(f"[sampler {kind}] SWD over {args.n_random_x} random x: {avg:.4f} +- {std:.4f} | at points ({out['metric_at_points']}): "
          + ", ".join(f"{n} {v['mean']:.4f}" for n, v in out["points"].items()))
    if c == 1:
        lo = min(float(m[0]) for m in mix["mu_list"]) - 1.5
        hi = max(float(m[0]) for m in mix["mu_list"]) + 1.5
        grid = np.linspace(lo, hi, args.grid)
        curve = [cond_error_at(model, kind, mix, torch.tensor([float(x)]), args) for x in grid]
        out["grid"] = {"x": grid.tolist(), "mean": [v[0] for v in curve], "std": [v[1] for v in curve]}
    return out


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------
def make_figure(res, mix, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    c = mix["condition_on"]
    ax = axes[0]
    if "score" in res:
        s = res["score"]
        ax.plot([o["t"] for o in s], [o["rel_fisher"] for o in s], "o-", color="tab:red")
        ax.set_yscale("log"); ax.set_xlabel("noise level t"); ax.set_ylabel(r"$E\|s_\theta - \nabla\log p_t\|^2 / E\|\nabla\log p_t\|^2$")
        ax.set_title("score network vs. exact noised-mixture score"); ax.grid(alpha=0.3)
    ax = axes[1]
    if "prior" in res:
        p = res["prior"]
        if c == 1 and p.get("samples_x") is not None:
            mus, Sigs, alpha = x_marginal(mix)
            xs = np.asarray(p["samples_x"]).ravel()
            grid = np.linspace(xs.min() - 1, xs.max() + 1, 400)
            dens = sum(float(w) * np.exp(-0.5 * (grid - float(m)) ** 2 / float(S)) / np.sqrt(2 * np.pi * float(S))
                       for m, S, w in zip(mus, Sigs, alpha))
            ax.hist(xs, bins=80, density=True, color="steelblue", alpha=0.5, label="unguided samples")
            ax.plot(grid, dens, color="tab:orange", ls="-.", lw=2, label=r"$\mathcal{P}(x)$")
            ax.set_xlabel("x"); ax.set_ylabel("density")
        else:
            k = np.arange(len(p["component_mass_true"]))
            ax.bar(k - 0.2, p["component_mass_true"], 0.4, label="true weight", color="tab:orange")
            ax.bar(k + 0.2, p["component_mass_model"], 0.4, label="model mass", color="steelblue")
            ax.set_xlabel("component"); ax.set_ylabel("mass")
        ax.set_title(f"prior samples vs. P(X): SWD norm {p['swd_norm']:.3f}"); ax.legend(frameon=False); ax.grid(alpha=0.3)
    ax = axes[2]
    colors = {"cond_diffusion": "tab:red", "cond_cm": "tab:green"}
    for name, r in res.items():
        if not name.startswith("cond_"):
            continue
        col = colors.get(name, None)
        if "grid" in r:
            g = r["grid"]
            ax.plot(g["x"], g["mean"], color=col, lw=1.8, label=f"{name}: over x")
            ax.fill_between(g["x"], np.array(g["mean"]) - np.array(g["std"]), np.array(g["mean"]) + np.array(g["std"]),
                            color=col, alpha=0.15, lw=0)
        else:
            names = list(r["points"]); vals = [r["points"][n]["mean"] for n in names]
            ax.bar(np.arange(len(names)) + (0.2 if name == "cond_cm" else -0.2), vals, 0.4, color=col, label=name)
            ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    if mix["x_star"] is not None and c == 1:
        ax.axvline(float(mix["x_star"].view(-1)[0]), color="black", ls="--", lw=1.5)
        ax.annotate(r"$x^*$", (float(mix["x_star"].view(-1)[0]), ax.get_ylim()[1]), xytext=(-4, -4),
                    textcoords="offset points", ha="right", va="top")
    metric = next((r["metric_at_points"] for n, r in res.items() if n.startswith("cond_")), "")
    ax.set_ylabel(f"{metric} to exact " + r"$\mathcal{P}(Y\mid X=x)$"); ax.set_xlabel("x" if c == 1 else "")
    ax.set_title("conditional sampler vs. exact conditional"); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=170)
    print(f"[figure] wrote {out_png}")


def main():
    args = parse_args()
    mix, models, name = load_inputs(args)
    out = args.out or os.path.join("results", f"check_{name}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t0 = time.time()
    res = {"name": name, "args": vars(args), "condition_on": mix["condition_on"],
           "x_star": None if mix["x_star"] is None else mix["x_star"].view(-1).tolist()}
    if "prior" in models:
        m, _ = models["prior"]
        res["score"] = check_score(m, mix, args)
        res["prior"] = check_prior(m, mix, args)
    for k, (m, kind) in models.items():
        if k.startswith("cond_"):
            res[k] = check_sampler(m, kind, mix, args)
    res["time_s"] = time.time() - t0
    make_figure(res, mix, out + ".png")
    if "prior" in res:
        res["prior"]["samples_x"] = None   # not for the JSON
    with open(out + ".json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[check] wrote {out}.json ({res['time_s']:.0f}s)")


if __name__ == "__main__":
    main()
