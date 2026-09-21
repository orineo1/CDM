"""
Find x*: the input whose induced conditional P(Y | X = x) matches a target G(Y).

This is the paper's method as one command. Give it a prior over X (a trained
unconditional diffusion checkpoint), a conditional sampler for Y | X (a trained
consistency-model or conditional-diffusion checkpoint) and a target G(Y); it
runs guided reverse diffusion over X with the distributional loss as guidance,
several restarts, and returns the best x_hat.

    python scripts/find_xstar.py --prior runs/mog4d/models/model_unconditional.pt \
        --sampler runs/mog4d/models/model_conditional.pt --target runs/mog4d/data.pt \
        --n-attempts 10 --out runs/mog4d/xstar.json

Inputs
  --prior         checkpoint from scripts/train_models.py --model unconditional
  --sampler       checkpoint from scripts/train_models.py --model conditional
  --sampler-type  cm (few-step consistency model: MLGD-F) | diffusion (full
                  conditional diffusion: MLGD) | auto (read from the checkpoint)
  --target        a .pt from scripts/generate_data.py (target as a Gaussian
                  mixture, plus x* and the joint for evaluation), or a .pt/.npy
                  holding a [m, dim_y] array of target samples

Which algorithm (presets; any individual knob below overrides the preset)
  --guidance lgd  Loss-Guided Diffusion (default, the paper): n_mc Monte-Carlo
                  perturbations of the Tweedie point, log-sum-exp aggregate,
                  constant step zeta.
  --guidance dps  Diffusion Posterior Sampling: the single Tweedie point,
                  step zeta_t = zeta' * beta_t.

Inner estimator (Algorithm 2)
  --n-cond N          conditional samples per step for the loss         (250)
  --n-target M        target samples per step                       (= n_cond)
  --loss mmd|swd      MMD with a multi-bandwidth RBF kernel, or sliced Wasserstein
  --rbf-kernels K --rbf-mul F   RBF mixture: K bandwidths, geometric factor F
  --swd-projections P
  --select all|uniform --n-sel k   backpropagate through all n_cond samples,
                      or through a uniform random subset of k (rest detached)

Outer loop (Algorithm 1)
  --n-mc M            Monte-Carlo perturbations of the Tweedie point (lgd: 3, dps: 1)
  --perturb           beta_ratio | beta_prod_ratio_sq | sqrt_beta_prod | none
  --aggregate         logsumexp | mean
  --step-size S       a constant zeta, or 'beta_t' for zeta_t = zeta' * beta_t
  --zeta Z            zeta' used with --step-size beta_t                (1.0)
  --eta E             DDIM stochasticity, 0 = deterministic              (0.0)
  --guidance-cutoff-t T   no guidance for t < T (MNIST used 200)
  --init randn|zeros  x_T initialisation (paper: randn)

Run control
  --n-attempts R      independent restarts; the smallest final loss wins (10)
  --seed, --device, --final-n-cond, --out, --no-store, --progress

Output: x_hat of every attempt and the best one, printed and written to --out
(JSON). When the target file carries x* and the joint mixture, |x_hat - x*|
and the exact L2 distance between P(Y | X = x_hat) and G are reported too.
Each attempt is appended to results/runs.jsonl unless --no-store.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from guidance import GuidanceConfig, optimize_guided                                   # noqa: E402
from guidance.config import (init_randn, init_zeros, perturb_beta_ratio,               # noqa: E402
                             perturb_beta_prod_ratio_sq, perturb_sqrt_beta_prod, perturb_none,
                             aggregate_logsumexp, aggregate_mean, aggregate_single)
from losses.functions import MMDLoss, RBF, sliced_wasserstein_distance                  # noqa: E402
from utils.dist import (generate_mog_samples_not_differentiable, compute_conditionals,  # noqa: E402
                        compute_alpha, gmm_l2_distance)
from gcio.results import save_run                                                       # noqa: E402
from run_optimization import load_model                                                 # noqa: E402

PERTURB = {"beta_ratio": perturb_beta_ratio, "beta_prod_ratio_sq": perturb_beta_prod_ratio_sq,
           "sqrt_beta_prod": perturb_sqrt_beta_prod, "none": perturb_none}
PRESETS = {  # --guidance -> defaults for the knobs a user did not set
    "lgd": dict(n_mc=3, perturb="beta_ratio", aggregate="logsumexp", step_size="1.0"),
    "dps": dict(n_mc=1, perturb="none", aggregate="logsumexp", step_size="beta_t"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("inputs")
    g.add_argument("--prior", required=True)
    g.add_argument("--sampler", required=True)
    g.add_argument("--sampler-type", choices=["auto", "cm", "diffusion"], default="auto")
    g.add_argument("--target", required=True)
    g = p.add_argument_group("algorithm")
    g.add_argument("--guidance", choices=list(PRESETS), default="lgd")
    g.add_argument("--n-mc", type=int, default=None)
    g.add_argument("--perturb", choices=list(PERTURB), default=None)
    g.add_argument("--aggregate", choices=["logsumexp", "mean"], default=None)
    g.add_argument("--step-size", default=None, help="constant zeta, or 'beta_t'")
    g.add_argument("--zeta", type=float, default=1.0)
    g.add_argument("--eta", type=float, default=0.0)
    g.add_argument("--guidance-cutoff-t", type=int, default=None)
    g.add_argument("--init", choices=["randn", "zeros"], default="randn")
    g = p.add_argument_group("inner estimator")
    g.add_argument("--n-cond", type=int, default=250)
    g.add_argument("--n-target", type=int, default=None)
    g.add_argument("--loss", choices=["mmd", "swd"], default="mmd")
    g.add_argument("--rbf-kernels", type=int, default=5)
    g.add_argument("--rbf-mul", type=float, default=2.0)
    g.add_argument("--swd-projections", type=int, default=100)
    g.add_argument("--select", choices=["all", "uniform"], default="all")
    g.add_argument("--n-sel", type=int, default=None, help="samples that keep their gradient under --select uniform")
    g = p.add_argument_group("run control")
    g.add_argument("--n-attempts", type=int, default=10)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--final-n-cond", type=int, default=None, help="samples for the reported final loss (4*n_cond)")
    g.add_argument("--out", default=None, help="JSON output (default: <target>_xstar.json)")
    g.add_argument("--no-store", action="store_true", help="do not append to results/runs.jsonl")
    g.add_argument("--progress", action="store_true")
    a = p.parse_args()
    for k, v in PRESETS[a.guidance].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.select == "uniform" and a.n_sel is None:
        a.n_sel = a.n_cond // 2
    return a


def load_target(path, device):
    """Returns (target_generator, info); info carries x* and the mixture when the file has them."""
    if path.endswith(".npy"):
        obj, samples = None, torch.as_tensor(np.load(path)).float()
    else:
        obj, samples = torch.load(path, map_location="cpu", weights_only=False), None
    if isinstance(obj, dict) and "target_means" in obj:
        m, v, w = obj["target_means"], obj["target_variances"], obj["target_weights"]
        gen = lambda n, dev: generate_mog_samples_not_differentiable(n, m, v, w).to(dev)   # noqa: E731
        return gen, {"kind": "mixture", "x_star": obj.get("x_star"), "mu_list": obj.get("mu_list"),
                     "Sigma_list": obj.get("Sigma_list"), "alpha": obj.get("alpha"), "target": (m, v, w)}
    if samples is None:
        samples = torch.as_tensor(obj).float()
    samples = samples.view(samples.shape[0], -1).to(device)

    def gen(n, dev):
        return samples[torch.randint(0, samples.shape[0], (n,), device=samples.device)].to(dev)
    return gen, {"kind": "samples", "x_star": None, "n_samples": int(samples.shape[0])}


def make_loss(args):
    if args.loss == "mmd":
        return MMDLoss(kernel=RBF(n_kernels=args.rbf_kernels, mul_factor=args.rbf_mul, device=args.device))
    return lambda X, Y: sliced_wasserstein_distance(X, Y, num_projections=args.swd_projections)


def make_subsampler(args):
    """--select uniform: keep the gradient on n_sel random samples, detach the rest
    (the loss value still uses all of them)."""
    if args.select == "all":
        return None

    def keep_subset(y):
        n = y.shape[0]
        keep = torch.zeros(n, dtype=torch.bool, device=y.device)
        keep[torch.randperm(n, device=y.device)[:args.n_sel]] = True
        return torch.where(keep[:, None], y, y.detach())
    return keep_subset


def make_cfg(args, prior, sampler_type, target_generator, seed):
    if args.step_size == "beta_t":
        step_fn = lambda t, i, z=args.zeta: z * float(prior.betas[t])     # noqa: E731
    else:
        step_fn = lambda t, i, z=float(args.step_size): z                # noqa: E731
    perturb = PERTURB[args.perturb] if args.n_mc > 1 or args.perturb != "none" else perturb_none
    aggregate = aggregate_single if args.n_mc == 1 and args.perturb == "none" else \
        {"logsumexp": aggregate_logsumexp, "mean": aggregate_mean}[args.aggregate]
    gen = target_generator if not args.n_target else (lambda n, d, g=target_generator: g(args.n_target, d))
    return GuidanceConfig(
        n_cond=args.n_cond, n_mc=args.n_mc, final_n_cond=args.final_n_cond or 4 * args.n_cond,
        init_fn=init_randn if args.init == "randn" else init_zeros, unsqueeze_init=True,
        perturb_fn=perturb, perturb_square=(args.perturb == "beta_prod_ratio_sq"),
        aggregate_fn=aggregate, step_size_fn=step_fn, guidance_cutoff_t=args.guidance_cutoff_t,
        timestep_source="range", ddim_step_style="condition_x", eta=args.eta,
        repeat_condition=True, condition_on_slice=(sampler_type == "diffusion"),
        cond_output_transform=make_subsampler(args),
        target_generator=gen, compute_final_loss=True,
        device=args.device, seed=seed, progress=args.progress,
        tag=f"find_xstar:{args.guidance}:{'MLGD-F' if sampler_type == 'cm' else 'MLGD'}",
    )


def evaluate(x_hat, info):
    out = {}
    if info.get("x_star") is not None:
        out["l2_to_xstar"] = float(torch.norm(x_hat.cpu() - info["x_star"].float().view(-1)))
    if info.get("mu_list") is not None and info.get("target") is not None:
        mu_p, S_p = compute_conditionals(info["mu_list"], info["Sigma_list"], x_hat.cpu())
        w_p = compute_alpha(info["mu_list"], info["Sigma_list"], info["alpha"], x_hat.cpu())
        out["l2_gmm"] = float(gmm_l2_distance(mu_p, S_p, w_p, *info["target"]))
    return out


def main():
    args = parse_args()
    dev = args.device
    prior, prior_arch = load_model(args.prior, dev)
    sampler, sampler_arch = load_model(args.sampler, dev)
    assert prior_arch == "diffusion", "--prior must be an unconditional diffusion checkpoint"
    sampler_type = sampler_arch if args.sampler_type == "auto" else args.sampler_type
    method = "MLGD-F" if sampler_type == "cm" else "MLGD"
    target_generator, info = load_target(args.target, dev)
    loss_fn = make_loss(args)
    print(f"[find_xstar] {method} with {args.guidance.upper()} guidance | loss={args.loss} n_cond={args.n_cond} "
          f"n_mc={args.n_mc} perturb={args.perturb} aggregate={args.aggregate} step={args.step_size} "
          f"select={args.select}{'' if args.select == 'all' else f'(n_sel={args.n_sel})'} init={args.init} eta={args.eta} "
          f"| attempts={args.n_attempts} target={info['kind']} device={dev}")

    attempts = []
    for r in range(args.n_attempts):
        seed = args.seed * 1000 + r
        cfg = make_cfg(args, prior, sampler_type, target_generator, seed)
        t0 = time.time()
        res = optimize_guided(prior, sampler, loss_fn, cfg)
        x_hat = res["x_final"].detach().float().view(-1)
        rec = {"attempt": r, "seed": seed, "x_hat": x_hat.cpu().tolist(), "final_loss": float(res["final_loss"]),
               "time_s": time.time() - t0, **evaluate(x_hat, info)}
        attempts.append(rec)
        extra = "  ".join(f"{k}={v:.4f}" for k, v in rec.items() if k in ("l2_gmm", "l2_to_xstar"))
        print(f"  [{r+1}/{args.n_attempts}] loss={rec['final_loss']:.5f}  x_hat={np.round(x_hat.cpu().numpy(), 4)}  {extra}  ({rec['time_s']:.1f}s)")
        if not args.no_store:
            save_run("find_xstar", cfg, {k: v for k, v in rec.items() if k not in ("x_hat", "attempt")}, seed=seed,
                     method=method, guidance=args.guidance, prior=args.prior, sampler=args.sampler, target=args.target)

    best = min(attempts, key=lambda a: a["final_loss"])
    print(f"[find_xstar] best: attempt {best['attempt']+1}, loss={best['final_loss']:.5f}, x_hat={np.round(best['x_hat'], 4)}")
    if info.get("x_star") is not None:
        print(f"[find_xstar] true x* = {np.round(info['x_star'].float().view(-1).numpy(), 4)}")
    out = args.out or os.path.splitext(args.target)[0] + "_xstar.json"
    with open(out, "w") as f:
        json.dump({"method": method, "guidance": args.guidance, "args": vars(args), "best": best, "attempts": attempts,
                   "x_star": None if info.get("x_star") is None else info["x_star"].float().view(-1).tolist()}, f, indent=2)
    print(f"[find_xstar] wrote {out}")


if __name__ == "__main__":
    main()
