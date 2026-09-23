"""
Find x*: the input whose induced conditional P(Y | X = x) matches a target G(Y).

This is the paper's method as one command. Give it a prior over X (a trained
unconditional diffusion checkpoint), a conditional sampler for Y | X (a trained
consistency-model or conditional-diffusion checkpoint) and a target G(Y); it
runs guided reverse diffusion over X with the distributional loss as guidance,
as independent restarts or as a resampled particle population, optionally
polishes the answer with a few gradient steps, and returns the best x_hat.

    python scripts/find_xstar.py --prior runs/mog4d/models/model_unconditional.pt \
        --sampler runs/mog4d/models/model_conditional.pt --target runs/mog4d/data.pt \
        --n-attempts 10 --out runs/mog4d/xstar.json

Inputs
  --prior         checkpoint from scripts/train_models.py --model unconditional
  --sampler       checkpoint from scripts/train_models.py --model conditional, or 'exact':
                  the analytic conditional of the target file's mixture as an oracle f_phi
                  (differentiable by reparametrisation) to measure the search without sampler error
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
  --select all|uniform|witness --n-sel k   backpropagate through all n_cond samples, or
                      through k of them (rest detached): a uniform random subset, or rows drawn
                      proportionally to the MMD witness |w(y_i)| (utils/witness.py), scored
                      against the same target batch the loss uses
  --witness-floor F   witness: mix F of the uniform distribution into the selection  (0.0)
  --rescale none|ht   Horvitz-Thompson: divide each kept row's gradient by its inclusion
                      probability, so the subsampled gradient is unbiased for the full one (none)

Outer loop (Algorithm 1)
  --n-mc M            Monte-Carlo perturbations of the Tweedie point (lgd: 3, dps: 1)
  --perturb           beta_ratio | beta_prod_ratio_sq | sqrt_beta_prod | none
  --aggregate         logsumexp | mean
  --step-size S       a constant zeta, or 'beta_t' for zeta_t = zeta' * beta_t
  --zeta Z            zeta' used with --step-size beta_t                (1.0)
  --eta E             DDIM stochasticity, 0 = deterministic              (0.0)
  --guidance-cutoff-t T   no guidance for t < T (MNIST used 200)
  --init randn|zeros  x_T initialisation (paper: randn)
  --guidance-optim sgd|adam   plain gradient push (Eq. 1) or Adam-style push with moments
                      across the reverse steps (AdamDPS-like); --adam-betas b1 b2

Trajectories: independent restarts or a particle population
  --algo lgd          R = --n-restarts independent trajectories; the smallest
                      final loss wins (default)
  --algo particles    --n-particles trajectories advance in lockstep; after each
                      step they are weighted by exp(-beta * loss) and, when the
                      effective sample size drops below --ess-frac * N, resampled
                      (systematic), so trajectories stuck in a poor basin are
                      replaced by copies of better ones while the noise can still
                      spread the copies. Same cost as N restarts. The best particle
                      by final loss is returned, all particles are written out.
  --n-particles N     (10)     --resample-beta B  (5.0; 0 disables resampling)
  --ess-frac F        (0.5)    --resample-every k (1)    --weight-mode fresh|accumulated
  --resample-schedule constant|linear   linear ramps the strength 0 -> B (annealed)

Post-refinement (optional, either --algo): K plain gradient steps on the plug-in
loss from the returned x, no diffusion, optionally with a prior term
  --refine-steps K (0)  --refine-lr (0.05)  --refine-lambda L (0: none)
  --refine-optim adam|sgd (adam)  --refine-n-cond N (samples per refinement step; default n_cond)
  update: Adam/SGD step on grad L_hat(x) - L * score_theta(x)

Run control
  --n-restarts R      lgd: trajectories per attempt, best final loss returned      (10)
  --n-particles N     particles: population size, same role                        (10)
  --n-attempts A      independent runs of the whole algorithm; the JSON summary is
                      mean/std of the returned x_hat's loss over them               (1)
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

from guidance import GuidanceConfig, optimize_guided, optimize_guided_population, AdamGuidance   # noqa: E402
from guidance.config import update_ddim_minus_grad                                             # noqa: E402
from guidance.config import (init_randn, init_zeros, perturb_beta_ratio,               # noqa: E402
                             perturb_beta_prod_ratio_sq, perturb_sqrt_beta_prod, perturb_none,
                             aggregate_logsumexp, aggregate_mean, aggregate_single)
from losses.functions import MMDLoss, RBF, sliced_wasserstein_distance                  # noqa: E402
from utils import witness as W                                                          # noqa: E402
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
    g.add_argument("--sampler", required=True, help="conditional checkpoint, or 'exact' for the analytic conditional (oracle)")
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
    g.add_argument("--guidance-optim", choices=["sgd", "adam"], default="sgd",
                   help="adam: Adam-style guidance push (moments across the reverse steps, AdamDPS-like); "
                        "the step size is then a displacement in x units")
    g.add_argument("--adam-betas", type=float, nargs=2, default=[0.9, 0.999])
    g.add_argument("--algo", choices=["lgd", "particles"], default="lgd",
                   help="lgd: independent trajectories, best of --n-attempts; particles: --n-particles trajectories "
                        "run in lockstep with loss-weighted resampling (one population per attempt)")
    g.add_argument("--n-restarts", type=int, default=10, help="lgd: trajectories per attempt; the best final loss is returned")
    g.add_argument("--n-particles", type=int, default=10, help="particles: population size (same role as --n-restarts)")
    g.add_argument("--resample-beta", type=float, default=5.0, help="particle weights exp(-beta*loss); 'none' disables")
    g.add_argument("--ess-frac", type=float, default=0.5, help="resample when ESS < ess_frac * n_particles")
    g.add_argument("--resample-every", type=int, default=1)
    g.add_argument("--weight-mode", choices=["fresh", "accumulated"], default="fresh")
    g.add_argument("--resample-schedule", choices=["constant", "linear"], default="constant",
                   help="linear: resampling strength ramps from 0 to --resample-beta over the reverse process (annealing)")
    g = p.add_argument_group("post-refinement (any --algo)")
    g.add_argument("--refine-steps", type=int, default=0, help="gradient steps on the plug-in loss from the returned x")
    g.add_argument("--refine-lr", type=float, default=0.05)
    g.add_argument("--refine-lambda", type=float, default=0.0, help="weight of the prior score term")
    g.add_argument("--refine-optim", choices=["adam", "sgd"], default="adam")
    g.add_argument("--refine-n-cond", type=int, default=None, help="conditional samples per refinement step (default n_cond)")
    g = p.add_argument_group("inner estimator")
    g.add_argument("--n-cond", type=int, default=250)
    g.add_argument("--n-target", type=int, default=None)
    g.add_argument("--loss", choices=["mmd", "swd"], default="mmd")
    g.add_argument("--rbf-kernels", type=int, default=5)
    g.add_argument("--rbf-mul", type=float, default=2.0)
    g.add_argument("--swd-projections", type=int, default=100)
    g.add_argument("--select", choices=["all", "uniform", "witness"], default="all")
    g.add_argument("--n-sel", type=int, default=None, help="samples that keep their gradient under --select uniform|witness")
    g.add_argument("--witness-floor", type=float, default=0.0,
                   help="witness: selection p_i = floor/n + (1 - floor) |w_i| / sum |w|")
    g.add_argument("--rescale", choices=["none", "ht"], default="none",
                   help="ht: kept rows' gradients divided by their inclusion probabilities (Horvitz-Thompson)")
    g = p.add_argument_group("run control")
    g.add_argument("--n-attempts", type=int, default=1,
                   help="independent runs of the whole algorithm (each returns one x_hat); mean/std are over these")
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
    if a.select != "all" and a.n_sel is None:
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


def make_subsampler(args, target_generator):
    """--select uniform|witness: keep the gradient on n_sel of the samples, detach the rest (the
    loss value still uses all of them); --rescale ht divides the kept gradients by their inclusion
    probabilities. Returns (transform, target_generator). For witness the transform draws the
    target batch itself to score the samples and hands that same batch to the loss: the core
    always calls the target generator right after the conditional draw."""
    if args.select == "all":
        return None, target_generator
    pending = []
    kernel = RBF(n_kernels=args.rbf_kernels, mul_factor=args.rbf_mul, device=args.device)

    def keep_subset(y):
        target = None
        if args.select == "witness":
            target = target_generator(y.shape[0], y.device)
            pending[:] = [target]
        keep, pi = W.select(y, args.n_sel, args.select, target=target, kernel=kernel, floor=args.witness_floor)
        return W.reweight(y, keep, pi, args.rescale)

    def gen(n, dev):
        return pending.pop().to(dev) if pending else target_generator(n, dev)
    return keep_subset, (gen if args.select == "witness" else target_generator)


def make_cfg(args, prior, sampler_type, target_generator, seed):
    if args.step_size == "beta_t":
        step_fn = lambda t, i, z=args.zeta: z * float(prior.betas[t])     # noqa: E731
    else:
        step_fn = lambda t, i, z=float(args.step_size): z                # noqa: E731
    perturb = PERTURB[args.perturb] if args.n_mc > 1 or args.perturb != "none" else perturb_none
    aggregate = aggregate_single if args.n_mc == 1 and args.perturb == "none" else \
        {"logsumexp": aggregate_logsumexp, "mean": aggregate_mean}[args.aggregate]
    gen = target_generator if not args.n_target else (lambda n, d, g=target_generator: g(args.n_target, d))
    transform, gen = make_subsampler(args, gen)
    return GuidanceConfig(
        n_cond=args.n_cond, n_mc=args.n_mc, final_n_cond=args.final_n_cond or 4 * args.n_cond,
        init_fn=init_randn if args.init == "randn" else init_zeros, unsqueeze_init=True,
        perturb_fn=perturb, perturb_square=(args.perturb == "beta_prod_ratio_sq"),
        aggregate_fn=aggregate, step_size_fn=step_fn, guidance_cutoff_t=args.guidance_cutoff_t,
        update_fn=(AdamGuidance(*args.adam_betas) if args.guidance_optim == "adam" else update_ddim_minus_grad),
        timestep_source="range", ddim_step_style="condition_x", eta=args.eta,
        repeat_condition=True, condition_on_slice=(sampler_type == "diffusion"),
        cond_output_transform=transform,
        target_generator=gen, compute_final_loss=True,
        n_particles=args.n_particles, resample_beta=(None if args.resample_beta is None or args.resample_beta <= 0
                                                       else args.resample_beta),
        ess_frac=args.ess_frac, resample_every=args.resample_every, weight_mode=args.weight_mode,
        resample_schedule=args.resample_schedule,
        refine_steps=args.refine_steps, refine_lr=args.refine_lr, refine_lambda=args.refine_lambda,
        refine_optim=args.refine_optim, refine_n_cond=args.refine_n_cond,
        device=args.device, seed=seed, progress=args.progress,
        tag=f"find_xstar:{args.guidance}:{args.algo}:{'MLGD-F' if sampler_type == 'cm' else 'MLGD'}",
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
    if args.sampler == "exact":   # oracle: the analytic conditional of the target file's mixture, no sampler error
        from samplers.exact import ExactConditionalSampler
        _, info0 = load_target(args.target, "cpu")
        assert info0.get("mu_list") is not None, "--sampler exact needs a target file with the mixture"
        c0 = torch.load(args.target, map_location="cpu", weights_only=False)["condition_on"]
        sampler, sampler_arch = ExactConditionalSampler(info0["mu_list"], info0["Sigma_list"], info0["alpha"], c0), "exact"
    else:
        sampler, sampler_arch = load_model(args.sampler, dev)
    assert prior_arch == "diffusion", "--prior must be an unconditional diffusion checkpoint"
    sampler_type = sampler_arch if args.sampler_type == "auto" else args.sampler_type
    method = ("P" if args.algo == "particles" else "") + {"cm": "MLGD-F", "exact": "MLGD-X"}.get(sampler_type, "MLGD")
    R = args.n_particles if args.algo == "particles" else args.n_restarts
    target_generator, info = load_target(args.target, dev)
    loss_fn = make_loss(args)
    print(f"[find_xstar] {method} with {args.guidance.upper()} guidance | loss={args.loss} n_cond={args.n_cond} "
          f"n_mc={args.n_mc} perturb={args.perturb} aggregate={args.aggregate} step={args.step_size} "
          f"select={args.select}{'' if args.select == 'all' else f'(n_sel={args.n_sel}, rescale={args.rescale}' + (f', floor={args.witness_floor}' if args.select == 'witness' else '') + ')'} init={args.init} eta={args.eta} "
          f"| budget R={R} {'particles' if args.algo == 'particles' else 'restarts'}, attempts={args.n_attempts} "
          f"target={info['kind']} device={dev}"
          + (f" | resample_beta={args.resample_beta} ess_frac={args.ess_frac}" if args.algo == "particles" else "")
          + (f" | refine: {args.refine_steps} steps lr={args.refine_lr} lambda={args.refine_lambda}"
             if args.refine_steps > 0 else ""))

    def one_trajectory(seed, run):
        cfg = make_cfg(args, prior, sampler_type, target_generator, seed)
        t0 = time.time()
        res = run(prior, sampler, loss_fn, cfg)
        x_hat = res["x_final"].detach().float().view(-1)
        rec = {"seed": seed, "x_hat": x_hat.cpu().tolist(), "final_loss": float(res["final_loss"]),
               "best_loss": res["best_loss"], "n_steps": res["n_steps"], "time_s": time.time() - t0,
               **evaluate(x_hat, info), "trace": res["trace"]}
        if args.algo == "particles":
            rec.update({"best_particle": res["best_particle"], "n_resamples": res["n_resamples"],
                        "resample_steps": res["resample_steps"],
                        "particles": [{"x_hat": x.tolist(), "final_loss": l, **evaluate(x, info)}
                                      for x, l in zip(res["x_particles"].float().cpu(), res["final_losses"])]})
        if res.get("refine") is not None:
            rec["refine"] = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in res["refine"].items()}
        if not args.no_store:
            save_run("find_xstar", cfg, {k: v for k, v in rec.items() if k not in ("x_hat", "trace", "particles", "refine")},
                     seed=seed, method=method, guidance=args.guidance, prior=args.prior, sampler=args.sampler, target=args.target)
        return rec, cfg

    attempts = []
    for a in range(args.n_attempts):
        t_att = time.time()
        if args.algo == "particles":
            rec, cfg = one_trajectory(args.seed * 1000 + a, optimize_guided_population)
            att = {"attempt": a, **rec}
        else:   # R independent restarts, the smallest final loss wins: that is the algorithm's answer
            runs = [one_trajectory(args.seed * 1000 + a * R + r, optimize_guided)[0] for r in range(R)]
            best = min(runs, key=lambda q: q["final_loss"])
            att = {"attempt": a, **{k: v for k, v in best.items() if k != "trace"},
                   "restarts": [{k: v for k, v in q.items() if k != "trace"} for q in runs],
                   "traces": [q["trace"] for q in runs], "time_s": sum(q["time_s"] for q in runs)}
        att["time_s"] = time.time() - t_att
        attempts.append(att)
        extra = "  ".join(f"{k}={v:.4f}" for k, v in att.items() if k in ("l2_gmm", "l2_to_xstar"))
        if args.algo == "particles":
            extra += f"  resamplings={att['n_resamples']}"
        if att.get("refine") is not None and att["refine"].get("final_loss_before") is not None:
            extra += f"  loss_before_refine={att['refine']['final_loss_before']:.5f}"
        print(f"  [{a+1}/{args.n_attempts}] loss={att['final_loss']:.5f}  x_hat={np.round(np.array(att['x_hat']), 4)}  {extra}  ({att['time_s']:.1f}s)")

    best = min(attempts, key=lambda a: a["final_loss"])
    summary = {}
    for k in ("final_loss", "l2_gmm", "l2_to_xstar", "time_s"):
        vals = [a[k] for a in attempts if k in a]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    print(f"[find_xstar] {method}: over {args.n_attempts} attempt(s) "
          + "  ".join(f"{k} {v['mean']:.4f}+-{v['std']:.4f}" for k, v in summary.items())
          + f" | best attempt {best['attempt']+1}: loss={best['final_loss']:.5f}, x_hat={np.round(np.array(best['x_hat']), 4)}")
    if info.get("x_star") is not None:
        print(f"[find_xstar] true x* = {np.round(info['x_star'].float().view(-1).numpy(), 4)}")
    out = args.out or os.path.splitext(args.target)[0] + "_xstar.json"
    with open(out, "w") as f:
        json.dump({"method": method, "algo": args.algo, "guidance": args.guidance, "budget": R, "args": vars(args),
                   "summary": summary, "best": best, "attempts": attempts,
                   "x_star": None if info.get("x_star") is None else info["x_star"].float().view(-1).tolist()}, f, indent=2)
    print(f"[find_xstar] wrote {out}")


if __name__ == "__main__":
    main()
