"""
GB-FESO on the Gaussian-mixture experiments: the competitor of the paper, run on our networks.

GB-FESO (Zick & Mendels, arXiv:2607.06421) optimizes the condition x directly: from a start x0 it
draws a batch from the frozen conditional sampler at x, scores it against a fixed target set with a
distributional loss, and takes Adam steps on x through the differentiated sampler. No prior over X,
no diffusion over X. This script ports their optimizer loop (`GB-FESO/run_feso.py`,
`optimize_epsilon_condition` / `run_phase`, commit 4e021a4) onto this repository's samplers, targets
and evaluation, so it is scored exactly like search_mog.py's MLGD:

  phase 0  --steps Adam steps at --lr; after each step Gaussian noise of std --noise-std * (1 - k/steps)
           (decaying linearly to 0) is added to x; x is clamped to the box after each step; the
           iterate with the lowest loss is kept.
  phase 1  from the best phase-0 point, --steps more Adam steps at --lr-refine, no noise, best iterate
           kept. Their code runs it only when the best phase-0 KL exceeds 0.2; --phase1 always (our
           default) runs it in every case, so the KL and MMD arms are treated identically
           (--phase1 threshold --phase1-threshold 0.2 reproduces their rule).
  answer   the better of the two best iterates, each scored on the same --final-n fresh generated and
           target samples (their code compares the phases' search losses, which here can come from
           different sample counts); the run returns the start with the lowest score (MLGD scores
           and selects its restarts the same way). With --select-top K the K lowest-scoring starts
           are re-scored with --select-n fresh samples, averaged over --select-repeats draws, and the
           lowest of those is returned: MLGD's selection (cdm.guidance.search.select), same code.

Samples: phase 0 draws --num-samples generated samples per step, phase 1 --num-samples-phase1, and
every loss is computed against the same number of target samples, drawn fresh at each step (as MLGD
does). Their code instead keeps one fixed target set (4,496 frames of a simulation); --fixed-target N
restores that. Their command-line defaults otherwise: 201 steps, lr 0.05, lr_refine 0.05, 3,500
samples per step, KDE bandwidth 0.5, kl_lambda 1 (symmetric KL), noise 0.05, clamping on. Two of their rules have no counterpart here and are not ported: the 0.005 noise for
starts with both eps >= 0.9 (tied to their eps units), and the corner starts. Each start is
instead a random row of X from the training data (--init data): a real design, as their starts are
valid eps values, drawn without any part of our method (--init randn: x0 ~ N(0, I), the
distribution of MLGD's x_T). The box is the range of X in the data.

--grad zo   (with --grad-phase1) replaces backpropagation through the sampler by the finite-difference
            gradient of MLGD* (cdm.guidance.core._zo_grad: central differences of the sample loss along the
            coordinate directions of x, step --zo-eps, common random numbers), so that the search can be
            compared with MLGD* at the same gradient estimator. MLGD* uses finite differences in the guided
            steps and backpropagation in the refinement: --grad zo --grad-phase1 pathwise mirrors it.

--loss kl   their KDE-KL (cdm/losses/kde_kl.py, copied from their loss.py), Euclidean coordinates
--loss mmd  the multi-bandwidth RBF MMD of Table 1 (detached bandwidth), same samples and optimizer

A run (attempt) is --n-restarts starts; the start whose answer has the lowest score is returned, as
for MLGD's restarts. The JSON has the layout of search_mog.py, so make_table1.py --collect scores it.

    python experiments/synthetic/pipeline/search_gbfeso.py --sampler runs/table1v2/x5_y5/models/diffusion/model_conditional.pt \\
        --target runs/table1v2/x5_y5/data.pt --loss kl --n-restarts 10 --n-attempts 5 --out GBFESO-KL.json
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # sibling pipeline modules

from cdm.guidance.config import GuidanceConfig                      # noqa: E402
from cdm.guidance.core import _draw_conditional, _zo_grad           # noqa: E402
from cdm.guidance.search import select                              # noqa: E402
from cdm.losses.functions import MMDLoss, RBF                        # noqa: E402
from cdm.losses.kde_kl import KDEKLLoss                              # noqa: E402
from run_optimization import load_model                              # noqa: E402
from search_mog import load_target, evaluate                         # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("inputs")
    g.add_argument("--prior", default=None, help="ignored (GB-FESO has no prior over X); accepted for make_table1.py")
    g.add_argument("--sampler", required=True, help="conditional checkpoint (train_models.py --model conditional)")
    g.add_argument("--sampler-type", choices=["auto", "cm", "diffusion"], default="auto")
    g.add_argument("--target", required=True, help="a generate_data.py .pt (mixture, x*, joint)")
    g = p.add_argument_group("loss")
    g.add_argument("--loss", choices=["kl", "mmd"], default="kl")
    g.add_argument("--kde-bandwidth", type=float, default=0.5)
    g.add_argument("--kl-lambda", type=float, default=1.0, help="weight of the reverse KL (their default 1)")
    g.add_argument("--rbf-kernels", type=int, default=5)
    g.add_argument("--rbf-mul", type=float, default=2.0)
    g = p.add_argument_group("optimizer (their CLI defaults)")
    g.add_argument("--steps", type=int, default=201, help="Adam steps per phase")
    g.add_argument("--lr", type=float, default=5e-2)
    g.add_argument("--lr-refine", type=float, default=5e-2)
    g.add_argument("--noise-std", type=float, default=0.05, help="phase-0 perturbation std, decayed linearly to 0")
    g.add_argument("--phase1", choices=["always", "threshold", "never"], default="always")
    g.add_argument("--phase1-threshold", type=float, default=0.2, help="--phase1 threshold: run when best phase-0 loss > this")
    g.add_argument("--num-samples", type=int, default=3500, help="generated (and target) samples per phase-0 step")
    g.add_argument("--num-samples-phase1", type=int, default=None, help="per phase-1 step (default --num-samples)")
    g.add_argument("--final-n", type=int, default=None, help="samples scoring each start's answer (default the phase-1 count)")
    g.add_argument("--fixed-target", type=int, default=0,
                   help="N > 0: their scheme, one fixed set of N target samples per run (their 4,496) instead of fresh draws")
    g.add_argument("--grad", choices=["pathwise", "zo"], default="pathwise",
                   help="phase-0 gradient: backpropagation through the sampler (theirs), or MLGD*'s finite differences "
                        "(central, coordinate directions, common random numbers: 2 dim(x) + 1 loss values per step)")
    g.add_argument("--grad-phase1", choices=["pathwise", "zo"], default=None, help="phase-1 gradient (default: --grad)")
    g.add_argument("--zo-eps", type=float, default=0.05, help="finite-difference step (MLGD*'s default)")
    g.add_argument("--no-clamp", action="store_true", help="do not clamp x to the box (their CLI clamps)")
    g.add_argument("--init", choices=["data", "randn"], default="data",
                   help="start of each search: data = a random row of X from the training data (a real design, as "
                        "their starts are valid eps values); randn = x0 ~ N(0, I), the distribution of MLGD's x_T")
    g = p.add_argument_group("run control")
    g.add_argument("--n-restarts", type=int, default=10, help="starts per run; the best by loss is the run's answer")
    g.add_argument("--n-attempts", type=int, default=1)
    g.add_argument("--select-top", type=int, default=0,
                   help="re-score the K lowest-scoring starts (MLGD's selection); 0: the lowest score wins")
    g.add_argument("--select-n", type=int, default=2000, help="--select-top: fresh samples per re-score")
    g.add_argument("--select-repeats", type=int, default=5, help="--select-top: re-scores averaged per start")
    g.add_argument("--n-cond", type=int, default=None, help="ignored; accepted for make_table1.py")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--out", required=True)
    a, unknown = p.parse_known_args()
    a.num_samples_phase1 = a.num_samples_phase1 or a.num_samples
    a.final_n = a.final_n or a.num_samples_phase1
    a.grad_phase1 = a.grad_phase1 or a.grad
    if unknown:   # make_table1.py passes MLGD flags that have no meaning here
        print(f"[search_gbfeso] ignoring flags not used by GB-FESO: {' '.join(unknown)}")
    return a


def run_phase(sampler, cfg, loss_fn, target, x_start, steps, lr, noise_std, lo, hi, grad="pathwise", seed=0):
    """Port of their run_phase: Adam on x through the sampler, optional decaying noise, clamping,
    best iterate kept. `target(n)` returns the target samples of a step (fresh, or their fixed set);
    cfg.n_cond is the number of generated samples. grad="zo": the gradient of each step is the
    finite-difference estimate of MLGD* (`_zo_grad`, seeded by `seed` and the step) instead of
    backpropagation; the loss, optimizer, noise, clamping and best-iterate rule are unchanged.
    Returns (best_x, best_loss, trace)."""
    x = x_start.clone().detach()
    if lo is not None:
        x = torch.max(torch.min(x, hi), lo)
    x.requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    best_loss, best_x, trace = float("inf"), x.detach().clone(), []
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        if grad == "zo":
            loss, g = _zo_grad(sampler, loss_fn, cfg, cfg, x.detach(), cfg.n_cond, seed=seed * 100003 + step)
            x.grad = g.to(x.dtype).view_as(x)
        else:
            generated = _draw_conditional(sampler, x, cfg)
            loss = loss_fn(generated, target(cfg.n_cond))
            if step < steps:
                loss.backward()
        value = float(loss.detach())
        trace.append(value)
        if value < best_loss:
            best_loss, best_x = value, x.detach().clone()
        if step == steps:
            break
        opt.step()
        with torch.no_grad():
            if noise_std > 0:
                x.add_(noise_std * (1.0 - step / max(steps, 1)) * torch.randn_like(x))
            if lo is not None:
                x.copy_(torch.max(torch.min(x, hi), lo))
    return best_x, best_loss, trace


def main():
    args = parse_args()
    dev = args.device
    sampler, arch = load_model(args.sampler, dev)
    sampler_type = arch if args.sampler_type == "auto" else args.sampler_type
    for prm in sampler.parameters():
        prm.requires_grad_(False)
    method = (f"GBFESO-{args.loss.upper()}" + ("-F" if sampler_type == "cm" else "")
              + ("" if args.grad == args.grad_phase1 == "pathwise" else
                 "-ZO" if args.grad == args.grad_phase1 == "zo" else f"-{args.grad}-{args.grad_phase1}"))
    target_gen, info = load_target(args.target, dev)
    data = torch.load(args.target, map_location="cpu", weights_only=False)
    c = int(data["condition_on"])
    X = data["samples"][:, :c].float()
    lo, hi = (None, None) if args.no_clamp else (X.min(0).values.to(dev), X.max(0).values.to(dev))
    loss_fn = (KDEKLLoss(bandwidth=args.kde_bandwidth, kl_lambda=args.kl_lambda) if args.loss == "kl"
               else MMDLoss(kernel=RBF(n_kernels=args.rbf_kernels, mul_factor=args.rbf_mul, device=dev)))
    def sampling(n):
        return SimpleNamespace(n_cond=n, repeat_condition=True, device=dev, cond_sample_kwargs={},
                               condition_on_slice=(sampler_type == "diffusion"), cond_output_transform=None,
                               zo_eps=args.zo_eps, zo_dirs=0, target_generator=None)   # last three: --grad zo
    cfg0, cfg1, cfg_f = sampling(args.num_samples), sampling(args.num_samples_phase1), sampling(args.final_n)
    print(f"[search_gbfeso] {method} | loss={args.loss}"
          + (f" h={args.kde_bandwidth} lambda={args.kl_lambda}" if args.loss == "kl" else "")
          + f" | {args.steps} steps x 2 phases (phase 1: {args.phase1}) lr={args.lr}/{args.lr_refine} noise={args.noise_std}"
          + f" | samples/step {args.num_samples} (phase 0), {args.num_samples_phase1} (phase 1), {args.final_n} (score)"
          + f", target {'fixed ' + str(args.fixed_target) if args.fixed_target else 'fresh, same count'}"
          + f" | box={'off' if lo is None else 'data range'} | init={args.init}"
          + f" | grad={args.grad}/{args.grad_phase1}" + (f" (zo step {args.zo_eps})" if "zo" in (args.grad, args.grad_phase1) else "")
          + f" | R={args.n_restarts} starts, attempts={args.n_attempts}, device={dev}"
          + (f" | select top {args.select_top} on {args.select_n} x {args.select_repeats}" if args.select_top else ""))

    attempts = []
    for a in range(args.n_attempts):
        t_att = time.time()
        torch.manual_seed(args.seed * 1000 + a * args.n_restarts)
        if args.fixed_target:      # their scheme: one fixed target set per run
            with torch.no_grad():
                fixed = target_gen(args.fixed_target, dev).float()
            target = lambda n, f=fixed: f                                      # noqa: E731
        else:                      # fresh target samples at every step, as many as generated (as MLGD)
            target = lambda n: target_gen(n, dev).float()                      # noqa: E731
        for cfg in (cfg0, cfg1):   # the finite-difference gradient draws its target samples through the config
            cfg.target_generator = lambda n, d, t=target: t(n)
        runs = []
        for r in range(args.n_restarts):
            seed = args.seed * 1000 + a * args.n_restarts + r
            torch.manual_seed(seed)
            t0 = time.time()
            if args.init == "data":   # one random training design per start
                x0 = X[torch.randint(0, X.shape[0], (1,))].to(dev)
            else:
                x0 = torch.randn(1, c, device=dev)
            b0, l0, tr0 = run_phase(sampler, cfg0, loss_fn, target, x0, args.steps, args.lr, args.noise_std, lo, hi,
                                    grad=args.grad, seed=seed)
            do_p1 = args.phase1 == "always" or (args.phase1 == "threshold" and l0 > args.phase1_threshold)
            if do_p1:
                b1, l1, tr1 = run_phase(sampler, cfg1, loss_fn, target, b0, args.steps, args.lr_refine, 0.0, lo, hi,
                                        grad=args.grad_phase1, seed=seed + 500009)
            # score on fresh samples, the selection criterion (as MLGD's final loss). Both phases' answers are
            # scored on the same --final-n samples, since their search losses can come from different sample
            # counts (the minimum of many noisy small-batch losses is biased low).
            with torch.no_grad():
                score_fn = lambda xx: float(loss_fn(_draw_conditional(sampler, xx, cfg_f), target(args.final_n)))   # noqa: E731
                s0 = score_fn(b0)
                s1 = score_fn(b1) if do_p1 else float("inf")
            x_hat, loss, score, phase = (b1, l1, s1, "phase1") if s1 < s0 else (b0, l0, s0, "phase0")
            xv = x_hat.float().view(-1)
            runs.append({"seed": seed, "x0": x0.view(-1).tolist(), "x_hat": xv.cpu().tolist(), "final_loss": score,
                         "search_loss": loss, "phase": phase, "phase0_best_loss": l0, "phase0_score": s0,
                         "phase1_score": s1 if do_p1 else None,
                         "phase1_best_loss": l1 if do_p1 else None,
                         "trace": {"phase0": tr0, "phase1": tr1 if do_p1 else None},
                         "time_s": time.time() - t0, **evaluate(xv, info)})
            print(f"    start {r + 1}/{args.n_restarts}: score {score:.5f}, search loss {loss:.5f} ({phase})  {runs[-1].get('l2_gmm', float('nan')):.4f}"
                  f"  ({runs[-1]['time_s']:.1f}s)", flush=True)
        if args.select_top:   # MLGD's selection, same function: re-score the K lowest with fresh samples
            cfg_sel = GuidanceConfig(n_cond=args.final_n, repeat_condition=True, device=dev,
                                     condition_on_slice=(sampler_type == "diffusion"),
                                     target_generator=lambda n, d: target(n))
            for q in runs:
                q["x"] = torch.tensor(q["x_hat"], device=dev).view(1, -1)
            best = select(runs, sampler, loss_fn, cfg_sel, args.select_top, args.select_n, args.select_repeats)
            for q in runs:
                q.pop("x")
            print("    selection: " + ", ".join(f"{q['final_loss']:.5f} -> {q['select_loss']:.5f}"
                                               for q in runs if "select_loss" in q), flush=True)
        else:
            best = min(runs, key=lambda q: q["final_loss"])
        att = {"attempt": a, **{k: v for k, v in best.items() if k != "trace"},
               "restarts": [{k: v for k, v in q.items() if k != "trace"} for q in runs],
               "traces": [q["trace"] for q in runs], "time_s": time.time() - t_att}
        attempts.append(att)
        print(f"  [{a + 1}/{args.n_attempts}] loss={att['final_loss']:.5f}  x_hat={np.round(np.array(att['x_hat']), 4)}"
              f"  l2_gmm={att.get('l2_gmm', float('nan')):.4f}  ({att['time_s']:.1f}s)")

    summary = {}
    for k in ("final_loss", "l2_gmm", "l2_to_xstar", "time_s"):
        vals = [q[k] for q in attempts if k in q]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    best = min(attempts, key=lambda q: q["final_loss"])
    print(f"[search_gbfeso] {method}: " + "  ".join(f"{k} {v['mean']:.4f}+-{v['std']:.4f}" for k, v in summary.items()))
    with open(args.out, "w") as f:
        json.dump({"method": method, "algo": "gbfeso", "budget": args.n_restarts, "args": vars(args),
                   "summary": summary, "best": best, "attempts": attempts,
                   "x_star": None if info.get("x_star") is None else info["x_star"].float().view(-1).tolist()}, f, indent=2)
    print(f"[search_gbfeso] wrote {args.out}")


if __name__ == "__main__":
    main()
