"""The search: R independent guided trajectories, then selection of one design.

This is the algorithm every experiment runs; an experiment's driver only supplies the models, the
loss, the target and a `GuidanceConfig` factory. Each trajectory is `optimize_guided` (guided reverse
diffusion, then optional refinement, as set in the config). Selection takes the trajectory with the
smallest final loss, or, with `select_top = K`, re-scores the K smallest with `select_n` fresh
conditional samples (averaged over `select_repeats` draws and over any extra conditional samplers)
and takes the smallest of those.

The command-line flags of the search are shared by all drivers through `add_search_args` and
`search_overrides`, so every experiment exposes the same options under the same names.
"""

import dataclasses
import time

import torch

from .core import optimize_guided, _final_loss


def add_search_args(parser):
    """Add the search options (restarts, gradient estimator, refinement, selection) to an argparse parser."""
    g = parser.add_argument_group("search (shared by all experiments)")
    g.add_argument("--n-restarts", type=int, default=1, help="independent guided trajectories per run")
    g.add_argument("--guide-grad", choices=["pathwise", "zo"], default="pathwise",
                   help="loss gradient in the guided steps: backpropagation through the conditional sampler, or "
                        "zeroth-order central differences with common random numbers, pulled back through the "
                        "clean estimate (2 dim(x) + 1 loss values per step, or 2M + 1 with --zo-dirs M)")
    g.add_argument("--zo-eps", type=float, default=0.05, help="zeroth-order finite-difference step")
    g.add_argument("--zo-dirs", type=int, default=0, help="0: all coordinate directions; M > 0: M random directions")
    g.add_argument("--refine-steps", type=int, default=0, help="refinement steps on the sample loss after guidance")
    g.add_argument("--refine-lr", type=float, default=0.02)
    g.add_argument("--refine-n-cond", type=int, default=None, help="conditional samples per refinement step")
    g.add_argument("--refine-optim", choices=["adam", "sgd"], default="adam")
    g.add_argument("--refine-lambda", type=float, default=0.0, help="prior-score regularization of the refinement")
    g.add_argument("--refine-grad", choices=["pathwise", "zo"], default="pathwise")
    g.add_argument("--refine-keep-best", action="store_true",
                   help="return the refinement iterate with the lowest loss estimate instead of the last one")
    g.add_argument("--select-top", type=int, default=0,
                   help="re-score the K trajectories with the smallest final loss before choosing (0: no re-scoring)")
    g.add_argument("--select-n", type=int, default=2000, help="conditional samples per re-scoring")
    g.add_argument("--select-repeats", type=int, default=5, help="independent re-scorings averaged")
    g.add_argument("--select-samplers", nargs="*", default=[],
                   help="extra conditional-sampler checkpoints; the re-scoring loss is averaged over all samplers")
    return g


def search_overrides(args):
    """The `GuidanceConfig` fields set by the shared search flags."""
    return dict(guide_grad=args.guide_grad, zo_eps=args.zo_eps, zo_dirs=args.zo_dirs,
                refine_steps=args.refine_steps, refine_lr=args.refine_lr, refine_n_cond=args.refine_n_cond,
                refine_optim=args.refine_optim, refine_lambda=args.refine_lambda, refine_grad=args.refine_grad,
                refine_keep_best=args.refine_keep_best)


def _to_list(v):
    return v.detach().float().cpu().tolist() if torch.is_tensor(v) else v


def run_trajectory(model_uncond, model_cond, loss_function, cfg, seed, run=optimize_guided, evaluate=None):
    """One trajectory. Returns a JSON-ready record; the design tensor itself is under the key "x"."""
    t0 = time.time()
    res = run(model_uncond, model_cond, loss_function, cfg)
    x = res["x_final"].detach()
    rec = {"seed": seed, "x": x, "x_hat": x.float().view(-1).cpu().tolist(), "final_loss": float(res["final_loss"]),
           "best_loss": res.get("best_loss"), "n_steps": res.get("n_steps"), "time_s": time.time() - t0,
           "trace": res.get("trace")}
    if res.get("refine") is not None:
        rec["refine"] = {k: _to_list(v) for k, v in res["refine"].items()}
    if evaluate is not None:
        rec.update(evaluate(x))
    return rec, res


def select(runs, model_cond, loss_function, cfg, select_top=0, select_n=2000, select_repeats=5, extra_samplers=()):
    """Choose one run. `extra_samplers` is a list of (sampler, config transform) pairs; the transform adapts the
    config to that sampler (e.g. its output slicing). Adds "select_loss" to the re-scored runs."""
    if not select_top or len(runs) <= 1:
        return min(runs, key=lambda q: q["final_loss"])
    samplers = [(model_cond, lambda c: c)] + list(extra_samplers)
    cfg_sel = dataclasses.replace(cfg, final_n_cond=select_n)
    cand = sorted(runs, key=lambda q: q["final_loss"])[:select_top]
    for q in cand:
        tot = sum(float(_final_loss(s, loss_function, tf(cfg_sel), q["x"]))
                  for s, tf in samplers for _ in range(select_repeats))
        q["select_loss"] = tot / (select_repeats * len(samplers))
    return min(cand, key=lambda q: q["select_loss"])


def search(model_uncond, model_cond, loss_function, make_cfg, seeds, *, select_top=0, select_n=2000,
           select_repeats=5, extra_samplers=(), evaluate=None, run=optimize_guided, verbose=False):
    """R = len(seeds) independent trajectories, then `select`.

    make_cfg(seed) -> GuidanceConfig for one trajectory. Returns {"best", "runs", "time_s"}; records keep the
    design tensor under "x" (drop it with `strip` before saving)."""
    t0 = time.time()
    runs, cfg = [], None
    for s in seeds:
        cfg = make_cfg(s)
        rec, _ = run_trajectory(model_uncond, model_cond, loss_function, cfg, s, run=run, evaluate=evaluate)
        runs.append(rec)
        if verbose:
            print(f"    trajectory seed {s}: final loss {rec['final_loss']:.5f} ({rec['time_s']:.1f}s)", flush=True)
    best = select(runs, model_cond, loss_function, cfg, select_top, select_n, select_repeats, extra_samplers)
    return {"best": best, "runs": runs, "time_s": time.time() - t0}


def strip(rec, drop=("x",)):
    """A record without the keys in `drop` (the tensor), ready for json.dump."""
    return {k: v for k, v in rec.items() if k not in drop}
