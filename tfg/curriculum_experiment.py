"""EXP-009 -- tied curriculum K_t = n_t, end-to-end 2D optimisation.

Four methods, identical outer diffusion randomness and identical restart seeds:

  M1 ``pointwise``   K = n = 1 at every step (ordinary point-target TFG tier).
  M2 ``full``        n = 250 against the full fixed S_G at every step
                     (the original CDM; high-compute upper baseline).
  M3 ``adaptive_n``  scheduled n_t, but ALWAYS the full fixed S_G.
  M4 ``curriculum``  scheduled K_t = n_t, target G^(K_t) from the hierarchy.

M3 and M4 share the same n_t schedule and therefore the same
conditional-generation compute; comparing them isolates target-resolution
adaptation from sample-count adaptation.

Deviations from ``optimize_LGD``, both deliberate and both documented:
  * ``num_x_t = 1`` (the repository uses 3 Monte-Carlo x0 draws per step). With
    1, the conditional-generation cost per step is exactly n_t, which keeps the
    matched-compute comparison clean. Applied identically to all four methods.
  * fixed bandwidth 46.550137 (EXP-008 / decision D11), never the adaptive one,
    because the adaptive bandwidth makes the objective n_t-dependent and would
    confound the K_t effect.

Everything else follows the repository: x_T = 0, DDIM eta = 0, guidance applied
as ``x_{t-1} = DDIM(x_t) - grad`` (rho = 1), loop t = T-1 .. 1.
"""

import argparse
import hashlib
import math
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tfg._compat import ensure_ot_stub          # noqa: E402

ensure_ot_stub()

from tfg import oracle                           # noqa: E402
from tfg.n_schedule import n_at                  # noqa: E402
from tfg.run_estimator_v2 import make_fixed_target_set   # noqa: E402
from tfg.schedule import DiffusionSchedule       # noqa: E402
from tfg.target_hierarchy import (               # noqa: E402
    TargetHierarchy, pointwise_loss, weighted_mmd2,
)
from tfg.train_cm import PAPER_TS, SEEDS, load_one       # noqa: E402
from tfg.train_uncond import load as load_uncond          # noqa: E402

FIXED_BW = 46.550137          # EXP-008
X_OPT = -4.9984462966235057   # EXP-003
N_MAX = 250
METHODS = ("pointwise", "full", "adaptive_n", "curriculum")


def key_seed(*p):
    return int.from_bytes(hashlib.blake2b(repr(p).encode(), digest_size=8).digest(),
                          "big") % (2 ** 31 - 1)


def git_meta():
    def run(*a):
        try:
            return subprocess.run(a, cwd=str(_ROOT), capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("git", "status", "--porcelain")}


def schedule_n(t, sched, method, kappa, kind):
    if method == "pointwise":
        return 1
    if method == "full":
        return N_MAX
    return n_at(t, sched, N_MAX, kappa, kind)


def guidance_loss(y, K, hierarchy, method, bandwidth=FIXED_BW):
    """Loss for this step, given n_t conditional samples ``y`` (float64)."""
    if method == "pointwise":
        return pointwise_loss(y, hierarchy.mean()), 1
    if method in ("full", "adaptive_n"):
        c, w, _ = hierarchy.level(hierarchy.N)      # full empirical target
        return weighted_mmd2(y, c, w, bandwidth), hierarchy.N
    if K == 1:
        return pointwise_loss(y, hierarchy.mean()), 1
    c, w, _ = hierarchy.level(K)
    return weighted_mmd2(y, c, w, bandwidth), K


def optimize(model_cond, model_uncond, hierarchy, method, restart,
             kappa=1.0, kind="time", lam=0.0, device="cpu",
             num_x_t=1, legacy_compat=False, params=None, ts=PAPER_TS,
             bandwidth=FIXED_BW, reseed_per_step=True, n_override=None):
    """One optimisation run. Returns x_hat and a trajectory record.

    ``legacy_compat=True`` reproduces ``Optimization.optimize_LGD`` exactly:
    ``num_x_t`` Monte-Carlo x0 draws with the ``r_t`` perturbation, the target
    set REDRAWN inside the j loop at ``n`` samples, the ADAPTIVE bandwidth,
    log-mean-exp aggregation, the code-default ``ts``, and no per-step
    reseeding (the global RNG is consumed in legacy order).
    """
    T = model_uncond.diffusion_steps
    sched = DiffusionSchedule(T=T)
    x_t = torch.zeros(1, model_uncond.nfeatures, device=device)

    cond_calls = 0
    denoiser_calls = 0
    diverged = False
    g_bar = None
    traj = []
    t0 = time.perf_counter()

    for t in range(T - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)
        x_prev, pred_x0 = model_uncond.sample_ddim_step(
            x_t, t, condition_x=None, device=device, eta=0.0)
        denoiser_calls += 1

        n = (int(n_override) if n_override is not None
             else schedule_n(t, sched, method, kappa, kind))
        K = n                                    # tied curriculum

        if legacy_compat:
            from LossFunctions import MMDLoss, RBF
            import dist_utils as _du
            cur_var = model_uncond.betas[t].to(device)
            r_t = cur_var / torch.sqrt(1 + cur_var ** 2)
            mmd = MMDLoss(kernel=RBF(bandwidth=bandwidth, device="cpu"), device="cpu")
            tgt_means = params["target_means"].float()
            tgt_vars = torch.stack([v.float() for v in params["target_variances"]])
            tgt_w = params["target_weights"].float()
            losses = []
            for _ in range(num_x_t):
                x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
                cond = x0_sample.view(1, -1).repeat(n, 1)
                kw = {} if ts is None else {"ts": ts}
                y, _, _ = model_cond.sample(nsamples=n, condition_x=cond,
                                            device=device, **kw)
                cond_calls += n
                mog = _du.generate_mog_samples_not_differentiable(
                    n, tgt_means, tgt_vars, tgt_w)
                losses.append(-mmd(y, mog))
            loss = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
            K_used = n
        else:
            # Repository-default protocol: num_x_t Monte-Carlo x0 draws with the
            # r_t perturbation, aggregated by log-mean-exp exactly as
            # Optimization.py:113-132 does -- but against the FIXED target set
            # and at the FIXED bandwidth.
            cur_var = model_uncond.betas[t].to(device)
            r_t = cur_var / torch.sqrt(1 + cur_var ** 2)
            losses = []
            for j in range(num_x_t):
                # Key omits `method` so matched methods share the draw at each
                # (restart, t, j). Exact sharing holds only up to the smaller
                # n: the model's internal noise tensors have shape (n, d), so
                # the streams diverge after the first draw when n differs.
                if reseed_per_step:
                    torch.manual_seed(key_seed("cond", restart, t, j))
                x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
                cond = x0_sample.reshape(1, -1).repeat(n, 1)
                kw = {} if ts is None else {"ts": ts}
                y, _, _ = model_cond.sample(nsamples=n, condition_x=cond,
                                            device=device, **kw)
                cond_calls += n
                L, K_used = guidance_loss(y.double(), K, hierarchy, method,
                                          bandwidth=bandwidth)
                losses.append(-L)
            loss = (-torch.logsumexp(torch.stack(losses), dim=0)
                    + math.log(num_x_t))
        g, = torch.autograd.grad(loss, x_t, allow_unused=True)
        g = torch.zeros_like(x_t) if g is None else g

        # temporal aggregation, cache from the PREVIOUS executed outer step
        if lam > 0.0:
            g_bar = g.detach() if g_bar is None else (1 - lam) * g.detach() + lam * g_bar
            g_used = g_bar
        else:
            g_used = g.detach()

        with torch.no_grad():
            x_t = x_prev.detach().clone() - g_used

        # Divergence guard. The pointwise tier's gradient is ~30x larger than
        # the MMD tiers' (see tier-scale measurement), and with rho = 1 that can
        # run away. A diverged run is RECORDED as diverged, never silently
        # rescaled -- rescaling would be the normalisation ablation, which is a
        # separate experiment.
        xv = float(x_t.reshape(-1)[0])
        if not torch.isfinite(x_t).all() or abs(xv) > 50.0:
            diverged = True
            traj.append({"t": t, "n": n, "K": K_used,
                         "loss": float(loss.detach()),
                         "grad": float(g.reshape(-1)[0]),
                         "grad_used": float(g_used.reshape(-1)[0]),
                         "x": xv, "diverged_here": True})
            break

        traj.append({"t": t, "n": n, "K": K_used,
                     "loss": float(loss.detach()),
                     "grad": float(g.reshape(-1)[0]),
                     "grad_used": float(g_used.reshape(-1)[0]),
                     "x": float(x_t.reshape(-1)[0])})

    x_hat = float(x_t.detach().reshape(-1)[0])
    return x_hat, {"seconds": time.perf_counter() - t0,
                   "conditional_calls": cond_calls,
                   "denoiser_calls": denoiser_calls,
                   "diverged": diverged,
                   "steps_completed": len(traj),
                   "trajectory": traj}


def evaluate(x_hat, params, diverged=False):
    import math
    if diverged or not math.isfinite(x_hat) or abs(x_hat) > 50.0:
        return {"x_hat": x_hat, "L2_squared": float("inf"), "L2": float("inf"),
                "abs_err_vs_xopt": float("inf"), "diverged": True}
    x = torch.tensor([x_hat], dtype=torch.float64)
    l2sq = float(oracle.population_l2_squared(x, params))
    return {"x_hat": x_hat, "L2_squared": l2sq, "L2": l2sq ** 0.5,
            "abs_err_vs_xopt": abs(x_hat - X_OPT), "diverged": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tfg_results/curriculum.json")
    ap.add_argument("--ckpt-dir", default="tfg_results/cm_checkpoints")
    ap.add_argument("--restarts", type=int, default=25)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--kind", default="time")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[SEEDS[0]])
    ap.add_argument("--success-threshold", type=float, default=0.5)
    ap.add_argument("--num-x-t", type=int, default=3)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    params = oracle.load_params()
    S_G = make_fixed_target_set(params).double()
    H = TargetHierarchy(S_G)
    model_uncond, urec = load_uncond(out_dir=args.ckpt_dir)

    payload = {"metadata": {**git_meta(), "torch": torch.__version__,
                            "platform": platform.platform(),
                            "fixed_bandwidth": FIXED_BW, "x_opt": X_OPT,
                            "restarts": args.restarts, "kappa": args.kappa,
                            "p_t_kind": args.kind, "lambda": args.lam,
                            "num_x_t": args.num_x_t,
                            "uncond_record": urec,
                            "target_hierarchy": H.descriptor()},
               "results": []}

    for seed in args.seeds:
        model_cond, crec = load_one(seed, args.ckpt_dir)
        for lam in (0.0, args.lam):
            for method in args.methods:
                if lam > 0 and method in ("pointwise", "full"):
                    continue          # lambda factorial applies to M3/M4 only
                runs = []
                for r in range(args.restarts):
                    xh, info = optimize(model_cond, model_uncond, H, method, r,
                                        kappa=args.kappa, kind=args.kind, lam=lam,
                                        num_x_t=args.num_x_t)
                    ev = evaluate(xh, params, diverged=info["diverged"])
                    runs.append({**ev, "restart": r,
                                 "seconds": info["seconds"],
                                 "conditional_calls": info["conditional_calls"],
                                 "denoiser_calls": info["denoiser_calls"],
                                 "trajectory": info["trajectory"] if r == 0 else None})
                finite = [r for r in runs if not r["diverged"]]
                n_div = len(runs) - len(finite)
                l2 = torch.tensor([r["L2"] for r in finite]) if finite else torch.tensor([float('nan')])
                err = torch.tensor([r["abs_err_vs_xopt"] for r in finite]) if finite else torch.tensor([float('nan')])
                order = sorted(range(len(finite)), key=lambda i: finite[i]["L2"])[:10]
                summary = {
                    "seed": seed, "method": method, "lambda": lam,
                    "L2_mean": float(l2.mean()), "L2_std": float(l2.std(unbiased=True)),
                    "L2_median": float(l2.median()),
                    "L2_top10_mean": (float(torch.tensor([finite[i]["L2"] for i in order]).mean())
                                      if finite else float("nan")),
                    "diverged_runs": n_div,
                    "diverged_fraction": n_div / len(runs),
                    "abs_err_mean": float(err.mean()),
                    "abs_err_median": float(err.median()),
                    "success_rate": (float((err < args.success_threshold).double().sum()) / len(runs)),
                    "conditional_calls_total": sum(r["conditional_calls"] for r in runs),
                    "conditional_calls_per_run": runs[0]["conditional_calls"],
                    "denoiser_calls_per_run": runs[0]["denoiser_calls"],
                    "seconds_total": sum(r["seconds"] for r in runs),
                }
                payload["results"].append({"summary": summary, "runs": runs})
                print(f"  seed={seed} {method:<12} lam={lam:<4} "
                      f"L2 mean={summary['L2_mean']:.4f} top10={summary['L2_top10_mean']:.4f} "
                      f"|x-xopt| med={summary['abs_err_median']:.4f} "
                      f"succ={summary['success_rate']:.0%} div={n_div}/{len(runs)} "
                      f"cond/run={summary['conditional_calls_per_run']:>6} "
                      f"{summary['seconds_total']:.1f}s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
