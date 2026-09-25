"""EXP-019 -- LGD spatial smoothing vs temporal momentum, at matched budget.

Central question: legacy LGD spends ``M_t = 3`` spatial perturbations, each costing
``n_{t,j}`` conditional generations, so the conditional budget is
``B_t = M_t * n_{t,j} = 750`` per diffusion step. Can that 3x spatial spend be
re-allocated to MMD estimation (``M_t = 1``, ``n_{t,1} = B_t``) and the lost
stabilisation recovered by temporal momentum instead?

Spatial allocations (all share the same per-step budget symbol ``B_t``):

  A ``no_lgd``       M_t = 1, n_{t,1} = B_t                cost  B_t
  B ``lgd_matched``  M_t = 3, sum_j n_{t,j} = B_t          cost  B_t
  C ``lgd_high``     M_t = 3, n_{t,j} = B_t                cost  3 * B_t

B splits the budget with deterministic integer remainders: the first
``B_t mod M_t`` perturbations receive one extra sample. No randomness.

Aggregation for M_t > 1 is the TFG/LGD form, exactly as ``Optimization.py:132``:

    Ltilde_t = -log( (1/M_t) sum_j exp( -beta * MMD^2_j ) )

with ``beta = 1``. Outputs from different perturbations are NEVER pooled into a
single MMD -- that would be a different method.

Temporal choices:

  ``none``    use the raw gradient
  ``adam``    AdamDPS (arXiv:2603.16797v2), see tfg/adam_guidance.py
  ``lambda``  the deployable adaptive lambda_t of tfg/lambda_deployable.py,
              optionally smoothed (``lambda_smooth``)

The half-sample split used by the lambda rule costs **zero extra conditional
generations**: it re-uses the samples already drawn, splitting each
perturbation's ``n_{t,j}`` draws into halves and re-running the *same*
aggregation on each half. It costs 2 extra kernel evaluations and 2 extra
backward passes per step, which are counted and reported.
"""

import argparse
import hashlib
import json
import math
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

from LossFunctions import MMDLoss, RBF           # noqa: E402
from tfg import oracle                           # noqa: E402
from tfg.adam_guidance import AdamGuidance       # noqa: E402
from tfg.lambda_deployable import DeployableLambda   # noqa: E402
from tfg.n_schedule import n_at                           # noqa: E402
from tfg.run_estimator_v2 import make_fixed_target_set   # noqa: E402
from tfg.schedule import DiffusionSchedule               # noqa: E402
from tfg.train_cm import PAPER_TS, SEEDS, load_one       # noqa: E402
from tfg.train_uncond import load as load_uncond          # noqa: E402

FIXED_BW = 46.550137
X_OPT = -4.9984462966235057
N_MAX = 16                # practical default; primary settings are 8 and 16
PRACTICAL_NMAX = (4, 8, 16, 32)   # 4 = cheap/noisy, 8 & 16 primary, 32 upper
B_T = 750                 # DEPRECATED (PILOT): the n=750 regime is out of scope
M_LGD = 3
SPATIAL = ("no_lgd", "lgd_matched", "lgd_high")
TEMPORAL = ("none", "adam", "lambda", "const")


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


def n_t_at(t, sched, protocol, kappa=1.0, n_max=N_MAX, kind="time"):
    """Per-step conditional sample count, capped at ``n_max``.

    ``constant``: n_t = n_max (the CDM paper's setting).
    ``adaptive``: n_t = 1 + floor((n_max-1) * p_t^kappa), the validated schedule.
    """
    if protocol == "constant":
        return int(n_max)
    if protocol == "adaptive":
        return int(n_at(t, sched, n_max, kappa, kind))
    raise ValueError(f"unknown n_t protocol {protocol!r}")


def allocate(B, spatial):
    """Split ``n_t`` across the spatial perturbations.

    ``B`` is now the PER-MMD sample count ``n_t`` (<= 250), not a total budget:

      A no_lgd       -> [n_t]              cost  n_t
      B lgd_matched  -> deterministic split summing to n_t,  cost  n_t
      C lgd_high     -> [n_t] * M_LGD      cost  3 * n_t   (each MMD still <= n_t)
    """
    if spatial == "no_lgd":
        return [int(B)]
    if spatial == "lgd_matched":
        # M_t must not exceed n_t: splitting n_t < 3 across 3 perturbations
        # would give a perturbation ZERO samples, i.e. an MMD on an empty set
        # (NaN). Under the adaptive protocol n_t starts at 1, so this is
        # reachable and was hit in the first practical factorial run.
        M_eff = min(M_LGD, int(B))
        base, rem = divmod(int(B), M_eff)
        return [base + (1 if j < rem else 0) for j in range(M_eff)]   # deterministic
    if spatial == "lgd_high":
        return [int(B)] * M_LGD
    raise ValueError(f"unknown spatial mode {spatial!r}")


def _mmd(y, S_G, bandwidth):
    return MMDLoss(kernel=RBF(bandwidth=bandwidth, device="cpu"), device="cpu")(y, S_G)


def step_loss(model_cond, pred_x0, r_t, alloc, S_G, bandwidth, seed_base,
              want_split=False, device="cpu"):
    """Aggregated LGD/TFG loss and (optionally) two half-sample variants.

    Returns ``(loss, loss_half_a, loss_half_b, n_conditional_drawn)``.
    """
    M = len(alloc)
    terms, terms_a, terms_b = [], [], []
    drawn = 0
    for j, n_j in enumerate(alloc):
        torch.manual_seed(key_seed(seed_base, "pert", j))
        delta = torch.randn_like(pred_x0)
        x_j = pred_x0 + r_t * delta                     # LGD spatial perturbation
        cond = x_j.reshape(1, -1).repeat(n_j, 1)
        y, _, _ = model_cond.sample(nsamples=n_j, condition_x=cond,
                                    device=device, ts=PAPER_TS)
        drawn += n_j
        terms.append(-_mmd(y, S_G, bandwidth))
        if want_split and n_j >= 2:
            h = n_j // 2
            terms_a.append(-_mmd(y[:h], S_G, bandwidth))
            terms_b.append(-_mmd(y[h:2 * h], S_G, bandwidth))

    def agg(ts):
        return -torch.logsumexp(torch.stack(ts), dim=0) + math.log(len(ts))

    loss = agg(terms)
    la = agg(terms_a) if len(terms_a) == M else None
    lb = agg(terms_b) if len(terms_b) == M else None
    return loss, la, lb, drawn


def optimize(model_cond, model_uncond, S_G, spatial, temporal, restart,
             B=None, n_protocol="constant", kappa=1.0, n_max=N_MAX,
             bandwidth=FIXED_BW, adam_rho=0.01, beta1=0.9, beta2=0.995,
             lam_max=0.95, lam_smooth=0.0, lam_const=0.5,
             device="cpu", trace_out=False):
    """One end-to-end run under a (spatial, temporal) configuration."""
    T = model_uncond.diffusion_steps
    sched = DiffusionSchedule(T=T)
    x_t = torch.zeros(1, model_uncond.nfeatures, device=device)

    adam = AdamGuidance(beta1=beta1, beta2=beta2, rho=adam_rho) if temporal == "adam" else None
    dep = DeployableLambda(lambda_max=lam_max) if temporal == "lambda" else None
    prev_used = None
    lam_ema = None

    cond_calls = 0
    kernel_evals = 0
    extra_kernel = 0
    diverged = False
    traj = []
    t0 = time.perf_counter()

    for t in range(T - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)
        x_prev, pred_x0 = model_uncond.sample_ddim_step(
            x_t, t, condition_x=None, device=device, eta=0.0)
        cur_var = model_uncond.betas[t].to(device)
        r_t = cur_var / torch.sqrt(1 + cur_var ** 2)

        n_t = B if B is not None else n_t_at(t, sched, n_protocol, kappa, n_max)
        if n_t > n_max:
            raise ValueError(f"n_t={n_t} exceeds the {n_max}-sample cap")
        alloc = allocate(n_t, spatial)
        if max(alloc) > n_max:
            raise ValueError(f"a single MMD would use {max(alloc)} > {n_max} samples")

        want_split = (temporal == "lambda")
        loss, la, lb, drawn = step_loss(
            model_cond, pred_x0, r_t, alloc, S_G, bandwidth,
            seed_base=key_seed("run", restart, t), want_split=want_split)
        cond_calls += drawn
        kernel_evals += len(alloc)

        g, = torch.autograd.grad(loss, x_t, retain_graph=want_split,
                                 allow_unused=True)
        g = torch.zeros_like(x_t) if g is None else g

        if temporal == "none":
            g_used = g.detach()
        elif temporal == "adam":
            # Upstream divides the guidance by sqrt(alpha_t) with
            # alpha_t = alphabar_t / alphabar_{t-1} (adam_dps.py).
            ab = model_uncond.baralphas
            alpha_t = float(ab[t] / ab[t - 1]) if t >= 1 else 1.0
            g_used = adam.step(g, alpha_t=alpha_t)
        elif temporal == "const":
            # Constant temporal mixing: gbar_t = (1-l) g_t + l gbar_{t+1}.
            g_used = g.detach() if prev_used is None else (
                (1 - lam_const) * g.detach() + lam_const * prev_used)
            traj_lam = lam_const
        else:
            ga = gb = None
            if la is not None and lb is not None:
                ga, = torch.autograd.grad(la, x_t, retain_graph=True, allow_unused=True)
                gb, = torch.autograd.grad(lb, x_t, allow_unused=True)
                extra_kernel += 2 * len(alloc)
            lam, diag = dep.update(g.detach(), ga, gb)
            if lam_smooth > 0.0:
                lam_ema = lam if lam_ema is None else (
                    (1 - lam_smooth) * lam + lam_smooth * lam_ema)
                lam = lam_ema
            g_used = g.detach() if prev_used is None else (
                (1 - lam) * g.detach() + lam * prev_used)
            traj_lam = lam

        prev_used = g_used.detach().clone()
        with torch.no_grad():
            x_t = x_prev.detach().clone() - g_used

        xv = float(x_t.reshape(-1)[0])
        rec = {"t": t, "n_t": n_t, "alloc": alloc, "loss": float(loss.detach()),
               "grad": float(g.reshape(-1)[0]),
               "grad_used": float(g_used.reshape(-1)[0]), "x": xv}
        if temporal in ("lambda", "const"):
            rec["lambda"] = traj_lam
        if trace_out:
            traj.append(rec)
        if not torch.isfinite(x_t).all() or abs(xv) > 50.0:
            diverged = True
            if not trace_out:
                traj.append(rec)
            break

    return float(x_t.detach().reshape(-1)[0]), {
        "seconds": time.perf_counter() - t0,
        "conditional_calls": cond_calls,
        "kernel_evals": kernel_evals,
        "extra_kernel_evals": extra_kernel,
        "extra_conditional_samples": 0,
        "denoiser_calls": T - 1,
        "diverged": diverged,
        "trajectory": traj,
        "adam_state": None if adam is None else adam.state(),
        "lambda_overhead": None if dep is None else dep.overhead(),
    }


def evaluate(x_hat, params, diverged=False):
    if diverged or not math.isfinite(x_hat) or abs(x_hat) > 50.0:
        return {"x_hat": x_hat, "L2": float("inf"), "L2_squared": float("inf"),
                "abs_err_vs_xopt": float("inf"), "diverged": True}
    l2sq = float(oracle.population_l2_squared(
        torch.tensor([x_hat], dtype=torch.float64), params))
    return {"x_hat": x_hat, "L2_squared": l2sq, "L2": l2sq ** 0.5,
            "abs_err_vs_xopt": abs(x_hat - X_OPT), "diverged": False}


def run_arm(model_cond, model_uncond, S_G, params, spatial, temporal, restarts,
            restart_offset=0, **kw):
    """``restart_offset`` selects a disjoint block of restart seeds, so tuning
    (offset 0) and held-out confirmation (offset 1000) never share randomness."""
    runs = []
    for r in range(restart_offset, restart_offset + restarts):
        xh, info = optimize(model_cond, model_uncond, S_G, spatial, temporal, r, **kw)
        ev = evaluate(xh, params, info["diverged"])
        runs.append({**ev, "restart": r, "seconds": info["seconds"],
                     "conditional_calls": info["conditional_calls"],
                     "kernel_evals": info["kernel_evals"],
                     "extra_kernel_evals": info["extra_kernel_evals"],
                     "trajectory": info["trajectory"] if r == 0 else None,
                     "lambda_overhead": info["lambda_overhead"]})
    fin = [x for x in runs if not x["diverged"]]
    l2 = torch.tensor([x["L2"] for x in fin]) if fin else torch.tensor([float("nan")])
    err = torch.tensor([x["abs_err_vs_xopt"] for x in fin]) if fin else torch.tensor([float("nan")])
    summary = {
        "spatial": spatial, "temporal": temporal, "restarts": restarts,
        "L2_mean": float(l2.mean()), "L2_median": float(l2.median()),
        "L2_std": float(l2.std(unbiased=True)) if len(fin) > 1 else float("nan"),
        "abs_err_median": float(err.median()),
        "success_rate": float((err < 0.5).double().sum()) / len(runs),
        "diverged": len(runs) - len(fin),
        # Averaged over COMPLETED runs. Using runs[0] under-reports whenever
        # that restart diverges early and stops drawing samples.
        "conditional_calls_per_run": (sum(x["conditional_calls"] for x in fin) / len(fin)
                                      if fin else float("nan")),
        "conditional_calls_all_runs": sum(x["conditional_calls"] for x in runs) / len(runs),
        "kernel_evals_per_run": (sum(x["kernel_evals"] for x in fin) / len(fin)
                                 if fin else float("nan")),
        "extra_kernel_evals_per_run": (sum(x["extra_kernel_evals"] for x in fin) / len(fin)
                                       if fin else float("nan")),
        "restart_offset": restart_offset,
        "seconds_total": sum(x["seconds"] for x in runs),
        **{k: v for k, v in kw.items() if isinstance(v, (int, float, str))},
    }
    return {"summary": summary, "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "confirm"], default="tune")
    ap.add_argument("--restarts", type=int, default=50)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ckpt-dir", default="tfg_results/cm_checkpoints")
    ap.add_argument("--adam-rho", type=float, default=0.01)
    ap.add_argument("--lam-smooth", type=float, default=0.0)
    ap.add_argument("--lam-const", type=float, default=0.5)
    ap.add_argument("--n-protocol", choices=["constant", "adaptive"], default="constant")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--adam-beta2", type=float, default=0.995)
    ap.add_argument("--spatial", nargs="*", default=list(SPATIAL))
    ap.add_argument("--temporal", nargs="*", default=list(TEMPORAL))
    a = ap.parse_args()

    params = oracle.load_params()
    S_G = make_fixed_target_set(params)
    mc, crec = load_one(SEEDS[0], a.ckpt_dir)
    mu, urec = load_uncond(out_dir=a.ckpt_dir)

    payload = {"metadata": {**git_meta(), "stage": a.stage,
                            "torch": torch.__version__,
                            "platform": platform.platform(),
                            "n_protocol": a.n_protocol, "kappa": a.kappa,
                            "n_max": a.n_max, "M_LGD": M_LGD,
                            "fixed_bandwidth": FIXED_BW, "x_opt": X_OPT,
                            "adam_rho": a.adam_rho, "lam_smooth": a.lam_smooth,
                            "allocations_at_n_max": {s: allocate(a.n_max, s) for s in SPATIAL},
                            "cm_record": crec, "uncond_record": urec},
               "arms": []}
    print(f"stage={a.stage} restarts={a.restarts} n_protocol={a.n_protocol} "
          f"n_max={a.n_max} kappa={a.kappa}")
    for s in ("no_lgd", "lgd_matched", "lgd_high"):
        if s not in a.spatial:
            continue
        for tmp in ("none", "adam", "lambda", "const"):
            if tmp not in a.temporal:
                continue
            arm = run_arm(mc, mu, S_G, params, s, tmp, a.restarts,
                          n_protocol=a.n_protocol, kappa=a.kappa, n_max=a.n_max,
                          adam_rho=a.adam_rho,
                          lam_smooth=a.lam_smooth, lam_const=a.lam_const)
            payload["arms"].append(arm)
            m = arm["summary"]
            print(f"  {s:<12} {tmp:<7} L2 mean={m['L2_mean']:.4f} med={m['L2_median']:.4f} "
                  f"succ={m['success_rate']:.0%} div={m['diverged']}/{a.restarts} "
                  f"cond/run={m['conditional_calls_per_run']:>7} "
                  f"kern/run={m['kernel_evals_per_run']:>5}+{m['extra_kernel_evals_per_run']:<5} "
                  f"{m['seconds_total']:.0f}s", flush=True)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(payload, indent=2))
    print("written", a.output)


if __name__ == "__main__":
    main()
