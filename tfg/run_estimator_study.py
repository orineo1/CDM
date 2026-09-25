"""Estimator-level study on the analytic 2-D oracle. No training, no cluster.

Produces, all against the EXACT population gradient as ground truth:

  1. x_opt: the true minimiser of the exact GMM L2 objective, by bisection on
     dL/dx = 0 to machine precision, plus the irreducible loss floor.
  2. Validation of the corrected sampled estimator (pathwise + categorical
     score) on the diagonal-covariance isolation case.
  3. Bias / variance / MSE by n_t, with Monte-Carlo standard errors and a
     replication-count stability check.
  4. The lambda-bias curve for temporal aggregation, with the empirically
     optimal lambda compared against the value predicted from inter-step drift
     D_t^2 and within-step sampling variance v_t, at both a low-drift and a
     high-drift pair of consecutive steps.

Usage:
    python -m tfg.run_estimator_study --output tfg_results/estimator_study.json
"""

import argparse
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

from tfg import estimators as E          # noqa: E402
from tfg import oracle                    # noqa: E402

N_GRID = [1, 2, 4, 8, 16, 32, 64]
SIGMA = 1.0


def git_metadata():
    def run(*a):
        try:
            return subprocess.run(a, cwd=str(_ROOT), capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("git", "status", "--porcelain")}


# ---------------------------------------------------------------------------
# 1. x_opt by bisection on the exact gradient
# ---------------------------------------------------------------------------

def find_x_opt(params, lo=-6.0, hi=-4.0, iters=200):
    """Solve dL/dx = 0 on [lo, hi] by bisection, to float64 resolution.

    Bisection rather than a grid: the grid search reported earlier resolved
    x_opt only to its step size. Bisection on the sign of the exact gradient
    converges to the last representable double.
    """
    def grad(v):
        _, g = oracle.population_grad(torch.tensor([v], dtype=torch.float64), params)
        return float(g[0])

    glo, ghi = grad(lo), grad(hi)
    if glo * ghi > 0:
        raise RuntimeError(
            f"gradient does not change sign on [{lo}, {hi}]: "
            f"g({lo})={glo:.3e}, g({hi})={ghi:.3e}"
        )
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        gmid = grad(mid)
        if gmid == 0.0 or (hi - lo) < 1e-16:
            lo = hi = mid
            break
        if glo * gmid < 0:
            hi, ghi = mid, gmid
        else:
            lo, glo = mid, gmid
    x_opt = 0.5 * (lo + hi)
    xt = torch.tensor([x_opt], dtype=torch.float64)
    val, g = oracle.population_grad(xt, params)
    nominal = torch.tensor([-5.0], dtype=torch.float64)
    return {
        "x_opt": x_opt,
        "L_at_x_opt": float(val),
        "grad_at_x_opt": float(g[0]),
        "bracket_width": hi - lo,
        "x_nominal": -5.0,
        "L_at_nominal": float(oracle.population_l2_squared(nominal, params)),
        "offset_from_nominal": x_opt - (-5.0),
    }


# ---------------------------------------------------------------------------
# 2. Corrected-estimator validation on the isolation case
# ---------------------------------------------------------------------------

def isolation_params():
    """Diagonal joint covariance: conditional means/covs are constant in x, so
    the categorical score term is the ONLY source of gradient."""
    return dict(
        mu_list=[torch.tensor([-3.0, 2.0], dtype=torch.float64),
                 torch.tensor([3.0, -2.0], dtype=torch.float64)],
        Sigma_list=[torch.eye(2, dtype=torch.float64) * 0.5,
                    torch.eye(2, dtype=torch.float64) * 0.5],
        alpha=torch.tensor([0.5, 0.5], dtype=torch.float64),
        target_means=torch.tensor([[[2.0]]], dtype=torch.float64),
        target_variances=[torch.tensor([[0.5]], dtype=torch.float64)],
        target_weights=torch.tensor([1.0], dtype=torch.float64),
    )


def validate_corrected_estimator(reps_list=(2_000, 8_000, 32_000)):
    iso = isolation_params()
    x = torch.tensor([0.7], dtype=torch.float64)
    _, g_ref = E.exact_gradient(x, iso, sigma=SIGMA)
    out = {"g_exact": float(g_ref[0]), "runs": []}
    for est in E.ESTIMATORS:
        for reps in reps_list:
            _, grads = E.replicate(x, iso, n=8, reps=reps, seed=1,
                                   sigma=SIGMA, estimator=est)
            mean = float(grads.mean(0)[0])
            se = float(grads.std(0)[0] / reps ** 0.5)
            z = abs(mean - float(g_ref[0])) / se if se > 0 else float("inf")
            out["runs"].append({
                "estimator": est, "reps": reps, "mean": mean, "se": se,
                "z_vs_exact": z,
                "fraction_recovered": mean / float(g_ref[0]),
            })
    return out


# ---------------------------------------------------------------------------
# 3. Bias / variance / MSE by n
# ---------------------------------------------------------------------------

def sweep_by_n(params, x_value, reps, seed=0, unbiased_xx=True,
               estimator="pathwise_plus_score"):
    x = torch.tensor([x_value], dtype=torch.float64)
    rows = []
    for n in N_GRID:
        t0 = time.perf_counter()
        st = E.estimator_stats(x, params, n=n, reps=reps, seed=seed,
                               sigma=SIGMA, estimator=estimator,
                               unbiased_xx=unbiased_xx)
        st["seconds"] = time.perf_counter() - t0
        st["x"] = x_value
        st["unbiased_xx"] = unbiased_xx
        st["estimator"] = estimator
        rows.append(st)
    return rows


# ---------------------------------------------------------------------------
# 4. Temporal aggregation: the lambda curve
# ---------------------------------------------------------------------------

def lambda_study(params, x_t, x_next, n, reps, seed=0,
                 lambdas=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9)):
    """MSE of the aggregated gradient  gbar = (1-l) g_t + l g_{t+1}  vs g_t^ref.

    The model behind the plug-in rule: if both estimates are unbiased for their
    own step with per-estimate error variance v, then

        MSE(l) = l^2 D^2 + (1-l)^2 v_t + l^2 v_{t+1}

    whose minimiser is  l* = v / (D^2 + 2v)  when v_t = v_{t+1} = v. We measure
    the curve directly and compare its argmin to that prediction.
    """
    xt = torch.tensor([x_t], dtype=torch.float64)
    xn = torch.tensor([x_next], dtype=torch.float64)
    _, g_t = E.exact_gradient(xt, params, sigma=SIGMA)
    _, g_n = E.exact_gradient(xn, params, sigma=SIGMA)
    D2 = float(((g_t - g_n) ** 2).sum())

    _, gr_t = E.replicate(xt, params, n, reps, seed=seed, sigma=SIGMA)
    _, gr_n = E.replicate(xn, params, n, reps, seed=seed + 10_000, sigma=SIGMA)

    v_t = float(((gr_t - gr_t.mean(0)) ** 2).sum(1).mean())
    v_n = float(((gr_n - gr_n.mean(0)) ** 2).sum(1).mean())
    v = 0.5 * (v_t + v_n)

    curve = []
    for lam in lambdas:
        agg = (1.0 - lam) * gr_t + lam * gr_n
        sq = ((agg - g_t) ** 2).sum(1)
        curve.append({
            "lambda": lam,
            "mse": float(sq.mean()),
            "se_mse": float(sq.std(unbiased=True) / reps ** 0.5),
        })

    # Fine grid for the empirical argmin.
    fine = torch.linspace(0.0, 0.95, 192, dtype=torch.float64)
    mses = []
    for lam in fine:
        agg = (1.0 - lam) * gr_t + lam * gr_n
        mses.append(float(((agg - g_t) ** 2).sum(1).mean()))
    mses_t = torch.tensor(mses)
    lam_emp = float(fine[int(mses_t.argmin())])

    lam_pred = v / (D2 + 2.0 * v) if (D2 + 2.0 * v) > 0 else 0.0

    return {
        "x_t": x_t, "x_next": x_next, "n": n, "reps": reps,
        "D2": D2, "v_t": v_t, "v_next": v_n, "v_mean": v,
        "drift_vs_variance": "low_drift" if D2 < v else "high_drift",
        "D2_lt_v": bool(D2 < v),
        "lambda_empirical": lam_emp,
        "lambda_predicted": lam_pred,
        "mse_at_lambda0": float(((gr_t - g_t) ** 2).sum(1).mean()),
        "mse_at_lambda_emp": float(mses_t.min()),
        "improvement_vs_no_cache": 1.0 - float(mses_t.min()) / float(
            ((gr_t - g_t) ** 2).sum(1).mean()),
        "curve": curve,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tfg_results/estimator_study.json")
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--reps-stability", type=int, default=16000)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    params = oracle.load_params()

    print("== 1. x_opt by bisection on the exact gradient ==")
    xopt = find_x_opt(params)
    for k, v in xopt.items():
        print(f"   {k:24s} {v!r}")

    print("\n== 2. corrected-estimator validation (isolation case) ==")
    val = validate_corrected_estimator()
    print(f"   exact weight gradient = {val['g_exact']:+.8f}")
    for r in val["runs"]:
        print(f"   {r['estimator']:<20} reps={r['reps']:<6} "
              f"mean={r['mean']:+.6f} +- {r['se']:.6f}  "
              f"z={r['z_vs_exact']:.2f}  recovered={r['fraction_recovered']:.1%}")

    print("\n== 3. bias / variance / MSE by n ==")
    x_probe = xopt["x_opt"] + 0.35        # near, but not at, the optimum
    sweeps = {}
    for label, reps in (("main", args.reps), ("stability", args.reps_stability)):
        rows = sweep_by_n(params, x_probe, reps=reps)
        sweeps[label] = rows
        print(f"   -- reps={reps} at x={x_probe:.6f} --")
        print(f"   {'n':>3} {'|bias|':>12} {'se':>10} {'variance':>12} "
              f"{'MSE':>12} {'se':>10} {'rel_err':>9} {'cos':>7}")
        for r in rows:
            print(f"   {r['n']:>3} {r['bias_norm']:>12.4e} {r['se_bias_norm']:>10.2e} "
                  f"{r['variance']:>12.4e} {r['mse']:>12.4e} {r['se_mse']:>10.2e} "
                  f"{r['rel_error']:>9.3f} {r['cosine']:>7.3f}")

    print("\n   -- V-statistic vs U-statistic (gradient normalisation) --")
    v_rows = sweep_by_n(params, x_probe, reps=args.reps, unbiased_xx=False)
    for u, vv in zip(sweeps["main"], v_rows):
        print(f"   n={u['n']:>3}  U |bias|={u['bias_norm']:.4e}   "
              f"V |bias|={vv['bias_norm']:.4e}   "
              f"ratio V/U mean={vv['g_mean'][0] / u['g_mean'][0]:.4f}  "
              f"(predicted (n-1)/n on the xx term)")

    print("\n== 4. temporal aggregation: lambda ==")
    lam_runs = []
    pairs = [
        ("low_drift", x_probe, x_probe + 0.01),
        ("low_drift_finer", x_probe, x_probe + 0.003),
        ("high_drift", x_probe, x_probe + 0.5),
        ("high_drift_extreme", x_probe, x_probe + 1.5),
    ]
    for label, a, b in pairs:
        for n in (4, 32):
            r = lambda_study(params, a, b, n=n, reps=args.reps)
            r["label"] = label
            lam_runs.append(r)
            print(f"   {label:<19} n={n:<3} D2={r['D2']:.4e} v={r['v_mean']:.4e} "
                  f"({r['drift_vs_variance']})  "
                  f"lam_emp={r['lambda_empirical']:.3f} "
                  f"lam_pred={r['lambda_predicted']:.3f}  "
                  f"MSE gain={r['improvement_vs_no_cache']:+.1%}")

    payload = {
        "metadata": {
            **git_metadata(),
            "dtype": "float64", "device": "cpu",
            "torch": torch.__version__, "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sigma_kernel": SIGMA, "n_grid": N_GRID,
            "reps": args.reps, "reps_stability": args.reps_stability,
            "x_probe": x_probe,
        },
        "x_opt": xopt,
        "estimator_validation": val,
        "sweep_by_n": sweeps,
        "sweep_v_statistic": v_rows,
        "lambda_study": lam_runs,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
