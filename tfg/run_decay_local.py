"""EXP-016 -- local decay evaluation on the fixed-target adaptive-n_t method.

Compares, per (state pair, n_t): no reuse; constant lambda; the simplified
oracle rule; the generalised oracle rule; and the DEPLOYABLE estimate that uses
only sampled gradients (half-sample split; zero extra conditional samples).
"""
import argparse, json, sys
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tfg._compat import ensure_ot_stub          # noqa: E402
ensure_ot_stub()

from LossFunctions import MMDLoss, RBF          # noqa: E402
from tfg import oracle                          # noqa: E402
from tfg.lambda_deployable import DeployableLambda, oracle_lambda_simplified  # noqa: E402
from tfg.lambda_theory import lambda_general    # noqa: E402
from tfg.run_estimator_v2 import (              # noqa: E402
    key_seed, make_fixed_target_set, reference,
)
from tfg.train_cm import PAPER_TS, SEEDS, load_one   # noqa: E402

BW = 46.550137
CONSTANTS = [0.1, 0.25, 0.5, 0.75, 0.9]


def grad_with_split(model, x_value, n, S_G, seed):
    """Full-sample gradient plus the two half-sample gradients (same draws)."""
    torch.manual_seed(seed)
    x = torch.tensor([float(x_value)], dtype=torch.float32, requires_grad=True)
    cond = x.reshape(1, -1).repeat(n, 1)
    y, _, _ = model.sample(nsamples=n, condition_x=cond, device="cpu", ts=PAPER_TS)
    mmd = MMDLoss(kernel=RBF(bandwidth=BW, device="cpu"), device="cpu")

    L = mmd(y, S_G)
    g_full, = torch.autograd.grad(L, x, retain_graph=True)
    if n < 2:
        return float(g_full), None, None
    h = n // 2
    La = mmd(y[:h], S_G)
    ga, = torch.autograd.grad(La, x, retain_graph=True)
    Lb = mmd(y[h:2 * h], S_G)
    gb, = torch.autograd.grad(Lb, x)
    return float(g_full), float(ga), float(gb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tfg_results/decay_local.json")
    ap.add_argument("--reps", type=int, default=1500)
    ap.add_argument("--ref-reps", type=int, default=300)
    a = ap.parse_args()

    p = oracle.load_params()
    S_G = make_fixed_target_set(p)
    model, _ = load_one(SEEDS[0], "tfg_results/cm_checkpoints")

    pairs = [("low", -6.0, -5.98), ("medium", -6.0, -5.75), ("high", -6.0, -5.0)]
    rows = []
    for label, xt, xn in pairs:
        rt = reference(model, xt, S_G, BW, a.ref_reps, "dl-t")
        rn = reference(model, xn, S_G, BW, a.ref_reps, "dl-n")
        g_t, g_n = rt["g_ref"], rn["g_ref"]
        D2_true = (g_t - g_n) ** 2
        for n in (4, 16, 32, 64, 128, 250):
            GT, GA, GB, GN = [], [], [], []
            for r in range(a.reps):
                gf, ga, gb = grad_with_split(model, xt, n, S_G, key_seed("dl", label, n, r, "t"))
                gn_, _, _ = grad_with_split(model, xn, n, S_G, key_seed("dl", label, n, r, "n"))
                GT.append(gf); GA.append(ga); GB.append(gb); GN.append(gn_)
            gt = torch.tensor(GT, dtype=torch.float64).reshape(-1, 1)
            gn = torch.tensor(GN, dtype=torch.float64).reshape(-1, 1)
            ga = torch.tensor(GA, dtype=torch.float64).reshape(-1, 1)
            gb = torch.tensor(GB, dtype=torch.float64).reshape(-1, 1)

            V_t = float(gt.var(unbiased=True)); V_n = float(gn.var(unbiased=True))
            lam_simpl = oracle_lambda_simplified(V_t, V_n, D2_true)
            lam_gen, _ = lambda_general(gt - g_t, gn - g_t)

            # deployable, per replication, then averaged
            dep = []
            for i in range(a.reps):
                Vh_t = DeployableLambda.split_variance(ga[i], gb[i])
                Vh_n = Vh_t                      # same n at both steps here
                raw = float((gt[i] - gn[i]) ** 2)
                D2h = max(0.0, raw - Vh_t - Vh_n)
                d = Vh_t + Vh_n + D2h
                dep.append(0.0 if d <= 0 else min(0.95, Vh_t / d))
            dep_t = torch.tensor(dep, dtype=torch.float64)
            lam_dep_mean = float(dep_t.mean())

            def mse(l):
                if torch.is_tensor(l):
                    agg = (1 - l).reshape(-1, 1) * gt + l.reshape(-1, 1) * gn
                else:
                    agg = (1 - l) * gt + l * gn
                return float((agg - g_t).pow(2).sum(1).mean())

            grid = torch.linspace(0, 1, 501, dtype=torch.float64)
            mses = torch.tensor([mse(float(l)) for l in grid])
            lam_emp = float(grid[int(mses.argmin())])

            row = {"pair": label, "x_t": xt, "x_next": xn, "n": n, "reps": a.reps,
                   "g_ref_t": g_t, "g_ref_next": g_n, "D2_true": D2_true,
                   "V_t": V_t, "V_next": V_n,
                   "lambda_empirical": lam_emp, "lambda_simplified": lam_simpl,
                   "lambda_general": lam_gen,
                   "lambda_deployable_mean": lam_dep_mean,
                   "lambda_deployable_median": float(dep_t.median()),
                   "lambda_deployable_iqr": float(dep_t.quantile(.75) - dep_t.quantile(.25)),
                   "mse_no_reuse": mse(0.0), "mse_at_empirical": float(mses.min()),
                   "mse_at_simplified": mse(lam_simpl),
                   "mse_at_general": mse(lam_gen),
                   "mse_at_deployable_percall": mse(dep_t),
                   "mse_best_constant": min(mse(c) for c in CONSTANTS),
                   "best_constant": min(CONSTANTS, key=lambda c: mse(c)),
                   "extra_conditional_samples": 0,
                   "extra_kernel_evals_per_step": 2}
            row["gain_deployable_vs_no_reuse"] = 1 - row["mse_at_deployable_percall"] / row["mse_no_reuse"]
            row["gain_deployable_vs_best_const"] = 1 - row["mse_at_deployable_percall"] / row["mse_best_constant"]
            rows.append(row)
            print(f"  {label:<7} n={n:<3} lam: emp={lam_emp:.3f} simpl={lam_simpl:.3f} "
                  f"gen={lam_gen:.3f} dep={lam_dep_mean:.3f}(iqr {row['lambda_deployable_iqr']:.3f})  "
                  f"MSE: none={row['mse_no_reuse']:.4f} dep={row['mse_at_deployable_percall']:.4f} "
                  f"bestconst={row['mse_best_constant']:.4f}(l={row['best_constant']}) "
                  f"| dep vs none {row['gain_deployable_vs_no_reuse']:+.1%}, vs best-const {row['gain_deployable_vs_best_const']:+.1%}")
    Path(a.output).write_text(json.dumps(rows, indent=2))
    print("written", a.output)


if __name__ == "__main__":
    main()
