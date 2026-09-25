"""Fidelity of the retrained conditional consistency models vs the analytic conditional.

For every conditioning value x we compare the model's sample distribution
against the EXACT ``P(Y|X=x)`` obtained in closed form.  Two metrics, both in
1-D where they are cheap and exact:

  * ``W1``   -- exact 1-D Wasserstein-1 between the two empirical samples
                (mean absolute difference of sorted quantiles).
  * ``MMD2`` -- the repository's own RBF kernel, unbiased U-statistic, fixed
                bandwidth.

Crucially each is reported alongside a **finite-sample floor**: the same metric
computed between two INDEPENDENT draws from the analytic conditional itself.
Without that floor a raw W1 of 0.05 is uninterpretable -- it might be entirely
sampling noise.  The ratio metric/floor is the interpretable quantity: 1.0
means the model is indistinguishable from the truth at this sample size.

Note on comparison with the paper: ``CompareMethods-Final.ipynb`` never calls
``evalModels.evaluate_conditional_model``, so the paper reports **no**
conditional-model fidelity number for the 2-D experiment. There is therefore
nothing to compare these values against directly; the analytic conditional and
the finite-sample floor are the reference instead.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tfg._compat import ensure_ot_stub          # noqa: E402

ensure_ot_stub()

import dist_utils                                  # noqa: E402
from tfg import oracle                             # noqa: E402
from tfg.train_cm import SEEDS, load_one           # noqa: E402


def w1_1d(a, b):
    """Exact 1-D Wasserstein-1 between two equal-size empirical samples."""
    a = a.reshape(-1).sort().values
    b = b.reshape(-1).sort().values
    n = min(a.numel(), b.numel())
    return float((a[:n] - b[:n]).abs().mean())


def mmd2_u(a, b, bandwidth=1.0):
    """Unbiased MMD^2 under the repository's RBF kernel."""
    from LossFunctions import RBF
    kernel = RBF(bandwidth=bandwidth, device="cpu")
    a = a.reshape(-1, 1).double()
    b = b.reshape(-1, 1).double()
    n, m = a.shape[0], b.shape[0]
    K = kernel(torch.vstack([a, b]))
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    xx = (Kxx.sum() - Kxx.diagonal().sum()) / (n * (n - 1))
    yy = (Kyy.sum() - Kyy.diagonal().sum()) / (m * (m - 1))
    return float(xx - 2 * Kxy.mean() + yy)


def analytic_samples(x_value, params, n, generator=None):
    x = torch.tensor([x_value], dtype=torch.float64)
    m, c, w = oracle.conditional_params(
        x, params["mu_list"], params["Sigma_list"], params["alpha"])
    return dist_utils.generate_mog_samples_not_differentiable(
        n, m.unsqueeze(-1), c, w).double()


def model_samples(model, x_value, n, device="cpu"):
    cond = torch.full((n, 1), float(x_value), dtype=torch.float32, device=device)
    with torch.no_grad():
        y, _, _ = model.sample(nsamples=n, condition_x=cond, device=device)
    return y.double()


def evaluate(model, params, x_values, n=2000, bandwidth=1.0, seed=0):
    torch.manual_seed(seed)
    rows = []
    for xv in x_values:
        exact_a = analytic_samples(xv, params, n)
        exact_b = analytic_samples(xv, params, n)   # independent -> the floor
        got = model_samples(model, xv, n)

        rows.append({
            "x": float(xv),
            "w1_model": w1_1d(got, exact_a),
            "w1_floor": w1_1d(exact_b, exact_a),
            "mmd2_model": mmd2_u(got, exact_a, bandwidth),
            "mmd2_floor": mmd2_u(exact_b, exact_a, bandwidth),
            "mean_model": float(got.mean()), "mean_exact": float(exact_a.mean()),
            "std_model": float(got.std()), "std_exact": float(exact_a.std()),
        })
    return rows


def summarise(rows):
    def agg(k):
        vals = [r[k] for r in rows]
        return sum(vals) / len(vals)
    w1_m, w1_f = agg("w1_model"), agg("w1_floor")
    mmd_m, mmd_f = agg("mmd2_model"), agg("mmd2_floor")
    return {
        "w1_model_mean": w1_m, "w1_floor_mean": w1_f,
        "w1_ratio": w1_m / w1_f if w1_f > 0 else float("inf"),
        "mmd2_model_mean": mmd_m, "mmd2_floor_mean": mmd_f,
        "mmd2_excess_over_floor": mmd_m - mmd_f,
        "w1_worst": max(r["w1_model"] for r in rows),
        "w1_worst_x": max(rows, key=lambda r: r["w1_model"])["x"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="tfg_results/cm_checkpoints")
    ap.add_argument("--output", default="tfg_results/cm_validation.json")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    args = ap.parse_args()

    params = oracle.load_params()
    # A dense sweep over the region the optimiser actually explores, plus the
    # nominal construction point and the true optimum.
    x_values = [round(v, 4) for v in torch.linspace(-8, 8, 33).tolist()]
    for extra in (-5.0, -4.9984462966235057):
        if extra not in x_values:
            x_values.append(extra)
    x_values.sort()

    out = {"x_values": x_values, "n_samples": args.n,
           "bandwidth": args.bandwidth, "models": []}

    for i, seed in enumerate(SEEDS):
        role = "MAIN" if i == 0 else f"robustness-{i}"
        model, rec = load_one(seed, args.ckpt_dir)
        rows = evaluate(model, params, x_values, n=args.n,
                        bandwidth=args.bandwidth, seed=1000 + i)
        summary = summarise(rows)
        out["models"].append({"seed": seed, "role": role,
                              "training_record": rec,
                              "summary": summary, "per_x": rows})
        print(f"{role:<14} seed={seed}  "
              f"W1 model={summary['w1_model_mean']:.4f} "
              f"floor={summary['w1_floor_mean']:.4f} "
              f"ratio={summary['w1_ratio']:.2f}   "
              f"MMD2 excess={summary['mmd2_excess_over_floor']:+.5f}   "
              f"worst W1={summary['w1_worst']:.4f} at x={summary['w1_worst_x']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
