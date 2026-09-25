"""Reproduce Appendix Table 6 of "Inverse Design for Conditional Distribution Matching".

Protocol, quoted from Appendix A.5:

    "For Neval = 500 conditioning points x drawn i.i.d. from the analytic joint
     distribution P(X, Y), we draw n = 500 samples from each trained model
     given that x, and compare them against 500 analytic samples from the true
     conditional via MMD."

Reported for the 2D setting (Table 6):

    Diffusion (conditional)   MMD mean 0.011   std 0.014
    Consistency Model (CM)    MMD mean 0.163   std 0.214

Only the CM is reproduced here; the conditional diffusion model has not been
retrained (it is not needed until end-to-end optimisation).

Two ambiguities in the protocol, both reported rather than silently resolved:

  * The paper does not state which MMD estimator or bandwidth rule. We use the
    repository's ``MMDLoss(RBF())`` with its data-adaptive bandwidth, since
    that is what the codebase uses everywhere, and additionally report a
    fixed-bandwidth variant.
  * The paper states sampling noise levels [150, 50, 20, 10, 5, 1], which is
    the default ``ts`` of ``ConsistencyModel.sample`` -- NOT of
    ``ConsistencyModeliCT.sample``, whose default is a 15-entry Karras ladder.
    Both are evaluated.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tfg._compat import ensure_ot_stub      # noqa: E402

ensure_ot_stub()

import dist_utils                            # noqa: E402
from LossFunctions import MMDLoss, RBF       # noqa: E402
from tfg import oracle                       # noqa: E402
from tfg.train_cm import PAPER_TS, SEEDS, load_one   # noqa: E402

PAPER_2D = {"cm": {"mean": 0.163, "std": 0.214},
            "diffusion_conditional": {"mean": 0.011, "std": 0.014}}


def joint_conditioning_points(params, n_eval, seed):
    """x drawn i.i.d. from the analytic joint P(X, Y), keeping the X coordinate."""
    torch.manual_seed(seed)
    joint = dist_utils.generate_mog_samples_not_differentiable(
        n_eval, [m.float() for m in params["mu_list"]],
        [s.float() for s in params["Sigma_list"]], params["alpha"].float())
    return joint[:, 0]


def evaluate(model, params, n_eval=500, n=500, seed=0, bandwidth=None, ts=None):
    xs = joint_conditioning_points(params, n_eval, seed)
    mmd = MMDLoss(kernel=RBF(bandwidth=bandwidth, device="cpu"), device="cpu")
    vals = []
    for xv in xs.tolist():
        x = torch.tensor([xv], dtype=torch.float64)
        m, c, w = oracle.conditional_params(
            x, params["mu_list"], params["Sigma_list"], params["alpha"])
        exact = dist_utils.generate_mog_samples_not_differentiable(
            n, m.unsqueeze(-1).float(), c.float(), w.float())
        cond = torch.full((n, 1), float(xv), dtype=torch.float32)
        with torch.no_grad():
            kw = {} if ts is None else {"ts": ts}
            y, _, _ = model.sample(nsamples=n, condition_x=cond,
                                   device="cpu", **kw)
        vals.append(float(mmd(y, exact)))
    t = torch.tensor(vals, dtype=torch.float64)
    return {"mean": float(t.mean()), "std": float(t.std(unbiased=True)),
            "median": float(t.median()),
            "q01": float(t.quantile(0.01)), "q99": float(t.quantile(0.99)),
            "n_eval": n_eval, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="tfg_results/cm_checkpoints")
    ap.add_argument("--output", default="tfg_results/table6_reproduction.json")
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    params = oracle.load_params()
    out = {"paper_2d": PAPER_2D, "protocol": "Appendix A.5", "models": []}

    print(f"paper 2D CM: mean={PAPER_2D['cm']['mean']}  "
          f"std={PAPER_2D['cm']['std']}\n")
    for i, seed in enumerate(SEEDS):
        role = "MAIN" if i == 0 else f"robustness-{i}"
        model, rec = load_one(seed, args.ckpt_dir)
        entry = {"seed": seed, "role": role, "variants": {}}
        for label, bw, ts in (
                ("adaptive_bw / paper ts", None, PAPER_TS),
                ("adaptive_bw / code default ts", None, None),
                ("fixed_bw=1 / paper ts", 1.0, PAPER_TS)):
            r = evaluate(model, params, n_eval=args.n_eval, n=args.n,
                         seed=1234 + i, bandwidth=bw, ts=ts)
            entry["variants"][label] = r
            print(f"  {role:<14} {label:<30} "
                  f"mean={r['mean']:.4f}  std={r['std']:.4f}  "
                  f"median={r['median']:.4f}")
        out["models"].append(entry)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
