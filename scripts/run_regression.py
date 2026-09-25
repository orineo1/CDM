#!/usr/bin/env python
"""Tier 3: does the refactored code still reproduce the published numbers?

Tiers 0-2 prove the refactor did not change the code's behaviour. That is a
different claim from "the code's behaviour still matches what was published"
-- and given that the synthetic table in CompareMethods-Final.ipynb consists
of hand-pasted literals, the second claim is worth establishing independently.

Usage
-----
    # 1. On the pre-refactor code (or with --legacy), create the baseline:
    python scripts/run_regression.py --update-golden

    # 2. After the refactor, check against it:
    python scripts/run_regression.py

    # Compare the two code paths directly in one process:
    python scripts/run_regression.py --compare

Results are appended to results/runs.jsonl via gcio.results, so the table can
be regenerated from data instead of pasted.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcio.results import save_run  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "tests", "golden", "synthetic_2d.json")

# Tolerance for the seed-averaged final loss. Deliberately loose: this guards
# against a refactor moving the numbers materially, not against float noise.
REL_TOL = 0.05


def build_models(device, seed, nunits=128, diffusion_steps=100):
    from models.diffusion import DiffusionModel
    from models.samplers.consistency import ConsistencyModel

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_uncond = DiffusionModel(nfeatures=1, nunits=nunits, condition=False).to(device)
    model_uncond.diffusion_steps = diffusion_steps
    model_cond = ConsistencyModel(nfeatures=1, condition_on=1,
                                  nunits=nunits, depth=6).to(device)
    return model_uncond.eval(), model_cond.eval()


def run_seeds(optimize_fn, device, n_seeds, n_cond, n_mc, label):
    """Run the given optimizer over n_seeds and return per-seed final losses."""
    # Toy example from the paper's appendix: 2-component GMM, equal weights.
    mog_means = [torch.tensor([-3.0]), torch.tensor([3.0])]
    mog_variances = [torch.tensor([[0.5]]), torch.tensor([[0.5]])]
    mog_weights = torch.tensor([0.5, 0.5])

    losses, xs, times = [], [], []
    for seed in range(n_seeds):
        model_uncond, model_cond = build_models(device, seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        t0 = time.time()
        out = optimize_fn(
            model_uncond, model_cond, mog_means, mog_variances, mog_weights,
            None, None, None,
            nsamples=n_cond, num_x_t=n_mc, loss="MMD", CM=True,
            device=device, FLAG=False,
        )
        dt = time.time() - t0

        x_final, _, final_loss = out
        loss_v = float(final_loss.detach().cpu().reshape(-1)[0]) \
            if isinstance(final_loss, torch.Tensor) else float(final_loss)
        x_v = x_final.detach().cpu().reshape(-1).tolist()

        losses.append(loss_v)
        xs.append(x_v)
        times.append(dt)

        save_run(
            experiment="synthetic_2d_regression",
            config={"label": label, "n_cond": n_cond, "n_mc": n_mc,
                    "device": device},
            metrics={"final_loss": loss_v, "x_final": x_v, "time_s": dt},
            seed=seed,
        )
        print(f"  [{label}] seed {seed:2d}  loss={loss_v:.6f}  "
              f"x={x_v}  {dt:.1f}s")

    return {"losses": losses, "xs": xs, "times": times,
            "mean_loss": sum(losses) / len(losses)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-cond", type=int, default=250)
    ap.add_argument("--n-mc", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--update-golden", action="store_true",
                    help="write the current results as the new baseline")
    ap.add_argument("--legacy", action="store_true",
                    help="run Optimization_legacy instead of Optimization")
    ap.add_argument("--compare", action="store_true",
                    help="run both paths and compare them directly")
    args = ap.parse_args()

    from legacy import Optimization
    from legacy import Optimization_legacy

    if args.compare:
        print("Running LEGACY:")
        old = run_seeds(Optimization_legacy.optimize_LGD, args.device,
                        args.seeds, args.n_cond, args.n_mc, "legacy")
        print("Running REFACTORED:")
        new = run_seeds(Optimization.optimize_LGD, args.device,
                        args.seeds, args.n_cond, args.n_mc, "refactored")
        d = abs(new["mean_loss"] - old["mean_loss"])
        rel = d / max(abs(old["mean_loss"]), 1e-12)
        print(f"\nmean loss  legacy={old['mean_loss']:.6f}  "
              f"new={new['mean_loss']:.6f}  rel_diff={rel:.4%}")
        sys.exit(0 if rel <= REL_TOL else 1)

    fn = Optimization_legacy.optimize_LGD if args.legacy else Optimization.optimize_LGD
    label = "legacy" if args.legacy else "refactored"
    print(f"Running {label} on {args.device}, {args.seeds} seeds:")
    res = run_seeds(fn, args.device, args.seeds, args.n_cond, args.n_mc, label)
    print(f"\nmean final loss = {res['mean_loss']:.6f}")

    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    if args.update_golden:
        with open(GOLDEN, "w") as fh:
            json.dump({"mean_loss": res["mean_loss"],
                       "losses": res["losses"],
                       "seeds": args.seeds,
                       "n_cond": args.n_cond,
                       "n_mc": args.n_mc,
                       "label": label}, fh, indent=2)
        print(f"wrote baseline -> {GOLDEN}")
        return

    if not os.path.exists(GOLDEN):
        print(f"\nNo baseline at {GOLDEN}.")
        print("Create one first:  python scripts/run_regression.py "
              "--legacy --update-golden")
        sys.exit(2)

    with open(GOLDEN) as fh:
        golden = json.load(fh)
    rel = abs(res["mean_loss"] - golden["mean_loss"]) / max(abs(golden["mean_loss"]), 1e-12)
    print(f"baseline mean = {golden['mean_loss']:.6f}   rel_diff = {rel:.4%}   "
          f"(tolerance {REL_TOL:.0%})")
    if rel > REL_TOL:
        print("REGRESSION: results moved beyond tolerance.")
        sys.exit(1)
    print("OK: within tolerance.")


if __name__ == "__main__":
    main()
