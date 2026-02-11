"""
Step 3: Run optimization to find x_hat, then evaluate.

Loads trained models (from train_models.py) and distribution metadata
(from generate_data.py), runs the optimization, and reports results.

Usage:
  python run_optimization.py --data data/mog2d.pt --model-dir models/run1 --method LGD_CM
  python run_optimization.py --data data/mog2d.pt --model-dir models/run1 --method DFLOW --n-attempts 25
  python run_optimization.py --data data/mog2d.pt --model-dir models/run1 --method LGD --n-attempts 10 --optim-samples 500
"""

import argparse
import os
import sys
import time

# Add parent directory to path so we can import repo-level modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import dist_utils
import Diffusion
import ConsistencyModels
import FlowMatching
import Optimization


def parse_args():
    p = argparse.ArgumentParser(
        description="Run optimization to find x_hat and evaluate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs
    p.add_argument("--data", type=str, required=True,
                   help="Path to .pt file from generate_data.py (for distribution params + target).")
    p.add_argument("--model-dir", type=str, required=True,
                   help="Directory containing model_unconditional.pt and model_conditional.pt "
                        "from train_models.py.")

    # Optimization method
    p.add_argument("--method", type=str, required=True,
                   choices=["LGD", "LGD_CM", "DFLOW"],
                   help="LGD = diffusion-based, LGD_CM = consistency model, DFLOW = flow matching.")
    p.add_argument("--n-attempts", type=int, default=5,
                   help="Independent optimization runs (best is selected).")
    p.add_argument("--optim-samples", type=int, default=250,
                   help="Samples per loss evaluation from both P_hat(Y|X) and G(Y).")
    p.add_argument("--mc-samples", type=int, default=3,
                   help="Monte Carlo samples for LGD gradient estimation per step.")
    p.add_argument("--dflow-max-iter", type=int, default=100,
                   help="Max LBFGS iterations for D-FLOW.")

    # Output
    p.add_argument("--output", type=str, default=None,
                   help="Optional .pt file to save results.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


def load_model(path, device):
    """Load a model checkpoint saved by train_models.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt["architecture"]
    config = ckpt["config"]

    if arch == "diffusion":
        model = Diffusion.DiffusionModel(
            nfeatures=config["nfeatures"], nunits=config["nunits"],
            nblocks=config["nblocks"], condition=config["condition"],
            condition_on=config["condition_on"],
            diffusion_steps=config["diffusion_steps"], device=device,
        )
        model.load_state_dict(ckpt["state_dict"])

    elif arch == "cm":
        model = ConsistencyModels.ConsistencyModeliCT(**config, device=device)
        model.load_state_dict(ckpt["state_dict"])

    elif arch == "fm":
        model = FlowMatching.FMModel(**config, device=device)
        model.load_state_dict(ckpt["state_dict"])

    model.to(device)
    model.eval()
    print(f"  Loaded {arch} model from {path}")
    return model, arch


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load distribution metadata ----
    data = torch.load(args.data, weights_only=False)
    mu_list = data["mu_list"]
    Sigma_list = data["Sigma_list"]
    alpha = data["alpha"]
    target_means = data["target_means"]
    target_variances = data["target_variances"]
    target_weights = data["target_weights"]
    x_star = data["x_star"]
    condition_on = data["condition_on"]

    print(f"Loaded distribution from {args.data}")
    if x_star is not None:
        print(f"  True x* = {x_star.numpy()}")

    # ---- Load models ----
    print("Loading models...")
    uncond_path = os.path.join(args.model_dir, "model_unconditional.pt")
    cond_path = os.path.join(args.model_dir, "model_conditional.pt")
    model_uncond, uncond_arch = load_model(uncond_path, device)
    model_cond, cond_arch = load_model(cond_path, device)

    # ---- Run optimization ----
    print(f"\nRunning {args.method} optimization ({args.n_attempts} attempts)...")
    CM = args.method == "LGD_CM"

    results = []
    for i in range(args.n_attempts):
        t0 = time.time()

        if args.method in ("LGD", "LGD_CM"):
            x_hat, _, loss_val = Optimization.optimize_LGD(
                model_uncond, model_cond,
                target_means, target_variances, target_weights,
                mu_list, Sigma_list, alpha,
                nsamples=args.optim_samples, num_x_t=args.mc_samples,
                loss="MMD", CM=CM, device=device,
            )
        elif args.method == "DFLOW":
            x_hat, loss_val = Optimization.optimize_DFLOW(
                model_cond, model_uncond, device,
                target_means, target_variances, target_weights,
                max_iter=args.dflow_max_iter,
                n_sample=args.optim_samples, loss_method="MMD",
            )

        elapsed = time.time() - t0
        x_val = x_hat.detach().cpu().squeeze()
        loss_scalar = loss_val.item() if hasattr(loss_val, "item") else float(loss_val)
        results.append({"x_hat": x_val, "loss": loss_scalar, "time": elapsed})
        print(f"  [{i+1}/{args.n_attempts}] x_hat={x_val.numpy()}, "
              f"MMD={loss_scalar:.6f}, time={elapsed:.1f}s")

    # ---- Evaluate ----
    best = min(results, key=lambda r: r["loss"])
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Best x_hat:    {best['x_hat'].numpy()}")
    print(f"  Best MMD loss: {best['loss']:.6f}")

    if x_star is not None:
        l1 = dist_utils.warpper_L1_distance(
            best["x_hat"], mu_list, Sigma_list, alpha,
            target_means, target_variances, target_weights,
        )
        print(f"  L1 distance:   {l1.item():.6f}")
        print(f"  True x*:       {x_star.numpy()}")

    losses = [r["loss"] for r in results]
    print(f"  All losses:    {['%.4f' % l for l in losses]}")
    print(f"  Mean loss:     {sum(losses)/len(losses):.6f}")
    print(f"  Total time:    {sum(r['time'] for r in results):.1f}s")

    # ---- Save results ----
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        torch.save({
            "results": results,
            "best": best,
            "method": args.method,
            "args": vars(args),
        }, args.output)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
