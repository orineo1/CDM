"""
A minimal x_hat search on trained checkpoints, and `load_model`, the checkpoint loader the
other pipeline scripts import. The search used in the paper is `search_mog.py`.

Usage (from the repository root):
  python experiments/synthetic/pipeline/run_optimization.py --data runs/mog2d/data.pt --model-dir runs/mog2d/models --method LGD_CM
"""

import argparse
import os
import sys
import time

import torch
from cdm.utils import dist as dist_utils
from cdm.models import diffusion as Diffusion
from cdm.models.samplers import consistency as ConsistencyModels
from cdm.guidance import GuidanceConfig, optimize_guided
from cdm.guidance.config import init_randn, init_zeros, perturb_beta_ratio, aggregate_logsumexp
from cdm.losses.functions import MMDLoss, RBF
from cdm.utils.dist import generate_mog_samples_not_differentiable


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
                   choices=["LGD", "LGD_CM"],
                   help="LGD = diffusion-based, LGD_CM = consistency model.")
    p.add_argument("--n-attempts", type=int, default=5,
                   help="Independent optimization runs (best is selected).")
    p.add_argument("--optim-samples", type=int, default=250,
                   help="Samples per loss evaluation from both P_hat(Y|X) and G(Y).")
    p.add_argument("--mc-samples", type=int, default=3,
                   help="Monte Carlo samples for LGD gradient estimation per step.")
    p.add_argument("--init", choices=["randn", "zeros"], default="randn",
                   help="x_T initialisation of each restart (randn = independent random restarts).")
    p.add_argument("--step-size", type=float, default=1.0, help="Guidance step zeta (LGD / LGD_CM).")

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
            # the paper's LGD configuration, through the library, with independent
            # random restarts
            cfg = GuidanceConfig(
                n_cond=args.optim_samples, n_mc=args.mc_samples,
                init_fn=init_randn if args.init == "randn" else init_zeros, unsqueeze_init=True,
                perturb_fn=perturb_beta_ratio, perturb_square=False, aggregate_fn=aggregate_logsumexp,
                step_size_fn=lambda t, i, z=args.step_size: z,
                timestep_source="range", ddim_step_style="condition_x",
                repeat_condition=True, condition_on_slice=(not CM),
                target_generator=lambda n, dev: generate_mog_samples_not_differentiable(
                    n, target_means, target_variances, target_weights).to(dev),
                compute_final_loss=True, device=device,
                seed=None if args.seed is None else args.seed * 1000 + i,
                tag=f"run_optimization:{args.method}",
            )
            res = optimize_guided(model_uncond, model_cond, MMDLoss(kernel=RBF(device=device)), cfg)
            x_hat, loss_val, trace = res["x_final"], res["final_loss"], res["trace"]

        elapsed = time.time() - t0
        x_val = x_hat.detach().cpu().squeeze()
        loss_scalar = loss_val.item() if hasattr(loss_val, "item") else float(loss_val)
        rec = {"x_hat": x_val, "loss": loss_scalar, "time": elapsed, "trace": trace}
        if x_star is not None:
            mu_p, S_p = dist_utils.compute_conditionals(mu_list, Sigma_list, x_val.view(-1))
            w_p = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_val.view(-1))
            rec["l2_gmm"] = float(dist_utils.gmm_l2_distance(mu_p, S_p, w_p, target_means, target_variances, target_weights))
            rec["l2_to_xstar"] = float(torch.norm(x_val.view(-1) - x_star.float().view(-1)))
        results.append(rec)
        extra = f", L2GMM={rec['l2_gmm']:.4f}, |x-x*|={rec['l2_to_xstar']:.3f}" if "l2_gmm" in rec else ""
        print(f"  [{i+1}/{args.n_attempts}] x_hat={x_val.numpy()}, "
              f"MMD={loss_scalar:.6f}{extra}, time={elapsed:.1f}s")

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
        print(f"  L2 GMM:        {best['l2_gmm']:.6f}   (paper metric)")
        print(f"  |x_hat - x*|:  {best['l2_to_xstar']:.4f}")
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
