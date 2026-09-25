"""
Step 1: Generate data from P(X,Y) and define the target G(Y).

Outputs a .pt file containing:
  - samples:           [n, d] tensor of i.i.d. draws from P(X,Y)
  - mu_list:           list of component means
  - Sigma_list:        list of component covariances
  - alpha:             mixture weights
  - target_means:      G(Y) component means
  - target_variances:  G(Y) component covariances
  - target_weights:    G(Y) mixture weights
  - x_star:            the true optimal conditioning value (if known)
  - condition_on:      number of X dimensions

Usage:
  python generate_data.py --n-samples 10000 --output data/mog2d.pt
  python generate_data.py --params-file SimulationParameters/mog_2d_full.txt --n-samples 50000 --output data/mog2d_50k.pt
  python generate_data.py --random-mog --dim 10 --n-components 4 --n-samples 10000 --output data/mog10d.pt
"""

import argparse
import os
import sys

# Add parent directory to path so we can import repo-level modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import dist_utils


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate data from P(X,Y) and define target G(Y)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Distribution source
    src = p.add_mutually_exclusive_group()
    src.add_argument("--params-file", type=str, default=None,
                     help="SimulationParameters/*.txt file with mu_list, Sigma_list, alpha, "
                          "target_means, target_variances, target_weights.")
    src.add_argument("--random-mog", action="store_true",
                     help="Generate a random MoG via dist_utils.get_param_mog_with_target().")

    # MoG configuration
    p.add_argument("--dim", type=int, default=2,
                   help="Dimensionality of (X,Y) joint space.")
    p.add_argument("--n-components", type=int, default=4,
                   help="Number of MoG components (for --random-mog).")
    p.add_argument("--condition-on", type=int, default=1,
                   help="Number of leading dimensions that form X.")
    p.add_argument("--uncond-dim", type=int, default=1,
                   help="Number of unconditional (Y) dimensions per component.")
    p.add_argument("--min-separation", type=float, default=5.0,
                   help="Minimum distance between MoG component means (for --random-mog).")

    # Target G(Y) = P(Y|X=x*)
    p.add_argument("--x-star", type=float, nargs="+", default=None,
                   help="The true x* for computing G(Y)=P(Y|X=x*). "
                        "If not set, uses default or auto-generated value.")
    p.add_argument("--filter-threshold", type=float, default=0.01,
                   help="Weight threshold for pruning negligible G(Y) components.")

    # Data
    p.add_argument("--n-samples", type=int, required=True,
                   help="Number of i.i.d. samples to draw from P(X,Y).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility.")

    # Output
    p.add_argument("--output", type=str, required=True,
                   help="Output .pt file path.")

    return p.parse_args()


def build_default_2d_mog(args):
    """Built-in 11-component 2D MoG (same as mog_2d_full.txt)."""
    mu_list = [
        torch.tensor([-5.0, 5.0]),  torch.tensor([-5.0, -5.0]),
        torch.tensor([5.0, 3.0]),   torch.tensor([5.0, -1.0]),
        torch.tensor([0.0, -3.0]),  torch.tensor([-2.0, 4.0]),
        torch.tensor([-2.0, -3.0]), torch.tensor([1.0, 2.0]),
        torch.tensor([-7.0, 1.0]),  torch.tensor([7.0, 5.0]),
        torch.tensor([0.0, -5.0]),
    ]
    cov = torch.tensor([[0.5, 0.195], [0.195, 0.2]])
    Sigma_list = [cov.clone() for _ in mu_list]
    alpha = torch.ones(len(mu_list)) / len(mu_list)

    x_star = torch.tensor(args.x_star) if args.x_star else torch.tensor([-5.0])
    return mu_list, Sigma_list, alpha, x_star


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ---- Build distribution parameters ----
    if args.random_mog:
        mu_list, Sigma_list, alpha, target_means, target_variances, target_weights, x_star = \
            dist_utils.get_param_mog_with_target(
                dim_data=args.dim, num_components=args.n_components,
                device="cpu", conditional_modes=2, distanceOrScale="Distance",
                uncond_dim=args.uncond_dim, min_distance=args.min_separation,
            )

    elif args.params_file:
        local_vars = {}
        with open(args.params_file, "r") as f:
            exec(f.read(), {"torch": torch}, local_vars)
        mu_list = local_vars["mu_list"]
        Sigma_list = local_vars["Sigma_list"]
        alpha = local_vars["alpha"]
        target_means = local_vars["target_means"]
        target_variances = local_vars["target_variances"]
        target_weights = local_vars["target_weights"]
        x_star = local_vars.get("x_star", None)

    else:
        mu_list, Sigma_list, alpha, x_star = build_default_2d_mog(args)
        target_means = target_variances = target_weights = None  # computed below

    # ---- Compute target G(Y) if not already set ----
    if target_means is None:
        mu_cond, Sigma_cond = dist_utils.compute_conditionals(mu_list, Sigma_list, x_star)
        alpha_cond = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_star)
        target_means, target_variances, target_weights = \
            dist_utils.filter_and_normalize(mu_cond, Sigma_cond, alpha_cond,
                                            threshold=args.filter_threshold)

    # ---- Draw samples ----
    print(f"Drawing {args.n_samples} samples from P(X,Y)...")
    samples = dist_utils.generate_mog_samples_not_differentiable(
        args.n_samples, mu_list, Sigma_list, alpha,
    )

    # ---- Save ----
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "samples": samples,
        "mu_list": mu_list,
        "Sigma_list": Sigma_list,
        "alpha": alpha,
        "target_means": target_means,
        "target_variances": target_variances,
        "target_weights": target_weights,
        "x_star": x_star,
        "condition_on": args.condition_on,
        "n_samples": args.n_samples,
    }
    torch.save(payload, args.output)

    nfeatures = mu_list[0].shape[0]
    n_target = len(target_weights) if isinstance(target_weights, list) else target_weights.shape[0]
    print(f"Saved to {args.output}")
    print(f"  P(X,Y): {len(mu_list)}-component MoG in R^{nfeatures}")
    print(f"  Samples: {samples.shape}")
    print(f"  X dims: {args.condition_on}, Y dims: {nfeatures - args.condition_on}")
    print(f"  G(Y): {n_target}-component conditional MoG")
    if x_star is not None:
        print(f"  x* = {x_star.numpy()}")


if __name__ == "__main__":
    main()
