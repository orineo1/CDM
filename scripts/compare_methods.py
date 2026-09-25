"""
Compare the likelihood-based optimizer against D-FLOW on the same
P(X,Y), same trained flow matching model, and same target G(Y).

This script is self-contained: if --data and --model-dir are not provided,
it will automatically generate data and train the required FM models before
running the comparison.

Runs both methods with multiple attempts and reports:
  - x_hat found by each method
  - L1 distance between P(Y|x_hat) and G(Y)
  - MMD loss
  - Wall-clock time per attempt and total

Usage:
  # All-in-one (generates data + trains models + compares):
  python compare_methods.py --n-attempts 5 --direct

  # With pre-existing data (trains models + compares):
  python compare_methods.py --data experiment_output/data.pt --n-attempts 5 --direct

  # With pre-existing data and models (compares only):
  python compare_methods.py --data experiment_output/data.pt --model-dir experiment_output/models --n-attempts 5 --direct
"""

import argparse
import os
import sys
import time

# Add parent directory to path so we can import repo-level modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import numpy as np

from utils import dist as dist_utils
from models.samplers import flow_matching as FlowMatching
from legacy import Optimization
from likelihood import optimize_likelihood, optimize_likelihood_direct
from losses.functions import MMDLoss, RBF


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare likelihood-based optimizer vs D-FLOW",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs (all optional — will auto-generate if not provided)
    p.add_argument("--data", type=str, default=None,
                   help="Path to .pt file from generate_data.py. "
                        "If not provided, generates data automatically.")
    p.add_argument("--model-dir", type=str, default=None,
                   help="Directory with model_unconditional.pt and model_conditional.pt. "
                        "If not provided, trains models automatically.")

    # Data generation (used when --data is not provided)
    p.add_argument("--n-samples", type=int, default=10000,
                   help="Samples to draw from P(X,Y) (when auto-generating).")
    p.add_argument("--params-file", type=str, default=None,
                   help="SimulationParameters/*.txt file (optional).")
    p.add_argument("--random-mog", action="store_true",
                   help="Generate a random MoG distribution.")
    p.add_argument("--dim", type=int, default=2,
                   help="Dimensionality of (X,Y) joint space.")
    p.add_argument("--n-components", type=int, default=4,
                   help="Number of MoG components.")
    p.add_argument("--condition-on", type=int, default=1,
                   help="Number of leading dimensions that form X.")
    p.add_argument("--mean-sep", type=float, default=5.0,
                   help="Minimum distance between component means. "
                        "Larger = more separated components.")
    p.add_argument("--cov-scale", type=float, default=1.0,
                   help="Multiplier on covariance matrices. "
                        "Smaller = tighter components, less overlap. "
                        "E.g. 0.1 for well-separated, 2.0 for heavy overlap.")

    # Training (used when --model-dir is not provided)
    p.add_argument("--epochs", type=int, default=5,
                   help="Training epochs for the FM models.")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Training batch size.")
    p.add_argument("--nunits", type=int, default=128,
                   help="Hidden dimension for FM models.")
    p.add_argument("--nblocks", type=int, default=3,
                   help="Number of residual blocks.")
    p.add_argument("--guidance-dropout", type=float, default=0.2,
                   help="Classifier-free guidance dropout.")

    # Comparison settings
    p.add_argument("--n-runs", type=int, default=1,
                   help="Number of full pipeline repeats (new data + new model each time). "
                        "Use >1 to measure variance over training randomness.")
    p.add_argument("--n-attempts", type=int, default=5,
                   help="Independent optimization attempts per method per run. "
                        "Best-of-n-attempts is reported for each run.")
    p.add_argument("--optim-samples", type=int, default=250,
                   help="Samples for D-FLOW MMD evaluation.")

    # Likelihood optimizer settings
    p.add_argument("--n-rep", type=int, default=200,
                   help="Representative points from G for likelihood method.")
    p.add_argument("--n-steps", type=int, default=200,
                   help="Gradient steps for likelihood optimizer.")
    p.add_argument("--lr", type=float, default=0.05,
                   help="Learning rate for likelihood optimizer.")
    p.add_argument("--ode-step-size", type=float, default=0.02,
                   help="ODE step size for likelihood computation.")
    p.add_argument("--exact-div", action="store_true", default=False,
                   help="Use exact divergence (slower but more accurate).")
    p.add_argument("--t-dist", type=str, default="uniform",
                   choices=["uniform", "beta", "fixed"],
                   help="Distribution for sampling t at each step.")

    # D-FLOW settings
    p.add_argument("--dflow-max-iter", type=int, default=100,
                   help="Max LBFGS iterations for D-FLOW.")

    # General
    p.add_argument("--direct", action="store_true", default=False,
                   help="Use direct optimization (no unconditional model mapping). "
                        "Better for low-dim X without P(X) prior.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--work-dir", type=str, default="compare_output",
                   help="Directory for auto-generated data/models and results.")
    p.add_argument("--output", type=str, default=None,
                   help="Save comparison results to .pt file.")

    return p.parse_args()


# ---- Data Generation ----

def generate_mog_parameters(d, num_components, mean_sep=5.0, cov_scale=1.0,
                            seed=42, epsilon=1e-4):
    """
    Generate random MoG parameters with controllable spacing and overlap.

    Args:
        d: dimensionality
        num_components: number of Gaussian components
        mean_sep: minimum L2 distance between any two component means.
                  Larger = more separated.
        cov_scale: multiplier on covariance matrices.
                   Smaller = tighter components, less overlap.
                   The base covariance is A@A^T/d (normalized so eigenvalues ~ O(1)),
                   then multiplied by cov_scale.
        seed: random seed
        epsilon: regularization added to covariance diagonal

    Returns:
        mu_list, Sigma_list, alpha
    """
    torch.manual_seed(seed)

    # Generate means with minimum separation
    mu_list = []
    for i in range(num_components):
        for attempt in range(1000):
            candidate = torch.randn(d, dtype=torch.float64) * mean_sep
            valid = all(
                torch.norm(candidate - existing).item() >= mean_sep
                for existing in mu_list
            )
            if valid or attempt == 999:
                mu_list.append(candidate)
                break

    # Generate covariance matrices with controllable scale
    Sigma_list = []
    for _ in range(num_components):
        A = torch.randn(d, d, dtype=torch.float64)
        Sigma = (A @ A.T) / d  # normalize so eigenvalues ~ O(1)
        Sigma = Sigma * cov_scale  # user-controlled scale
        Sigma += torch.eye(d, dtype=torch.float64) * epsilon
        Sigma_list.append(Sigma)

    alpha = torch.full((num_components,), 1.0 / num_components, dtype=torch.float64)
    return mu_list, Sigma_list, alpha


def build_default_2d_mog(x_star_val=None):
    """Built-in 11-component 2D MoG."""
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
    x_star = torch.tensor([-5.0]) if x_star_val is None else torch.tensor(x_star_val)
    return mu_list, Sigma_list, alpha, x_star


def generate_data(args, device):
    """Generate data from P(X,Y) and define target G(Y)."""
    print(f"\n{'='*60}")
    print("Step 1: Generating data")
    print(f"{'='*60}")

    if args.random_mog:
        # Save RNG state before generate_mog_parameters (which reseeds to 42)
        rng_state = torch.random.get_rng_state()
        mu_list, Sigma_list, alpha = generate_mog_parameters(
            d=args.dim, num_components=args.n_components,
            mean_sep=args.mean_sep, cov_scale=args.cov_scale,
        )
        # Restore RNG so that samples and training differ across runs
        torch.random.set_rng_state(rng_state)

        # Pick x* as the X-part of a component mean so that G(Y) is multimodal
        cond = args.condition_on
        # Use fixed seed for x* selection so it's the same across runs
        rng_state2 = torch.random.get_rng_state()
        torch.manual_seed(42)
        selected = torch.randperm(len(mu_list))[:2]
        torch.random.set_rng_state(rng_state2)
        x_star_value = mu_list[selected[0]][:cond].clone()
        # Force these two components to share the same X-part
        for idx in selected:
            mu_list[idx][:cond] = x_star_value
        x_star = x_star_value.float()

        # Compute target G(Y) = P(Y | X = x*)
        mu_cond, Sigma_cond = dist_utils.compute_conditionals(mu_list, Sigma_list, x_star)
        alpha_cond = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_star)
        target_means, target_variances, target_weights = \
            dist_utils.filter_and_normalize(mu_cond, Sigma_cond, alpha_cond, threshold=0.01)

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
        mu_list, Sigma_list, alpha, x_star = build_default_2d_mog()
        mu_cond, Sigma_cond = dist_utils.compute_conditionals(mu_list, Sigma_list, x_star)
        alpha_cond = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_star)
        target_means, target_variances, target_weights = \
            dist_utils.filter_and_normalize(mu_cond, Sigma_cond, alpha_cond, threshold=0.01)

    # Draw samples
    print(f"Drawing {args.n_samples} samples from P(X,Y)...")
    samples = dist_utils.generate_mog_samples_not_differentiable(
        args.n_samples, mu_list, Sigma_list, alpha,
    )

    data = {
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

    # Save for reproducibility
    data_path = os.path.join(args.work_dir, "data.pt")
    os.makedirs(args.work_dir, exist_ok=True)
    torch.save(data, data_path)

    nfeatures = mu_list[0].shape[0]
    n_target = len(target_weights) if isinstance(target_weights, list) else target_weights.shape[0]
    print(f"  P(X,Y): {len(mu_list)}-component MoG in R^{nfeatures}")
    print(f"  Samples: {samples.shape}")
    print(f"  X dims: {args.condition_on}, Y dims: {nfeatures - args.condition_on}")
    print(f"  G(Y): {n_target}-component conditional MoG")
    if hasattr(args, 'mean_sep'):
        print(f"  mean_sep={args.mean_sep}, cov_scale={args.cov_scale}")
    if x_star is not None:
        print(f"  x* = {x_star.numpy()}")
    print(f"  Saved to {data_path}")

    return data


# ---- Model Training ----

def train_models(data, args, device):
    """Train unconditional + conditional FM models."""
    print(f"\n{'='*60}")
    print("Step 2: Training flow matching models")
    print(f"{'='*60}")

    samples = data["samples"]
    condition_on = data["condition_on"]
    nfeatures = samples.shape[1]

    X_only = samples[:, :condition_on].float()
    X_full = samples.float()

    model_dir = os.path.join(args.work_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # --- Unconditional FM model for P(X) ---
    print(f"\n--- Training unconditional flow matching on X (dim={condition_on}) ---")
    model_uncond = FlowMatching.FMModel(
        nfeatures=condition_on, condition_on=0,
        nunits=args.nunits, nblocks=args.nblocks, device=device,
    )

    def data_gen_uncond(batch_size, device="cpu"):
        idx = torch.randint(0, len(X_only), (batch_size,))
        return X_only[idx].to(device)

    total_iters = args.epochs * (len(X_only) // args.batch_size + 1)
    print(f"    {args.epochs} epochs x ~{len(X_only)//args.batch_size+1} batches = {total_iters} iterations")
    model_uncond.train_FM(lr=0.001, batch_size=args.batch_size,
                          data_generator=data_gen_uncond, nepochs=total_iters)

    # Save unconditional
    uncond_path = os.path.join(model_dir, "model_unconditional.pt")
    torch.save({
        "architecture": "fm",
        "state_dict": model_uncond.state_dict(),
        "config": {
            "nfeatures": condition_on,
            "nunits": args.nunits,
            "condition_on": 0,
            "nblocks": len(model_uncond.vf.blocks),
        },
    }, uncond_path)
    print(f"  Saved: {uncond_path}")

    # --- Conditional FM model for P(Y|X) ---
    print(f"\n--- Training conditional flow matching P(Y|X) ---")
    model_cond = FlowMatching.FMModel(
        nfeatures=nfeatures, condition_on=condition_on,
        nunits=args.nunits, nblocks=args.nblocks, device=device,
    )

    def data_gen_cond(batch_size, device="cpu"):
        idx = torch.randint(0, len(X_full), (batch_size,))
        return X_full[idx].to(device)

    total_iters = args.epochs * (len(X_full) // args.batch_size + 1)
    print(f"    {args.epochs} epochs x ~{len(X_full)//args.batch_size+1} batches = {total_iters} iterations")
    model_cond.train_FM(lr=0.001, batch_size=args.batch_size,
                        data_generator=data_gen_cond, nepochs=total_iters,
                        guidance_dropout=args.guidance_dropout)

    # Save conditional
    cond_path = os.path.join(model_dir, "model_conditional.pt")
    torch.save({
        "architecture": "fm",
        "state_dict": model_cond.state_dict(),
        "config": {
            "nfeatures": nfeatures,
            "nunits": args.nunits,
            "condition_on": condition_on,
            "nblocks": len(model_cond.vf.blocks),
        },
    }, cond_path)
    print(f"  Saved: {cond_path}")

    return model_uncond, model_cond


# ---- Model Loading ----

def load_model(path, device):
    """Load a model checkpoint saved by train_models.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt["architecture"]
    config = ckpt["config"]

    if arch == "fm":
        model = FlowMatching.FMModel(**config, device=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        raise ValueError(f"compare_methods.py requires flow matching models, got '{arch}'. "
                         f"Re-train with --architecture fm.")

    model.to(device)
    model.eval()
    print(f"  Loaded {arch} model from {path}")
    return model


# ---- Evaluation ----

def evaluate_solution(x_hat, mu_list, Sigma_list, alpha,
                      target_means, target_variances, target_weights,
                      model_cond, optim_samples, device):
    """Evaluate an x_hat: L1 distance and MMD."""
    x_hat_cpu = x_hat.detach().cpu().squeeze()

    # L1 distance (analytical, between true P(Y|x_hat) and G)
    l1 = dist_utils.warpper_L1_distance(
        x_hat_cpu, mu_list, Sigma_list, alpha,
        target_means, target_variances, target_weights,
    )

    # MMD loss (sample-based, between model's P_hat(Y|x_hat) and G)
    mmd_loss_fn = MMDLoss(kernel=RBF())
    with torch.no_grad():
        x_cond = x_hat.reshape(-1).to(device) * torch.ones(
            optim_samples, model_cond.condition_on, dtype=torch.float32, device=device
        )
        y_samples = model_cond.sample(optim_samples, x_cond)
        y_samples = y_samples.to(device)

        g_samples = dist_utils.generate_mog_samples_not_differentiable(
            optim_samples, target_means, target_variances, target_weights
        ).to(device)

        mmd = mmd_loss_fn(y_samples, g_samples).item()

    return l1.item(), mmd


# ---- Method Runners ----

def run_dflow(model_cond, model_uncond, target_means, target_variances,
              target_weights, optim_samples, dflow_max_iter, device):
    """Run a single D-FLOW attempt."""
    t0 = time.time()
    x_hat, loss_val = Optimization.optimize_DFLOW(
        model_cond, model_uncond, device,
        target_means, target_variances, target_weights,
        max_iter=dflow_max_iter,
        n_sample=optim_samples, loss_method="MMD",
    )
    elapsed = time.time() - t0
    return x_hat, loss_val, elapsed


def run_likelihood(model_cond, model_uncond, target_means, target_variances,
                   target_weights, args, device):
    """Run a single likelihood optimizer attempt."""
    t0 = time.time()

    if args.direct:
        x_hat, loss_val, history = optimize_likelihood_direct(
            model_cond,
            target_means, target_variances, target_weights,
            n_rep=args.n_rep, n_steps=args.n_steps, lr=args.lr,
            step_size=args.ode_step_size,
            exact_divergence=args.exact_div,
            t_distribution=args.t_dist,
            device=device, verbose=False,
        )
    else:
        x_hat, loss_val, history = optimize_likelihood(
            model_cond, model_uncond,
            target_means, target_variances, target_weights,
            n_rep=args.n_rep, n_steps=args.n_steps, lr=args.lr,
            step_size=args.ode_step_size,
            exact_divergence=args.exact_div,
            t_distribution=args.t_dist,
            device=device, verbose=False,
        )

    elapsed = time.time() - t0
    return x_hat, loss_val, history, elapsed


# ---- Single Run ----

def single_run(args, run_idx, device):
    """
    Execute one full pipeline: generate data -> train models -> optimize with
    both methods -> return best-of-n-attempts for each method.

    Returns:
        dict with keys: dflow_best_l1, dflow_best_mmd, dflow_time,
                        likelihood_best_l1, likelihood_best_mmd, likelihood_time,
                        x_star, dflow_results, likelihood_results
    """
    run_seed = (args.seed or 0) + run_idx
    torch.manual_seed(run_seed)

    print(f"\n{'#'*60}")
    print(f"  RUN {run_idx + 1}/{args.n_runs}  (seed={run_seed})")
    print(f"{'#'*60}")

    # ---- Data ----
    if args.data is not None:
        data = torch.load(args.data, weights_only=False)
        print(f"Loaded distribution from {args.data}")
    else:
        data = generate_data(args, device)

    mu_list = data["mu_list"]
    Sigma_list = data["Sigma_list"]
    alpha = data["alpha"]
    target_means = data["target_means"]
    target_variances = data["target_variances"]
    target_weights = data["target_weights"]
    x_star = data["x_star"]

    if x_star is not None:
        print(f"  True x* = {x_star.numpy()}")

    # ---- Models ----
    if args.model_dir is not None:
        print("Loading models...")
        uncond_path = os.path.join(args.model_dir, "model_unconditional.pt")
        cond_path = os.path.join(args.model_dir, "model_conditional.pt")
        model_uncond = load_model(uncond_path, device)
        model_cond = load_model(cond_path, device)
    else:
        model_uncond, model_cond = train_models(data, args, device)
        model_uncond.eval()
        model_cond.eval()

    # ---- D-FLOW ----
    print(f"\n{'='*60}")
    print(f"D-FLOW ({args.n_attempts} attempts, max_iter={args.dflow_max_iter})")
    print(f"{'='*60}")

    dflow_results = []
    for i in range(args.n_attempts):
        x_hat, loss_val, elapsed = run_dflow(
            model_cond, model_uncond,
            target_means, target_variances, target_weights,
            args.optim_samples, args.dflow_max_iter, device,
        )
        l1, mmd = evaluate_solution(
            x_hat, mu_list, Sigma_list, alpha,
            target_means, target_variances, target_weights,
            model_cond, args.optim_samples, device,
        )
        x_np = x_hat.detach().cpu().squeeze().numpy()
        dflow_results.append({
            "x_hat": x_hat.detach().cpu().squeeze(),
            "loss": loss_val.item() if hasattr(loss_val, "item") else float(loss_val),
            "l1": l1, "mmd": mmd, "time": elapsed,
        })
        print(f"  [{i+1}/{args.n_attempts}] x_hat={x_np}, L1={l1:.6f}, "
              f"MMD={mmd:.6f}, time={elapsed:.1f}s")

    # ---- Likelihood ----
    print(f"\n{'='*60}")
    mode = "direct" if args.direct else "with P(X) mapping"
    print(f"Likelihood ({mode}, {args.n_attempts} attempts, "
          f"n_rep={args.n_rep}, n_steps={args.n_steps}, t~{args.t_dist})")
    print(f"{'='*60}")

    likelihood_results = []
    for i in range(args.n_attempts):
        x_hat, loss_val, history, elapsed = run_likelihood(
            model_cond, model_uncond,
            target_means, target_variances, target_weights,
            args, device,
        )
        l1, mmd = evaluate_solution(
            x_hat, mu_list, Sigma_list, alpha,
            target_means, target_variances, target_weights,
            model_cond, args.optim_samples, device,
        )
        x_np = x_hat.detach().cpu().squeeze().numpy()
        likelihood_results.append({
            "x_hat": x_hat.detach().cpu().squeeze(),
            "loss": float(loss_val),
            "l1": l1, "mmd": mmd, "time": elapsed,
            "history": history,
        })
        print(f"  [{i+1}/{args.n_attempts}] x_hat={x_np}, L1={l1:.6f}, "
              f"MMD={mmd:.6f}, time={elapsed:.1f}s")

    # Best of n-attempts for this run
    dflow_best = min(dflow_results, key=lambda r: r["l1"])
    likelihood_best = min(likelihood_results, key=lambda r: r["l1"])

    print(f"\n  Run {run_idx+1} best: D-FLOW L1={dflow_best['l1']:.6f}, "
          f"Likelihood L1={likelihood_best['l1']:.6f}")

    return {
        "dflow_best_l1": dflow_best["l1"],
        "dflow_best_mmd": dflow_best["mmd"],
        "dflow_time": sum(r["time"] for r in dflow_results),
        "likelihood_best_l1": likelihood_best["l1"],
        "likelihood_best_mmd": likelihood_best["mmd"],
        "likelihood_time": sum(r["time"] for r in likelihood_results),
        "x_star": x_star,
        "dflow_results": dflow_results,
        "likelihood_results": likelihood_results,
    }


# ---- Main ----

def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Configuration: {args.n_runs} run(s), {args.n_attempts} attempts/run")

    os.makedirs(args.work_dir, exist_ok=True)

    all_runs = []
    total_t0 = time.time()

    for run_idx in range(args.n_runs):
        run_result = single_run(args, run_idx, device)
        all_runs.append(run_result)

    total_elapsed = time.time() - total_t0

    # ---- Aggregate across runs ----
    dflow_l1s = [r["dflow_best_l1"] for r in all_runs]
    dflow_mmds = [r["dflow_best_mmd"] for r in all_runs]
    dflow_times = [r["dflow_time"] for r in all_runs]
    like_l1s = [r["likelihood_best_l1"] for r in all_runs]
    like_mmds = [r["likelihood_best_mmd"] for r in all_runs]
    like_times = [r["likelihood_time"] for r in all_runs]

    print(f"\n{'='*60}")
    if args.n_runs > 1:
        print(f"FINAL SUMMARY ({args.n_runs} runs, best-of-{args.n_attempts} per run)")
    else:
        print(f"FINAL SUMMARY (best-of-{args.n_attempts})")
    print(f"{'='*60}")

    x_star = all_runs[0]["x_star"]
    if x_star is not None:
        print(f"True x*: {x_star.numpy()}")

    for name, l1s, mmds, times in [
        ("D-FLOW", dflow_l1s, dflow_mmds, dflow_times),
        ("Likelihood", like_l1s, like_mmds, like_times),
    ]:
        print(f"\n  {name}:")
        if args.n_runs == 1:
            print(f"    Best L1:     {l1s[0]:.6f}")
            print(f"    Best MMD:    {mmds[0]:.6f}")
            print(f"    Time:        {times[0]:.1f}s")
        else:
            print(f"    L1:          {np.mean(l1s):.6f} +/- {np.std(l1s):.6f}  "
                  f"(best={min(l1s):.6f}, worst={max(l1s):.6f})")
            print(f"    MMD:         {np.mean(mmds):.6f} +/- {np.std(mmds):.6f}  "
                  f"(best={min(mmds):.6f}, worst={max(mmds):.6f})")
            print(f"    Time/run:    {np.mean(times):.1f}s +/- {np.std(times):.1f}s")

    print(f"\n  Total wall time: {total_elapsed:.1f}s")

    # ---- Save ----
    output_path = args.output or os.path.join(args.work_dir, "comparison.pt")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({
        "all_runs": all_runs,
        "dflow_l1s": dflow_l1s,
        "likelihood_l1s": like_l1s,
        "args": vars(args),
    }, output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
