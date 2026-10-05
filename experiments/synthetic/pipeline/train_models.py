"""
Step 2: Train generative models from a saved dataset.

Reads the .pt file produced by generate_data.py and trains:
  --model unconditional   :  P(X)   diffusion model
  --model conditional     :  P(Y|X) model (diffusion or cm)
  --model both            :  both of the above

Saves model checkpoints to --output-dir.

Usage (from the repository root):
  python experiments/synthetic/pipeline/train_models.py --data runs/mog2d/data.pt --model both --output-dir runs/mog2d/models
  python experiments/synthetic/pipeline/train_models.py --data runs/mog2d/data.pt --model conditional --architecture cm \
      --epochs 10 --batch-size 512 --output-dir runs/mog2d/models
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from cdm.models import diffusion as Diffusion
from cdm.models.samplers import consistency as ConsistencyModels


def parse_args():
    p = argparse.ArgumentParser(
        description="Train generative models on a fixed dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    p.add_argument("--data", type=str, required=True,
                   help="Path to .pt file from generate_data.py.")

    # What to train
    p.add_argument("--model", type=str, required=True,
                   choices=["unconditional", "conditional", "both"],
                   help="Which model(s) to train.")
    p.add_argument("--architecture", type=str, default="cm",
                   choices=["diffusion", "cm"],
                   help="Architecture for the conditional model. "
                        "diffusion = DDIM, cm = Improved Consistency Training. "
                        "The unconditional model is always diffusion.")

    # Architecture hyperparameters
    p.add_argument("--nunits", type=int, default=128,
                   help="Hidden dimension.")
    p.add_argument("--nblocks", type=int, default=3,
                   help="Number of residual blocks.")
    p.add_argument("--diffusion-steps", type=int, default=100,
                   help="Discrete timesteps in diffusion noise schedule.")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=5,
                   help="Number of passes over the dataset (true epochs).")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Mini-batch size.")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate. If not set, uses model defaults "
                        "(0.001 for diffusion/fm, 0.0001 for cm).")
    p.add_argument("--fresh-samples", action="store_true",
                   help="train on fresh draws from the mixture in the data file (one batch per epoch) "
                        "instead of the stored sample set; --epochs then counts batches")
    p.add_argument("--guidance-dropout", type=float, default=0.2,
                   help="Classifier-free guidance dropout probability.")

    # Output
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to save model checkpoints.")
    p.add_argument("--device", type=str, default=None,
                   help="'cpu' or 'cuda'. Auto-detected if not set.")
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


GEN = {}   # fresh-sample generators, set in main() with --fresh-samples


def train_unconditional(X_train, args, nfeatures_x, device):
    """Train unconditional model on X (first condition_on dims)."""

    print(f"\n--- Training unconditional diffusion on X (dim={nfeatures_x}) ---")
    model = Diffusion.DiffusionModel(
        nfeatures=nfeatures_x, nblocks=args.nblocks, nunits=args.nunits,
        condition=False, diffusion_steps=args.diffusion_steps, device=device,
    )
    if args.fresh_samples:
        print(f"    Fresh draws: {args.epochs} batches of {args.batch_size}")
        model.train_model(X=None, data_generator=GEN["x"], nepochs=args.epochs, batch_size=args.batch_size,
                          device=device)
    else:
        print(f"    Data: {X_train.shape[0]} samples, {args.epochs} epochs, "
              f"batch_size={args.batch_size}")
        print(f"    Gradient steps: ~{args.epochs * (len(X_train) // args.batch_size + 1)}")
        model.train_model(
            X=X_train.to(device), nepochs=args.epochs, batch_size=args.batch_size,
            device=device,
        )

    return model


def train_conditional(X_train_full, args, nfeatures, condition_on, device):
    """Train conditional model on Y|X."""

    if args.architecture == "cm":
        print(f"\n--- Training conditional consistency model (iCT) P(Y|X) ---")
        # iCT models Y | X only: its feature dimension is dim(Y) = nfeatures - condition_on
        # (the conditional diffusion model below models the full joint and slices Y later)
        model = ConsistencyModels.ConsistencyModeliCT(
            nfeatures=nfeatures - condition_on, condition_on=condition_on,
            nunits=args.nunits, depth=args.nblocks, device=device,
        )
        print(f"    Data: {X_train_full.shape[0]} samples, {args.epochs} epochs, "
              f"batch_size={args.batch_size}")
        print(f"    Gradient steps: ~{args.epochs * (len(X_train_full) // args.batch_size + 1)}")
        if args.fresh_samples:
            print(f"    Fresh draws: {args.epochs} batches of {args.batch_size}")
            model.train_model(X=None, data_generator=GEN["joint"], nepochs=args.epochs, batch_size=args.batch_size,
                              device=device, condition=condition_on, use_improved_training=True)
        else:
            model.train_model(
                X=X_train_full.to(device), nepochs=args.epochs,
                batch_size=args.batch_size, device=device,
                condition=condition_on, use_improved_training=True,
            )

    elif args.architecture == "diffusion":
        print(f"\n--- Training conditional diffusion P(Y|X) ---")
        model = Diffusion.DiffusionModel(
            nfeatures=nfeatures, nblocks=args.nblocks, nunits=args.nunits,
            condition=True, condition_on=condition_on,
            diffusion_steps=args.diffusion_steps, device=device,
        )
        print(f"    Data: {X_train_full.shape[0]} samples, {args.epochs} epochs, "
              f"batch_size={args.batch_size}")
        if args.fresh_samples:
            print(f"    Fresh draws: {args.epochs} batches of {args.batch_size}")
            model.train_model(X=None, data_generator=GEN["joint"], nepochs=args.epochs, batch_size=args.batch_size,
                              condition_on=condition_on, guidance_dropout=args.guidance_dropout, device=device)
        else:
            model.train_model(
                X=X_train_full.to(device), nepochs=args.epochs,
                batch_size=args.batch_size, condition_on=condition_on,
                guidance_dropout=args.guidance_dropout, device=device,
            )

    return model


def _n_blocks(net):
    """Number of residual blocks of a GenericNN, robust to the attribute name
    (`blocks` or `res_blocks`)."""
    import torch.nn as nn
    for attr in ("blocks", "res_blocks"):
        if hasattr(net, attr):
            return len(getattr(net, attr))
    return sum(isinstance(m, nn.Linear) for m in net.modules()) - 1

def save_model(model, path, architecture, config_extra=None):
    """Save model in a uniform format."""
    checkpoint = {
        "architecture": architecture,
        "state_dict": model.state_dict(),
    }
    # Save constructor args so we can reconstruct
    if architecture == "diffusion":
        checkpoint["config"] = {
            "nfeatures": model.nfeatures,
            "nunits": model.nunits,
            "nblocks": _n_blocks(model.net),
            "condition": model.condition,
            "condition_on": model.condition_on,
            "diffusion_steps": model.diffusion_steps,
        }
    elif architecture == "cm":
        checkpoint["config"] = model.get_config()
    if config_extra:
        checkpoint.update(config_extra)
    torch.save(checkpoint, path)
    print(f"  Saved: {path}")


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load data ----
    data = torch.load(args.data, weights_only=False)
    samples = data["samples"]
    condition_on = data["condition_on"]
    nfeatures = samples.shape[1]

    print(f"Loaded {args.data}")
    print(f"  Samples: {samples.shape} ({samples.shape[0]} points in R^{nfeatures})")
    print(f"  X dims: {condition_on}, Y dims: {nfeatures - condition_on}")

    if args.fresh_samples:
        from functools import partial
        from cdm.utils.dist import generate_mog_samples_not_differentiable
        mu, Sg, al = data["mu_list"], data["Sigma_list"], data["alpha"]
        gj = partial(generate_mog_samples_not_differentiable, means=mu, variances=Sg, weights=al)
        gx = partial(generate_mog_samples_not_differentiable, means=mu, variances=Sg, weights=al,
                     kernel_func=lambda X: X[:, :condition_on])
        GEN["joint"] = lambda n, device="cpu": gj(n, device=device).float()      # mixture params may be float64
        GEN["x"] = lambda n, device="cpu": gx(n, device=device).float()
    X_only = samples[:, :condition_on].float()
    X_full = samples.float()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Train ----
    if args.model in ("unconditional", "both"):
        model_uncond = train_unconditional(X_only, args, condition_on, device)
        save_model(model_uncond,
                   os.path.join(args.output_dir, "model_unconditional.pt"),
                   "diffusion")

    if args.model in ("conditional", "both"):
        model_cond = train_conditional(X_full, args, nfeatures, condition_on, device)
        save_model(model_cond,
                   os.path.join(args.output_dir, "model_conditional.pt"),
                   args.architecture)

    print("\nDone.")


if __name__ == "__main__":
    main()
