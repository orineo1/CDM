"""
Step 2: Train generative models from a saved dataset.

Reads the .pt file produced by generate_data.py and trains:
  --model unconditional   :  P(X)   diffusion model
  --model conditional     :  P(Y|X) model (diffusion, cm, or fm)
  --model both            :  both of the above

Saves model checkpoints to --output-dir.

Usage:
  python train_models.py --data data/mog2d.pt --model both --output-dir models/run1
  python train_models.py --data data/mog2d.pt --model conditional --architecture cm --epochs 10 --batch-size 512 --output-dir models/run1
  python train_models.py --data data/mog2d.pt --model unconditional --architecture diffusion --epochs 5 --output-dir models/run1
"""

import argparse
import os
import sys

# Add parent directory to path so we can import repo-level modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import Diffusion
import ConsistencyModels
import FlowMatching


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
                   choices=["diffusion", "cm", "fm"],
                   help="Architecture for the conditional model. "
                        "diffusion = DDIM, cm = Improved Consistency Training, "
                        "fm = Flow Matching. The unconditional model is always diffusion "
                        "(except when architecture=fm, then it is also flow matching).")

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
    p.add_argument("--guidance-dropout", type=float, default=0.2,
                   help="Classifier-free guidance dropout probability.")

    # Output
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to save model checkpoints.")
    p.add_argument("--device", type=str, default=None,
                   help="'cpu' or 'cuda'. Auto-detected if not set.")
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


def train_unconditional(X_train, args, nfeatures_x, device):
    """Train unconditional model on X (first condition_on dims)."""

    if args.architecture == "fm":
        print(f"\n--- Training unconditional flow matching on X (dim={nfeatures_x}) ---")
        model = FlowMatching.FMModel(
            nfeatures=nfeatures_x, condition_on=0,
            nunits=args.nunits, nblocks=args.nblocks, device=device,
        )
        # FM only supports data_generator, so we wrap the fixed dataset
        dataset = torch.utils.data.TensorDataset(X_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        def data_gen_from_loader(batch_size, device="cpu"):
            # This is a workaround: sample a random batch from X_train
            idx = torch.randint(0, len(X_train), (batch_size,))
            return X_train[idx].to(device)

        total_iters = args.epochs * (len(X_train) // args.batch_size + 1)
        print(f"    {args.epochs} epochs x ~{len(X_train)//args.batch_size+1} batches = {total_iters} iterations")
        lr = args.lr if args.lr else 0.001
        model.train_FM(lr=lr, batch_size=args.batch_size,
                       data_generator=data_gen_from_loader, nepochs=total_iters)
    else:
        print(f"\n--- Training unconditional diffusion on X (dim={nfeatures_x}) ---")
        model = Diffusion.DiffusionModel(
            nfeatures=nfeatures_x, nblocks=args.nblocks, nunits=args.nunits,
            condition=False, diffusion_steps=args.diffusion_steps, device=device,
        )
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
        model = ConsistencyModels.ConsistencyModeliCT(
            nfeatures=nfeatures, condition_on=condition_on,
            nunits=args.nunits, depth=args.nblocks, device=device,
        )
        print(f"    Data: {X_train_full.shape[0]} samples, {args.epochs} epochs, "
              f"batch_size={args.batch_size}")
        print(f"    Gradient steps: ~{args.epochs * (len(X_train_full) // args.batch_size + 1)}")
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
        model.train_model(
            X=X_train_full.to(device), nepochs=args.epochs,
            batch_size=args.batch_size, condition_on=condition_on,
            guidance_dropout=args.guidance_dropout, device=device,
        )

    elif args.architecture == "fm":
        print(f"\n--- Training conditional flow matching P(Y|X) ---")
        model = FlowMatching.FMModel(
            nfeatures=nfeatures, condition_on=condition_on,
            nunits=args.nunits, nblocks=args.nblocks, device=device,
        )
        # FM only supports data_generator
        def data_gen_from_dataset(batch_size, device="cpu"):
            idx = torch.randint(0, len(X_train_full), (batch_size,))
            return X_train_full[idx].to(device)

        total_iters = args.epochs * (len(X_train_full) // args.batch_size + 1)
        print(f"    {args.epochs} epochs x ~{len(X_train_full)//args.batch_size+1} batches = {total_iters} iterations")
        lr = args.lr if args.lr else 0.001
        model.train_FM(lr=lr, batch_size=args.batch_size,
                       data_generator=data_gen_from_dataset, nepochs=total_iters,
                       guidance_dropout=args.guidance_dropout)

    return model


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
            "nblocks": len(model.net.blocks),
            "condition": model.condition,
            "condition_on": model.condition_on,
            "diffusion_steps": model.diffusion_steps,
        }
    elif architecture == "cm":
        checkpoint["config"] = model.get_config()
    elif architecture == "fm":
        checkpoint["config"] = {
            "nfeatures": model.nfeatures,
            "nunits": model.nunits,
            "condition_on": model.condition_on,
            "nblocks": len(model.vf.blocks),
        }
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

    X_only = samples[:, :condition_on].float()
    X_full = samples.float()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Train ----
    if args.model in ("unconditional", "both"):
        uncond_arch = "fm" if args.architecture == "fm" else "diffusion"
        model_uncond = train_unconditional(X_only, args, condition_on, device)
        save_model(model_uncond,
                   os.path.join(args.output_dir, "model_unconditional.pt"),
                   uncond_arch)

    if args.model in ("conditional", "both"):
        model_cond = train_conditional(X_full, args, nfeatures, condition_on, device)
        save_model(model_cond,
                   os.path.join(args.output_dir, "model_conditional.pt"),
                   args.architecture)

    print("\nDone.")


if __name__ == "__main__":
    main()
