#!/usr/bin/env python
"""Train the unconditional DDPM prior P(X) on rotated MNIST.

`UnconditionalUnet` and `RotatedMNISTDataset` come from `cdm.models.mnist`.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from cdm.models.mnist.checkpoints import make_noise_scheduler  # noqa: E402
from cdm.models.mnist.dataset import RotatedMNISTDataset       # noqa: E402
from cdm.models.mnist.uncond_model import UnconditionalUnet    # noqa: E402


def visualize_samples(model, noise_scheduler, epoch, device, n=16, plots_dir="plots"):
    model.eval()
    with torch.no_grad():
        ddim = DDIMScheduler.from_config(noise_scheduler.config)
        ddim.set_timesteps(num_inference_steps=50)
        x = torch.randn(n, 1, 28, 28, device=device)
        for t in ddim.timesteps:
            x = ddim.step(model(x, t), t, x).prev_sample
    grid = torchvision.utils.make_grid(x.cpu().clip(-1, 1), nrow=8)[0]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(grid, cmap="gray")
    ax.set_title(f"Generated Samples — Epoch {epoch} (DDIM, unconditional)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"samples_epoch_{epoch}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    model.train()


def train(net, dataloader, noise_scheduler, n_epochs, lr, device,
          checkpoint_dir, resume_from_epoch=None, plots_dir="plots",
          checkpoint_every=25):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    losses, start = [], 0

    if resume_from_epoch is not None:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{resume_from_epoch}.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            losses, start = ckpt["losses"], resume_from_epoch
            print(f"Resumed from epoch {start}")
        else:
            print(f"Checkpoint not found: {ckpt_path}")

    net.train()
    for epoch in range(start, n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for x, _, __ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            x = x.to(device) * 2 - 1
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],), device=device).long()
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
            loss = loss_fn(net(noisy_x, timesteps), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            epoch_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch+1:4d} | avg loss {epoch_loss / max(n_batches, 1):.5f}")
        if (epoch + 1) % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({"epoch": epoch, "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": opt.state_dict(), "losses": losses,
                        "avg_loss": epoch_loss / max(n_batches, 1)}, path)
            print(f"Checkpoint saved: {path}")
            visualize_samples(net, noise_scheduler, epoch + 1, device, plots_dir=plots_dir)
    return net, losses


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--resume", type=int, default=None)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--plots-dir", default="plots")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--subset", type=int, default=None,
                   help="use only the first N images (smoke tier)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = RotatedMNISTDataset(root=args.data_root)
    if args.subset:
        dataset = torch.utils.data.Subset(dataset, range(args.subset))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))

    net = UnconditionalUnet().to(device)
    print(f"Model parameters: {sum(q.numel() for q in net.parameters()):,}")
    net, losses = train(net, dataloader, make_noise_scheduler(), args.epochs, args.lr,
                        device, args.ckpt_dir, args.resume, args.plots_dir,
                        args.checkpoint_every)

    os.makedirs(args.plots_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_dir, "loss_curve.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
