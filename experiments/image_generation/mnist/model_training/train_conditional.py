#!/usr/bin/env python
"""Train the conditional iCT angle sampler f_phi = P(Y|X) (Table 12).

Three stages: the dataset-construction classifier, the rotation-augmented
search set, then the iCT model itself.

Ported from `MNIST/train/train_conditional.py` of
`orineo1/conditional-matching-paper` (branch `main`, MNIST tree at 7c856e2);
the bare `sys.path`-into-src imports are replaced by `mnist_cdm` imports and
the hard-coded output paths are flags.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import _path  # noqa: F401
from mnist_cdm.cond_model import CircularAngleConsistencyModel  # noqa: E402
from mnist_cdm.dataset import AugmentedMNISTDataset, train_classifier  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clf-epochs", type=int, default=15)
    p.add_argument("--clf-batch", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.9999,
                   help="confidence above which a 90/180/270 copy is kept")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--plots-dir", default="plots")
    p.add_argument("--resume", default=None)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--subset", type=int, default=None,
                   help="use only the first N MNIST images (smoke tier)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    classifier = train_classifier(
        epochs=args.clf_epochs, batch_size=args.clf_batch, device=device,
        save_path=os.path.join(args.ckpt_dir, "construction_classifier.pth"),
        root=args.data_root, limit=args.subset)

    dataset = AugmentedMNISTDataset(classifier, confidence_threshold=args.threshold,
                                    device=device, root=args.data_root, limit=args.subset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)} samples")

    model = CircularAngleConsistencyModel(nfeatures=2, img_features=784, eps=0.002,
                                          nunits=128, depth=5, device=device)
    print(f"Model parameters: {sum(q.numel() for q in model.parameters()):,}")

    losses = model.train_model(dataloader, nepochs=args.epochs, device=device,
                               use_improved_training=True, save_dir=args.ckpt_dir,
                               checkpoint_path=args.resume, plots_dir=args.plots_dir)

    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title("iCT Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_dir, "loss_curve.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
