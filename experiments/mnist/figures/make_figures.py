#!/usr/bin/env python
"""MNIST figures: the top-k digit panel, the per-seed rows, and the polar /
cartesian histograms of P(Y | X = x_hat) against the target density
(the paper's MNIST figure and its all-seeds appendix figures).

Reads the `.pkl` run records written by `table_mlgdf/search_mnist.py`.

    python make_figures.py --results-dir <dir with .pkl> --out <dir>

The plotting itself is in `cdm.models.mnist.plotting`; this file is the entry point only.
"""

import argparse
import os
import pickle
import random

import numpy as np
import torch

from cdm.models.mnist.checkpoints import load_cond_model  # noqa: E402
from cdm.models.mnist.plotting import (clamp_normalize, plot_all_seeds_grid,  # noqa: E402
                                plot_top_k_distributions, plot_top_k_images)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT_DIR = os.path.join(os.path.dirname(HERE), "checkpoints")

GLOBAL_SEED = 42


def set_global_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_pkls(results_dir):
    found = []
    for root, _, files in os.walk(results_dir):
        for fname in sorted(files):
            if fname.endswith(".pkl"):
                with open(os.path.join(root, fname), "rb") as fh:
                    found.append((os.path.splitext(fname)[0], clamp_normalize(pickle.load(fh))))
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, help="directory holding run .pkl files")
    p.add_argument("--out", required=True, help="where to write the figures")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--n-cond-plot", type=int, default=500,
                   help="angle samples drawn per displayed digit for the histograms")
    p.add_argument("--no-distributions", action="store_true",
                   help="skip the histogram panels (they need the conditional model)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt-dir", default=None,
                   help="local checkpoint directory; HuggingFace is the fallback")
    args = p.parse_args()

    set_global_seed()
    os.makedirs(args.out, exist_ok=True)

    payloads = find_pkls(args.results_dir)
    if not payloads:
        raise SystemExit(f"no .pkl files under {args.results_dir}")

    cond_model = None
    if not args.no_distributions:
        ckpt_dir = args.ckpt_dir or DEFAULT_CKPT_DIR
        cond_model = load_cond_model(device=args.device, local_dir=ckpt_dir)

    for label, payload in payloads:
        print(f"== {label}")
        # three files per target: <run>_all_seeds.png, <run>_top5_notitle.png,
        # <run>_dist_polar_notitle.png
        plot_all_seeds_grid(payload, args.out, dpi=args.dpi)
        top_ix = plot_top_k_images(payload, args.out, top_k=args.top_k, dpi=args.dpi,
                                   save_titled=False, save_no_title=True)
        if cond_model is not None:
            plot_top_k_distributions(payload, cond_model, top_ix, args.out,
                                     top_k_dist=args.top_k, dpi=args.dpi,
                                     save_titled=False, save_cart=False, save_no_title=True,
                                     device=args.device, n_cond_plot=args.n_cond_plot)


if __name__ == "__main__":
    main()
