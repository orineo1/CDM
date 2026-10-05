#!/usr/bin/env python
"""Compact MNIST figure for the main text: one column per target (uniform, bimodal, unimodal), the
`--top-k` recovered digits with the lowest SWD on top and, below them, the polar histogram of
P(Y | X = x_hat) for the best digit against the target density.

Reads the digits from Table 2's committed records (`../table_mlgdf/results/images.npz`: all 15 seeds per
target with their SWD), so nothing is optimized again; the histogram samples the conditional model (CPU is
enough). Same sampling, colors and orientation as `make_figures.py` (cdm.models.mnist.plotting).

    python experiments/mnist/figures/make_compact_figure.py --out results/mnist_compact.png
"""
import argparse
import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np   # noqa: E402
import torch   # noqa: E402

from cdm.models.mnist.checkpoints import load_cond_model   # noqa: E402
from cdm.models.mnist.plotting import ICT_TS, circular_to_angles   # noqa: E402
from cdm.models.mnist.targets import make_target   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(os.path.dirname(HERE), "table_mlgdf")
TARGETS = ("uniform", "bimodal", "unimodal")


def target_variances():
    """bimodal / unimodal variances of the published runs (config/targets.tsv)."""
    var = {}
    with open(os.path.join(TABLE, "config", "targets.tsv")) as f:
        for row in csv.reader((l for l in f if not l.startswith("#")), delimiter="\t"):
            row = [c for c in row if c.strip()] or [c for c in " ".join(row).split()]
            if len(row) >= 6 and row[5] != "-":
                var[row[0]] = float(row[5])
    return var


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default=os.path.join(TABLE, "results", "images.npz"))
    p.add_argument("--out", required=True)
    p.add_argument("--top-k", type=int, default=3, help="digits shown per target")
    p.add_argument("--n-cond-plot", type=int, default=500, help="angle samples for the histogram")
    p.add_argument("--width", type=float, default=3.25, help="figure width in inches (one column)")
    p.add_argument("--font", type=float, default=8.0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ckpt-dir", default=os.path.join(os.path.dirname(HERE), "checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rows-are-targets", action="store_true",
                   help="one row per target (digits, then the dial) instead of one column per target")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    d = np.load(a.images)
    var = target_variances()
    model = load_cond_model(device=a.device, local_dir=a.ckpt_dir)
    k = a.top_k
    T = a.rows_are_targets
    if T:       # rows: targets; columns: k digits and the dial; target names as row labels
        lab = 0.06
        cell = a.width * (1 - lab - 0.01) / (k + 1.1)
        fig = plt.figure(figsize=(a.width, 3 * cell + 0.08))
        gs = fig.add_gridspec(3, k + 1, hspace=0.05, wspace=0.05, left=lab, right=0.99, top=0.985, bottom=0.015,
                              width_ratios=[1] * k + [1.1])
    else:
        cell = a.width / 3
        fig = plt.figure(figsize=(a.width, cell * (k + 1) + 0.25))
        gs = fig.add_gridspec(k + 1, 3, hspace=0.06, wspace=0.06, left=0.01, right=0.99, top=0.95, bottom=0.05,
                              height_ratios=[1] * k + [1.15])
    for c, name in enumerate(TARGETS):
        imgs, swd = d[f"{name}_images"], d[f"{name}_swd"]
        order = np.argsort(swd)[:k]
        for r, idx in enumerate(order):
            ax = fig.add_subplot(gs[c, r] if T else gs[r, c])
            ax.imshow((np.clip(imgs[idx], -1, 1) + 1) / 2, cmap="gray")       # per-image range, as make_figures.py
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                if T:
                    ax.set_ylabel(name, fontsize=a.font + 1, labelpad=3)
                else:
                    ax.set_title(name, fontsize=a.font + 1, pad=3)
        tgt = make_target(name, unimodal_var=var.get("unimodal", 515.0), bimodal_var=var.get("bimodal", 300.0))
        x_range, pdf = tgt.density_curve()
        with torch.no_grad():
            cond = torch.tensor(imgs[order[0]], dtype=torch.float32).flatten().unsqueeze(0).to(a.device)
            ang = circular_to_angles(model.sample(nsamples=a.n_cond_plot, condition_x=cond, ts=ICT_TS)[0])
        ang_rad = np.deg2rad(ang.detach().cpu().numpy())
        edges = np.linspace(0, 2 * np.pi, 37)
        counts, _ = np.histogram(ang_rad, bins=edges, density=True)
        tx = np.deg2rad(np.append(x_range, x_range[0]))
        ty = np.append(pdf, pdf[0]) * (360.0 / (2 * np.pi))
        ax = fig.add_subplot(gs[c, k] if T else gs[k, c], projection="polar")
        ax.bar(0.5 * (edges[:-1] + edges[1:]), counts, width=edges[1] - edges[0], alpha=0.55,
               color="skyblue", edgecolor="black", linewidth=0.3, zorder=2)
        ax.fill(tx, ty, color="orange", alpha=0.35, zorder=3)
        ax.plot(tx, ty, color="orange", linewidth=1.2, zorder=4)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_yticks([]); ax.set_yticklabels([])
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        # 90 / 270 would collide with the next dial; with rows per target every label would (orientation in the caption)
        ax.set_xticklabels(["", "", "", ""] if T else ["0", "", "180", ""], fontsize=a.font - 1)
        ax.tick_params(axis="x", pad=0)
        ax.set_ylim(0, max(counts.max(), ty.max()) * 1.05)
        ax.grid(linewidth=0.3)
        b = ax.get_position()                       # shrink the dial so its angle labels fit in the cell
        s_ = 0.92 if T else 0.8
        ax.set_position([b.x0 + b.width * (1 - s_) / 2, b.y0 + b.height * (1 - s_) / 2, b.width * s_, b.height * s_])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
