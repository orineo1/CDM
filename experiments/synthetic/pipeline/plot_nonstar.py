"""
Figure fig:nonstar (no-star target): the joint P(x, y), the two candidate conditionals against the
point target y* = 4, and where each guidance arm's 25 restarts land. Reads the scenario constants
(configs/mog/nonstar_2d.txt) and the arm JSONs (results/nonstar/arm_*.json). Run from the repository root:

    python experiments/synthetic/pipeline/plot_nonstar.py --out experiments/synthetic/results/nonstar/no_star_mmd_vs_l2.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

ARMS = [("MMD_n250", "MMD, $n_{cond}=250$", "#1f77b4"),
        ("L2_n250", "pointwise, $n_{cond}=250$", "#d62728"),
        ("L2_n1", "pointwise, $n_{cond}=1$", "#9467bd")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="experiments/synthetic/configs/mog/nonstar_2d.txt")
    p.add_argument("--runs", default="experiments/synthetic/results/nonstar")
    p.add_argument("--out", default="experiments/synthetic/results/nonstar/no_star_mmd_vs_l2.png")
    p.add_argument("--x-bi", type=float, default=-2.0)
    p.add_argument("--x-uni", type=float, default=2.0)
    a = p.parse_args()
    cfg = {}
    exec(open(a.config).read(), {"torch": torch}, cfg)
    mu = np.array([m.numpy() for m in cfg["mu_list"]])
    S = np.array([s.numpy() for s in cfg["Sigma_list"]])
    w = cfg["alpha"].numpy()
    y_star = float(cfg["target_means"].view(-1)[0])
    sx, sy = np.sqrt(S[:, 0, 0]), np.sqrt(S[:, 1, 1])     # diagonal covariances: Y | x keeps each component's y law

    plt.rcParams.update({"font.size": 15, "axes.labelsize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 13})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # left: the joint, 5,000 draws
    rng = np.random.default_rng(42)
    k = rng.choice(len(w), size=5000, p=w)
    xy = mu[k] + rng.standard_normal((5000, 2)) * np.stack([sx[k], sy[k]], 1)
    ax[0].scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.35)
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$")

    # middle: P(Y | X = x) at x_bi and x_uni (exact: component weights reweighted by the x-marginals)
    ys = np.linspace(0, 6, 1200)

    def cond_pdf(x):
        r = w * np.exp(-0.5 * ((x - mu[:, 0]) / sx) ** 2) / sx
        r = r / r.sum()
        return sum(ri * np.exp(-0.5 * ((ys - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)) for ri, m, s in zip(r, mu[:, 1], sy))
    ax[1].plot(ys, cond_pdf(a.x_bi), color="#ff7f0e", label=f"$P(Y \\mid X = x_{{bi}} = {a.x_bi:g})$")
    ax[1].plot(ys, cond_pdf(a.x_uni), color="#2ca02c", label=f"$P(Y \\mid X = x_{{uni}} = {a.x_uni:g})$")
    ax[1].axvline(y_star, color="#d62728", ls="--", label=f"target $y^* = {y_star:g}$")
    ax[1].set_xlabel("$y$"); ax[1].set_ylabel("density"); ax[1].set_ylim(-0.05, 2.15)

    # right: where the restarts land
    xs = {}
    for name, _, _ in ARMS:
        f = os.path.join(a.runs, f"arm_{name}.json")
        if os.path.exists(f):
            xs[name] = np.array([att["x_hat"][0] for att in json.load(open(f))["attempts"]])
    lo = min(v.min() for v in xs.values()); hi = max(v.max() for v in xs.values())
    bins = np.arange(np.floor(min(lo, a.x_bi - 1) * 4) / 4, np.ceil(max(hi, a.x_uni + 1) * 4) / 4 + 0.25, 0.25)
    for name, label, color in ARMS:
        if name in xs:
            v = xs[name]
            at_bi = np.mean(np.abs(v - a.x_bi) < np.abs(v - a.x_uni)) * 100
            ax[2].hist(v, bins=bins, weights=np.full(len(v), 100 / len(v)), alpha=0.6, color=color)
            print(f"[nonstar] {name} ({color}): {at_bi:.0f}% at x_bi")
    for x0, lbl in ((a.x_bi, "$x_{bi}$"), (a.x_uni, "$x_{uni}$")):
        ax[2].axvline(x0, color="k", ls=":", lw=1)
        ax[2].text(x0, ax[2].get_ylim()[1] * 0.98, lbl, ha="center", va="top", fontsize=16,
                   bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax[2].set_xlabel(r"recovered $\hat x$"); ax[2].set_ylabel("% of restarts")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=170)
    print(f"[nonstar] wrote {a.out}")


if __name__ == "__main__":
    main()
