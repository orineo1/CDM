"""Table 9 and Figure 4 (infeasible target) from the find_xstar.py arm JSONs.

The scenario (config/nonstar_params.txt) has two designs: x_bi = -2, whose conditional is
bimodal (modes at y = 2 and 4), and x_uni = 2, unimodal at y = 3. The target is a point
mass at y* = 4, which no x realises. For each arm (MMD loss, pointwise L2 loss with
n_cond = 250, pointwise with n_cond = 1) this reads the restarts of <arm>.json and reports
the fraction that land closer to x_bi than to x_uni (the paper's "% at x_bi") and the time
per restart, mean +- std. Written to <out>.md and <out>.csv.

Figure 4 (<out>_fig4.png): (a) the recovered x_hat of every restart per arm against the two
designs; (b) the analytic conditionals P(Y | x_bi) and P(Y | x_uni) with y* marked.

    python experiments/synthetic/pipeline/make_table9.py experiments/synthetic/infeasible_target/outputs
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import torch

ARM_ORDER = ["MMD", "L2"]      # loss; arms are then sorted by n_cond, largest first
LABEL = {"MMD": "MMD", "L2": "Pointwise"}


def load_mixture(data_pt):
    d = torch.load(data_pt, weights_only=False)
    mu = np.stack([m.numpy() for m in d["mu_list"]])                  # (K, 2): columns x, y
    var = np.stack([np.diag(S.numpy()) for S in d["Sigma_list"]])    # (K, 2), diagonal covariances
    w = d["alpha"].numpy()
    y_star = float(d["target_means"].reshape(-1)[0])
    return mu, var, w, y_star


def designs(mu, w):
    """x_bi: the x shared by two components; x_uni: the x of the remaining one."""
    xs, counts = np.unique(np.round(mu[:, 0], 6), return_counts=True)
    x_bi = float(xs[np.argmax(counts)])
    x_uni = float(xs[np.argmin(counts)])
    return x_bi, x_uni


def cond_pdf(y, x, mu, var, w):
    """P(Y = y | X = x) for a mixture with diagonal covariances."""
    px = w * np.exp(-0.5 * (x - mu[:, 0]) ** 2 / var[:, 0]) / np.sqrt(2 * np.pi * var[:, 0])
    wk = px / px.sum()
    py = np.exp(-0.5 * (y[:, None] - mu[None, :, 1]) ** 2 / var[None, :, 1]) / np.sqrt(2 * np.pi * var[None, :, 1])
    return py @ wk


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("outputs_dir", help="directory with arm_*.json and data.pt")
    p.add_argument("--out", default=None, help="output stem (default <outputs_dir>/table9)")
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args()
    out = args.out or os.path.join(args.outputs_dir, "table9")

    mu, var, w, y_star = load_mixture(os.path.join(args.outputs_dir, "data.pt"))
    x_bi, x_uni = designs(mu, w)

    arms = []
    for path in glob.glob(os.path.join(args.outputs_dir, "arm_*.json")):
        m = re.match(r"arm_([A-Za-z0-9]+)_n(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        rec = json.load(open(path))
        x_hat = np.array([a["x_hat"][0] for a in rec["attempts"]], dtype=float)
        t = np.array([a["time_s"] for a in rec["attempts"]], dtype=float)
        at_bi = np.abs(x_hat - x_bi) < np.abs(x_hat - x_uni)
        arms.append({"arm": f"{m.group(1)}_n{m.group(2)}", "loss": m.group(1), "n_cond": int(m.group(2)),
                     "restarts": len(x_hat), "pct_at_x_bi": 100.0 * at_bi.mean(),
                     "time_mean": t.mean(), "time_std": t.std(), "x_hat": x_hat})
    if not arms:
        raise SystemExit(f"no arm_*.json under {args.outputs_dir}")
    arms.sort(key=lambda a: (ARM_ORDER.index(a["loss"]) if a["loss"] in ARM_ORDER else 9, -a["n_cond"]))

    with open(out + ".csv", "w", newline="") as fh:
        cols = ["arm", "loss", "n_cond", "restarts", "pct_at_x_bi", "time_mean", "time_std"]
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows([{k: a[k] for k in cols} for a in arms])
    with open(out + "_restarts.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "restart", "x_hat", "at_x_bi"])
        for a in arms:
            for i, x in enumerate(a["x_hat"]):
                wr.writerow([a["arm"], i, f"{x:.6f}", int(abs(x - x_bi) < abs(x - x_uni))])
    md = ["| Loss | n_cond | % at x_bi | Time (s) |", "|---|---|---|---|"]
    for a in arms:
        md.append(f"| {LABEL.get(a['loss'], a['loss'])} | {a['n_cond']} | {a['pct_at_x_bi']:.0f}% | "
                  f"{a['time_mean']:.2f} ± {a['time_std']:.2f} |")
    md.append("")
    md.append(f"R = {arms[0]['restarts']} restarts per arm; x_bi = {x_bi:g}, x_uni = {x_uni:g}, y* = {y_star:g}. "
              "% at x_bi: restarts whose x_hat is closer to x_bi than to x_uni.")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}.md, {out}.csv, {out}_restarts.csv")

    if args.no_figure:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 3.6))
    colors = {"MMD": "tab:red", "L2": "tab:blue"}
    rng = np.random.default_rng(0)
    for i, a in enumerate(arms):
        y = i + rng.uniform(-0.18, 0.18, size=len(a["x_hat"]))
        ax_a.scatter(a["x_hat"], y, s=22, alpha=0.7, color=colors.get(a["loss"], "tab:gray"), edgecolor="none")
    ax_a.axvline(x_bi, color="k", ls="--", lw=1); ax_a.axvline(x_uni, color="k", ls=":", lw=1)
    ax_a.text(x_bi, len(arms) - 0.45, r"$x_{\rm bi}$", ha="center", va="bottom", fontsize=10)
    ax_a.text(x_uni, len(arms) - 0.45, r"$x_{\rm uni}$", ha="center", va="bottom", fontsize=10)
    ax_a.set_yticks(range(len(arms)))
    ax_a.set_yticklabels([f"{LABEL.get(a['loss'], a['loss'])}, $n_{{\\rm cond}}$={a['n_cond']}\n{a['pct_at_x_bi']:.0f}% at $x_{{\\rm bi}}$"
                          for a in arms], fontsize=9)
    ax_a.set_ylim(-0.6, len(arms) - 0.2)
    ax_a.set_xlabel(r"recovered $\hat{x}$ (one point per restart)")
    ax_a.set_title("(a) where each restart lands", fontsize=11)
    ax_a.grid(alpha=0.3, axis="x")

    ys = np.linspace(mu[:, 1].min() - 3 * np.sqrt(var[:, 1].max()), max(mu[:, 1].max(), y_star) + 3 * np.sqrt(var[:, 1].max()), 600)
    ax_b.plot(ys, cond_pdf(ys, x_bi, mu, var, w), color="tab:red", lw=2, label=r"$P(Y \mid x_{\rm bi})$")
    ax_b.plot(ys, cond_pdf(ys, x_uni, mu, var, w), color="tab:blue", lw=2, label=r"$P(Y \mid x_{\rm uni})$")
    ax_b.axvline(y_star, color="k", ls="--", lw=1.2, label=fr"target $y^* = {y_star:g}$")
    ax_b.set_xlabel("$y$"); ax_b.set_ylabel("density")
    ax_b.set_title("(b) the two designs' conditionals", fontsize=11)
    ax_b.legend(frameon=False, fontsize=9); ax_b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out + "_fig4.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out}_fig4.png")


if __name__ == "__main__":
    main()
