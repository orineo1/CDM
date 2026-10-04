"""Free-energy maps over the two bond angles for chosen targets: target vs. the eps returned by each method.

Each panel is F(theta123, theta234) = -log p (k_BT units, shifted so the minimum is 0) of a fresh
simulation at that eps (`simulate.py`, the evaluation's run length and replica count), in the style of
Zick & Mendels (2026), Fig. 3-4. Panels are annotated with eps, the basin populations (their basin
definitions) and, for methods, the error of the free-energy differences between basins.

  python experiments/peptide/pipeline/plot_fes.py --per-run experiments/peptide/results/per_run.json \
      --targets 8 19 18 --methods "GB-FESO [low-high]" MLGD --out experiments/peptide/results/fes_examples.png
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common   # noqa: E402,F401  (puts GB-FESO on the path)
from simulate import simulate   # noqa: E402
from utils import basin_probabilities, compare_basin_metrics   # noqa: E402  (GB-FESO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-run", required=True)
    p.add_argument("--targets", type=int, nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=["GB-FESO [low-high]", "MLGD"])
    p.add_argument("--labels", nargs="+", default=None, help="column titles for the methods")
    p.add_argument("--n-rep", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=4_000_000)
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--device", default=None)
    p.add_argument("--cache", default=None, help=".pt cache of the simulated ensembles (reused when present)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = json.load(open(a.per_run))
    by = {(r["method"], r["target_id"]): r for r in runs}
    eps_list, keys = [], []
    for t in a.targets:
        any_run = next(r for r in runs if r["target_id"] == t)
        eps_list.append(any_run["eps_star"]); keys.append(("target", t))
        for m in a.methods:
            eps_list.append(by[(m, t)]["eps_eval"]); keys.append((m, t))
    if a.cache and os.path.exists(a.cache):
        Y = torch.load(a.cache)
    else:
        Y = simulate(eps_list, n_rep=a.n_rep, n_steps=a.n_steps, device=a.device, verbose=True)
        if a.cache:
            torch.save(Y, a.cache)

    labels = a.labels or a.methods
    nr, nc = len(a.targets), 1 + len(a.methods)
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.9 * nr), squeeze=False)
    edges = np.linspace(-np.pi, np.pi, a.bins + 1)
    vmax = 6.0
    for i, (key, eps) in enumerate(zip(keys, eps_list)):
        row, col = i // nc, i % nc
        ang = Y[i][:, 3:5].numpy()
        H, _, _ = np.histogram2d(ang[:, 0], ang[:, 1], bins=[edges, edges], density=True)
        with np.errstate(divide="ignore"):
            F = -np.log(H)
        F -= np.nanmin(F[np.isfinite(F)])
        F[~np.isfinite(F)] = vmax
        ax = axes[row, col]
        im = ax.imshow(np.minimum(F.T, vmax), origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi],
                       cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        probs = basin_probabilities(Y[i]).numpy()
        txt = f"$\\varepsilon$=({eps[0]:.2f}, {eps[1]:.2f})\npop " + " ".join(f"{v:.2f}" for v in probs[:3])
        if key[0] != "target":
            m = compare_basin_metrics(Y[nc * row], Y[i])
            if m.get("df_mae") is not None:
                txt += f"\n$\\Delta F$ err {m['df_mae']:.2f} $k_BT$"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7, color="white")
        title = f"target {key[1]}" if key[0] == "target" else labels[a.methods.index(key[0])]
        if row == 0 or key[0] == "target":
            ax.set_title(title, fontsize=9)
        ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["$-\\pi$", "0", "$\\pi$"], fontsize=7)
        ax.set_yticks([-np.pi, 0, np.pi]); ax.set_yticklabels(["$-\\pi$", "0", "$\\pi$"], fontsize=7)
        if row == nr - 1:
            ax.set_xlabel(r"$\theta_{123}$", fontsize=8)
        if col == 0:
            ax.set_ylabel(r"$\theta_{234}$", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.6, label=r"$F/k_BT$")
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf", bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
