#!/usr/bin/env python
"""Trajectories of the Figure 1 toy search (candidate panel (e)): the clean estimate x0_hat(x_t) of every
restart at every reverse step, for MLGD (conditional diffusion sampler) and MLGD-F (consistency sampler).

Reruns the guided search of fig1_toy_example.py (its functions, imported; same setting and restarts) with
`record_x` on, so every step's state x_t is kept; x0_hat at each step is the prior's Tweedie estimate of
that state. Saves the trajectories (<out-dir>/trajectories.npz + .json) and draws:

  traj_steps.png    step on the x-axis (noise -> end), x0_hat on the y-axis, one panel per method
  traj_sharedx.png  x0_hat on the x-axis (the x-axis of Figure 1), step going down
  option_a.png      joint (a) / loss landscape (b) / trajectories (e) stacked on one x-axis, from this run

    python experiments/synthetic/pipeline/fig1_trajectories.py --device cpu \
        --compare <published fig1_hard_sim.json> --out-dir experiments/synthetic/results/fig1_traj
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig1_toy_example as f1   # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def record_cfg():
    """fig1's MLGD configuration with record_x on (x_t after every reverse step in the trace)."""
    orig = f1._mlgd_cfg

    def cfg(*a, **k):
        c = orig(*a, **k)
        c.record_x = True
        return c
    f1._mlgd_cfg = cfg


@torch.no_grad()
def tweedie(uncond, xs, ts, device):
    """x0_hat of states xs [S] at timesteps ts [S] (the prior's DDIM prediction; deterministic)."""
    out = []
    for x, t in zip(xs, ts):
        _, x0 = uncond.sample_ddim_step(torch.tensor([[x]], dtype=torch.float32), int(t), condition_x=None,
                                        device=device, eta=0.0)
        out.append(float(x0.view(-1)[0]))
    return np.array(out)


def run(args):
    record_cfg()
    sys.argv = ["fig1", "--setting", "hard", "--device", args.device, "--seed", "42", "--n-restarts", str(args.n_restarts),
                "--n-cond", "250", "--methods", *args.methods, "--ckpt-dir", os.path.join(REPO, "checkpoints", "toy"),
                "--uncond", "load", "--cond", "load", "--cm", "load", "--out", os.path.join(args.out_dir, "unused.png")]
    fa = f1.parse_args()
    st = f1.make_setting("hard")
    uncond, cond, cm, _ = f1.get_models(st, fa)
    out = {"x_star": float(st.x_star[0]), "methods": {}}
    for name in args.methods:
        use_cm = name.endswith("-F")
        res = f1.run_mlgd_restarts(st, uncond, cm if use_cm else cond, use_cm, fa)
        R = []
        for tr, rs in zip(res["traces"], res["restarts"]):
            ts = [e["t"] for e in tr]
            xt = np.array([e["x_t"][0] for e in tr])                 # state after step k, at timestep ts[k+1]
            x0 = tweedie(uncond, xt[:-1], ts[1:], args.device)        # the last state is the returned x
            R.append({"t": ts, "x_t": xt.tolist(), "x0_hat": x0.tolist() + [float(xt[-1])],
                      "x_hat": rs["x_hat"], "final_loss": rs["final_loss"], "l2_gmm": rs["l2_gmm"],
                      "loss": [e["loss"] for e in tr]})
        sel = int(np.argmin([r["final_loss"] for r in R]))
        out["methods"][name] = {"restarts": R, "selected": sel, "x_hat": res["x_hat"], "l2_gmm": res["l2_gmm"],
                                "hits_xstar_basin": res["hits_xstar_basin"]}
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(args.out_dir, "trajectories.json"), "w"))
    np.savez(os.path.join(args.out_dir, "trajectories.npz"),
             **{f"{m}_x0hat": np.array([r["x0_hat"] for r in d["restarts"]]) for m, d in out["methods"].items()})
    return out, st


def draw(out, st, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    xs = out["x_star"]
    decoys = sorted({round(float(m[0]), 2) for m in st.mu_list} - {round(xs, 2)})
    col = {"MLGD": "tab:red", "MLGD-F": "tab:green"}
    xlo, xhi = st.x_range

    def lines(ax, name, d, shared):
        for k, r in enumerate(d["restarts"]):
            y = np.array(r["x0_hat"]); s = np.arange(len(y))
            sel = k == d["selected"]
            kw = dict(color=col[name], lw=1.6 if sel else 0.7, alpha=1.0 if sel else 0.45, zorder=3 if sel else 2)
            ax.plot(y, s, **kw) if shared else ax.plot(s, y, **kw)

    # (i) conventional: step -> x-axis
    names = list(out["methods"])
    fig, axes = plt.subplots(1, len(names), figsize=(2.3 * len(names), 1.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        lines(ax, name, out["methods"][name], shared=False)
        ax.axhline(xs, color="black", ls="--", lw=1)
        for dx in decoys:
            ax.axhline(dx, color="0.6", ls=":", lw=0.8)
        ax.set_title(name, fontsize=8); ax.set_xlabel("reverse step"); ax.set_ylim(xlo, xhi)
    axes[0].set_ylabel(r"$\hat{x}_0(x_t)$")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "traj_steps.png"), dpi=300); plt.close(fig)

    # (ii) shared x: x0_hat -> x-axis, step going down
    fig, axes = plt.subplots(len(names), 1, figsize=(2.3, 1.5 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        lines(ax, name, out["methods"][name], shared=True)
        ax.axvline(xs, color="black", ls="--", lw=1)
        ax.invert_yaxis(); ax.set_ylabel("step"); ax.set_title(name, fontsize=8); ax.set_xlim(xlo, xhi)
    axes[-1].set_xlabel(r"$\hat{x}_0(x_t)$")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "traj_sharedx.png"), dpi=300); plt.close(fig)

    # Option A: joint (a) / loss (b) / trajectories (e) on one x-axis, from this run
    fig, (a, b, e) = plt.subplots(3, 1, figsize=(2.6, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 0.8, 1.2], hspace=0.08))
    f1.set_seed(0)
    xy = f1.generate_mog_samples_not_differentiable(4000, st.mu_list, st.Sigma_list, st.alpha).cpu().numpy()
    a.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.3, color="steelblue")
    xg = torch.linspace(xlo, xhi, 601)
    G = tuple(g.detach() for g in f1.cond_params(st, st.x_star[0]))
    L = f1.l2_1d(f1.cond_params(st, xg), G).numpy()
    b.plot(xg.numpy(), L, color="0.25", lw=1.4)
    for name, d in out["methods"].items():
        for ax in (a, b):
            ax.axvline(d["x_hat"], color=col[name], lw=2.5 if name == "MLGD" else 1.2, alpha=0.7 if name == "MLGD" else 1)
        lines(e, name, d, shared=True)
    for ax in (a, b, e):
        ax.axvline(xs, color="black", ls="--", lw=1)
    e.invert_yaxis(); e.set_xlim(xlo, xhi)
    a.set_ylabel("y"); b.set_ylabel(r"$\mathcal{L}(x)$"); e.set_ylabel("reverse step"); e.set_xlabel("x")
    for ax, l in zip((a, b, e), "abe"):
        ax.text(0.02, 0.95, f"({l})", transform=ax.transAxes, fontweight="bold", va="top")
    fig.savefig(os.path.join(out_dir, "option_a.png"), dpi=300, bbox_inches="tight"); plt.close(fig)


def compare(out, path):
    pub = json.load(open(path))
    lines = []
    for m, d in out["methods"].items():
        if m not in pub:
            continue
        p = pub[m]
        lines.append(f"{m}: published x_hat {p['x_hat']:+.3f} (L2 {p['l2_gmm']:.2e}, {p['hits_xstar_basin']}/10 in x* basin) | "
                     f"rerun x_hat {d['x_hat']:+.3f} (L2 {d['l2_gmm']:.2e}, {d['hits_xstar_basin']}/10 in x* basin)")
        lines.append(f"   published restarts: {[round(r['x_hat'], 2) for r in p['restarts']]}")
        lines.append(f"   rerun restarts:     {[round(r['x_hat'], 2) for r in d['restarts']]}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--methods", nargs="+", default=["MLGD", "MLGD-F"])
    p.add_argument("--n-restarts", type=int, default=10)
    p.add_argument("--out-dir", default=os.path.join(REPO, "experiments", "synthetic", "results", "fig1_traj"))
    p.add_argument("--compare", default=None, help="published fig1_hard_sim.json to compare the endpoints with")
    p.add_argument("--plot-only", action="store_true", help="redraw from <out-dir>/trajectories.json")
    a = p.parse_args()
    if a.plot_only:
        out, st = json.load(open(os.path.join(a.out_dir, "trajectories.json"))), f1.make_setting("hard")
    else:
        out, st = run(a)
    draw(out, st, a.out_dir)
    if a.compare:
        txt = compare(out, a.compare)
        print(txt)
        open(os.path.join(a.out_dir, "comparison.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
