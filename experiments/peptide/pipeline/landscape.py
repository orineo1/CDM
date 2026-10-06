"""Stage 2 of the rugged model: the true loss landscape around hidden targets, by simulation only (no networks).

For each of --n-targets target designs x* drawn from the design's prior, simulates the target ensemble and
  (a) --n-random designs drawn uniformly from the prior (random-search hardness),
  (b) --n-lines straight lines from random designs to x*, --n-points points each (barriers = separate minima),
  (c) --n-pert designs at each of --radii around x* (in standardized units of X; the smoothed loss / funnel),
and scores every design by the true loss: the energy distance between its ensemble and the target's on the
standardized features (common.raw_to_feat, scaled by the target's feature std), plus the basin TV
(common.basin_probs). The noise floor is the same loss between two halves of the target's replicas.
One array task per target (--target-id), outputs one JSON per target; --analyze summarizes and plots.

    python experiments/peptide/pipeline/landscape.py --n-beads 8 --free sigma,eta --fixed eps=0.5 \
        --target-id 0 --out runs/rugged/landscape/t000.json
    python experiments/peptide/pipeline/landscape.py --analyze runs/rugged/landscape --out-md report.md
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common   # noqa: E402
from simulate import simulate, Design   # noqa: E402


def energy_distance(A, B, n=2048, gen=None):
    ia = torch.randperm(A.shape[0], generator=gen)[:n]
    ib = torch.randperm(B.shape[0], generator=gen)[:n]
    A, B = A[ia].double(), B[ib].double()
    return float(2 * torch.cdist(A, B).mean() - torch.cdist(A, A).mean() - torch.cdist(B, B).mean())


def tv(p, q):
    return 0.5 * float((p - q).abs().sum())


def score_all(Ys, T, gen):
    """energy distance (standardized by the target's features) and basin TV of every ensemble against T"""
    fT = common.raw_to_feat(T)
    mu, sd = fT.mean(0), fT.std(0).clamp_min(1e-6)
    zT = (fT - mu) / sd
    pT = common.basin_probs(T)
    out = []
    for Y in Ys:
        out.append({"ed": energy_distance((common.raw_to_feat(Y) - mu) / sd, zT, gen=gen), "tv": tv(common.basin_probs(Y), pT)})
    return out


def run(a):
    design = Design.from_args(a.n_beads, a.free, a.fixed, a.box, a.prior)
    common.set_design(design)
    g = torch.Generator().manual_seed(a.seed)
    xstars = design.sample(a.n_targets, generator=g)
    xs = xstars[a.target_id]
    gt = torch.Generator().manual_seed(a.seed + 1000 + a.target_id)
    std = (design.from_std(torch.ones(design.dim_x)) - design.from_std(torch.zeros(design.dim_x)))
    rand = design.sample(a.n_random, generator=gt)
    starts = design.sample(a.n_lines, generator=gt)
    lam = torch.linspace(0, 1, a.n_points)
    lines = torch.stack([(1 - l) * starts + l * xs for l in lam], 1).reshape(-1, design.dim_x)     # [lines*points, K]
    pert = []
    for r in a.radii:
        u = torch.randn(a.n_pert, design.dim_x, generator=gt)
        pert.append(design.clip(xs + r * std * u))
    pert = torch.cat(pert) if pert else torch.zeros(0, design.dim_x)
    sim = dict(n_steps=a.n_steps, stride=a.stride, burn=a.burn, design=design, temp=a.temp, verbose=False)
    t0 = time.time()
    T = simulate(xs.view(1, -1), n_rep=a.n_rep_target, seed=a.seed + 7, **sim)[0]
    h = T.shape[0] // 2
    gs = torch.Generator().manual_seed(0)
    floor = score_all([T[:h]], T[h:], gs)[0]
    X = torch.cat([rand, lines, pert])
    Ys = []
    for c in range(0, X.shape[0], a.chunk):
        Ys += list(simulate(X[c:c + a.chunk], n_rep=a.n_rep, seed=a.seed + 11 + c, **sim))
        print(f"[landscape] t{a.target_id}: {min(c + a.chunk, X.shape[0])}/{X.shape[0]} designs, {time.time() - t0:.0f}s", flush=True)
    sc = score_all(Ys, T, gs)
    nr, nl = rand.shape[0], lines.shape[0]
    res = {"args": vars(a), "design": design.to_dict(), "target_id": a.target_id, "x_star": xs.tolist(), "floor": floor,
           "random": {"x": rand.tolist(), "score": sc[:nr]},
           "lines": {"start": starts.tolist(), "lam": lam.tolist(), "score": [sc[nr + i * a.n_points: nr + (i + 1) * a.n_points] for i in range(a.n_lines)]},
           "pert": {"radii": list(a.radii), "score": [sc[nr + nl + i * a.n_pert: nr + nl + (i + 1) * a.n_pert] for i in range(len(a.radii))]},
           "time_s": time.time() - t0}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(f"[landscape] wrote {a.out} ({time.time() - t0:.0f}s)")


def analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(a.analyze, "t*.json")))
    lines_md = [f"Landscape summary of {a.analyze} ({len(files)} targets)", "",
                "| target | floor ED (target halves) | noise ED (design at x*) | floor TV | random: ED<2 noise | ED<5 noise | min ED | lines with a barrier | pert ED by radius |",
                "|---|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for f in files:
        d = json.load(open(f))
        fl = max(d["floor"]["ed"], 1e-4)
        ed = np.array([s["ed"] for s in d["random"]["score"]])
        # noise of one design's loss at this replica count: the designs closest to x* (smallest radius)
        noise = max(float(np.mean([s["ed"] for s in d["pert"]["score"][0]])), fl)
        nb = 0
        for li, sc in enumerate(d["lines"]["score"]):
            e = np.array([s["ed"] for s in sc])
            e = np.array([np.median(e[max(0, i - 1):i + 2]) for i in range(len(e))])     # 3-point median
            # barrier: a rise of more than 3 noise levels after an earlier minimum on the way to x*
            run_min = np.minimum.accumulate(e)
            if np.any(e - run_min > 3 * noise):
                nb += 1
            if d["target_id"] == 0:
                axes[0].plot(d["lines"]["lam"], e, lw=0.8)
        pe = [np.mean([s["ed"] for s in sc]) for sc in d["pert"]["score"]]
        axes[1].plot(d["pert"]["radii"], pe, marker="o", lw=0.8, label=f"t{d['target_id']}")
        lines_md.append(f"| {d['target_id']} | {fl:.4f} | {noise:.4f} | {d['floor']['tv']:.3f} | {np.mean(ed < 2 * noise):.3f} | "
                        f"{np.mean(ed < 5 * noise):.3f} | {ed.min():.4f} | {nb}/{len(d['lines']['score'])} | "
                        + ", ".join(f"{r:g}:{v:.3f}" for r, v in zip(d["pert"]["radii"], pe)) + " |")
    axes[0].set_xlabel("position on line (0 = random start, 1 = x*)"); axes[0].set_ylabel("energy distance"); axes[0].set_title("line scans, target 0")
    axes[1].set_xlabel("perturbation radius (std of X)"); axes[1].set_ylabel("mean energy distance"); axes[1].set_title("loss around x*")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    png = os.path.splitext(a.out_md)[0] + ".png"
    fig.savefig(png, dpi=150)
    open(a.out_md, "w").write("\n".join(lines_md) + f"\n\n![landscape]({os.path.basename(png)})\n")
    print("\n".join(lines_md))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analyze", default=None, help="directory of per-target JSONs to summarize")
    p.add_argument("--out-md", default=None)
    p.add_argument("--n-beads", type=int, default=8)
    p.add_argument("--free", default="sigma,eta")
    p.add_argument("--fixed", default="eps=0.5")
    p.add_argument("--box", default=None)
    p.add_argument("--prior", default="uniform")
    p.add_argument("--temp", type=float, default=None)
    p.add_argument("--n-targets", type=int, default=5)
    p.add_argument("--target-id", type=int, default=0)
    p.add_argument("--n-random", type=int, default=500)
    p.add_argument("--n-lines", type=int, default=10)
    p.add_argument("--n-points", type=int, default=20)
    p.add_argument("--n-pert", type=int, default=20)
    p.add_argument("--radii", type=float, nargs="*", default=[0.02, 0.05, 0.1, 0.2, 0.4])
    p.add_argument("--n-rep", type=int, default=16)
    p.add_argument("--n-rep-target", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=4_000_000)
    p.add_argument("--burn", type=int, default=400_000)
    p.add_argument("--stride", type=int, default=400)
    p.add_argument("--chunk", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    analyze(a) if a.analyze else run(a)


if __name__ == "__main__":
    main()
