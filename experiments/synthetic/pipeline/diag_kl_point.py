"""
Why does a search with --loss kl end at a given x_hat? Compares candidate points of one scenario by the
true TV and by the search losses (KDE-KL and MMD), each averaged over fresh conditional and target draws:
if the KL is lower at x_hat than at x*, the loss prefers the wrong point (a loss problem); if it is
higher, the search did not reach x* (a search problem). Also scans the KL along the segment from x* to
x_hat, and prints the target's and the conditional's mixture components in Y.

    python experiments/synthetic/pipeline/diag_kl_point.py --dir <runs>/x1_y1 \\
        --json MLGD_kl.json GBFESO-KL_main.json MLGD_kl3500.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cdm.guidance.core import _draw_conditional                     # noqa: E402
from cdm.losses.functions import MMDLoss, RBF                        # noqa: E402
from cdm.losses.kde_kl import KDEKLLoss                              # noqa: E402
from cdm.utils.dist import compute_conditionals, compute_alpha, filter_and_normalize   # noqa: E402
from make_table1 import scale_free_distances                         # noqa: E402
from run_optimization import load_model                              # noqa: E402
from search_mog import load_target                                   # noqa: E402
from types import SimpleNamespace                                    # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="scenario directory (data.pt, models/, the method JSONs)")
    p.add_argument("--json", nargs="+", default=["MLGD_kl.json", "GBFESO-KL_main.json"])
    p.add_argument("--n", type=int, nargs="+", default=[1000, 3500], help="samples per loss evaluation")
    p.add_argument("--reps", type=int, default=20, help="fresh draws per loss evaluation")
    p.add_argument("--kde-bandwidth", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev = a.device
    torch.manual_seed(0)

    data_path = os.path.join(a.dir, "data.pt")
    d = torch.load(data_path, map_location="cpu", weights_only=False)
    gen, info = load_target(data_path, dev)
    sampler, arch = load_model(os.path.join(a.dir, "models", "diffusion", "model_conditional.pt"), dev)
    for prm in sampler.parameters():
        prm.requires_grad_(False)
    mix = {"mu_list": [m.numpy() for m in d["mu_list"]], "raw": d}
    kl = KDEKLLoss(bandwidth=a.kde_bandwidth)
    mmd = MMDLoss(kernel=RBF(n_kernels=5, mul_factor=2.0, device=dev))

    def cfg(n):
        return SimpleNamespace(n_cond=n, repeat_condition=True, device=dev, cond_sample_kwargs={},
                               condition_on_slice=(arch == "diffusion"), cond_output_transform=None)

    @torch.no_grad()
    def losses(x, n):
        x = torch.as_tensor(x, dtype=torch.float32, device=dev).view(1, -1)
        k, m = [], []
        for _ in range(a.reps):
            y = _draw_conditional(sampler, x, cfg(n)); t = gen(n, dev).float()
            k.append(float(kl(y, t))); m.append(float(mmd(y, t)))
        return np.mean(k), np.std(k), np.mean(m), np.std(m)

    def components(x):
        x = torch.as_tensor(x, dtype=torch.float32).view(-1)
        mu, S = compute_conditionals(d["mu_list"], d["Sigma_list"], x)
        w = compute_alpha(d["mu_list"], d["Sigma_list"], d["alpha"], x)
        mu, S, w = filter_and_normalize(mu, S, w, threshold=1e-3)
        return "  ".join(f"w={float(wi):.2f} mean={np.round(mi.view(-1).numpy(), 3).tolist()} sd={np.round(Si.diagonal().sqrt().numpy(), 3).tolist()}"
                         for mi, Si, wi in zip(mu, S, w))

    x_star = np.array(info["x_star"].float().view(-1)) if info.get("x_star") is not None else None
    print(f"[diag] {a.dir}  x* = {None if x_star is None else np.round(x_star, 4).tolist()}  KDE h = {a.kde_bandwidth}")
    tm, tv_, tw = d["target_means"], d["target_variances"], d["target_weights"]
    print("  target G(Y):  " + "  ".join(f"w={float(w):.2f} mean={np.round(m.view(-1).numpy(), 3).tolist()} "
                                         f"sd={np.round((torch.as_tensor(v).diagonal() if torch.as_tensor(v).dim() == 2 else torch.as_tensor(v).reshape(-1)).sqrt().numpy(), 3).tolist()}"
                                         for m, v, w in zip(tm, tv_, tw)))
    if x_star is not None:
        print(f"  P(Y | x*):    {components(x_star)}")

    points = [("x*", x_star)] if x_star is not None else []
    for f in a.json:
        fp = os.path.join(a.dir, f)
        if not os.path.exists(fp):
            print(f"  (no {f})"); continue
        r = json.load(open(fp))
        for att in r["attempts"]:
            points.append((f"{f} att{att['attempt']} x_hat", np.array(att["x_hat"])))
            before = (att.get("refine") or {}).get("x_before")
            if before is not None:
                points.append((f"{f} att{att['attempt']} before refine", np.array(before).reshape(-1)))
        # every trajectory of the file: its own final loss and TV, to see whether a good one was available
        trs = [q for att in r["attempts"] for q in att.get("restarts", [att])]
        tvs = [scale_free_distances(mix, q["x_hat"], n_mc=4000, seed=i)["tv"] for i, q in enumerate(trs)]
        good = [(t, q["final_loss"]) for t, q in zip(tvs, trs) if t < 0.3]
        bad = [(t, q["final_loss"]) for t, q in zip(tvs, trs) if t >= 0.3]
        # one line per trajectory: is the failure one bad run, or all of them?
        print(f"  {f}: every trajectory (attempt / restart: |x_hat - x*|, TV, final loss; * = the attempt's answer)")
        k = 0
        for att in r["attempts"]:
            for j, q in enumerate(att.get("restarts", [att])):
                dist = float(np.linalg.norm(np.array(q["x_hat"]) - x_star)) if x_star is not None else float("nan")
                star = "*" if np.allclose(q["x_hat"], att["x_hat"]) else " "
                print(f"    att{att['attempt']} r{j}{star} |x-x*|={dist:.4f}  TV={tvs[k]:.3f}  final loss={q['final_loss']:.4f}")
                k += 1
        print(f"  {f}: {len(trs)} trajectories; TV<0.3: {len(good)} (mean final loss "
              f"{np.mean([l for _, l in good]) if good else float('nan'):.4f}), TV>=0.3: {len(bad)} "
              f"(mean final loss {np.mean([l for _, l in bad]) if bad else float('nan'):.4f})")

    print(f"\n  {'point':40s} {'TV':>6s} {'|x-x*|':>8s}" + "".join(f"   KL@{n} (sd)        MMD@{n} (sd)    " for n in a.n))
    for name, x in points:
        tv = scale_free_distances(mix, x.tolist())["tv"]
        dist = float(np.linalg.norm(x - x_star)) if x_star is not None else float("nan")
        row = f"  {name:40s} {tv:6.3f} {dist:8.4f}"
        for n in a.n:
            k, ks, m, ms = losses(x, n)
            row += f"   {k:8.4f} ({ks:.4f})  {m:8.4f} ({ms:.4f})"
        print(row)

    # the KL along the segment x* -> the first x_hat of the first file
    first = next(((nm, x) for nm, x in points if nm != "x*"), None)
    if x_star is not None and first is not None:
        print(f"\n  KL and TV along x* + s (x_hat - x*), x_hat = {first[0]}")
        for s in [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
            x = x_star + s * (first[1] - x_star)
            k, ks, m, ms = losses(x, a.n[0])
            print(f"    s={s:5.2f}  TV={scale_free_distances(mix, x.tolist())['tv']:.3f}  KL@{a.n[0]}={k:.4f} ({ks:.4f})  "
                  f"MMD@{a.n[0]}={m:.4f}   P(Y|x): {components(x)}")


if __name__ == "__main__":
    main()
