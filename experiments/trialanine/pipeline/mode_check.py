"""Label-free target check: G_count = distribution of the number of residues in the phi ~ +60 basin.
Multi-start L-BFGS on KL(G_count || P_count(x)); minimizers are grouped by which residue carries the '+'
population; groups are tested with a long elastic band (32 images) and a 2-D slice is plotted."""
import argparse, glob, json, os
import numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import system as S
from rw import Frames
from paths import band_path

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="runs/trialanine/prod_x0_w*.npz")
ap.add_argument("--thin", type=int, default=4)
ap.add_argument("--xstar_npy", default="runs/trialanine/xstar_eng2.npy")
ap.add_argument("--index", type=int, default=1)
ap.add_argument("--starts", type=int, default=32)
ap.add_argument("--box", type=float, default=6.0)
ap.add_argument("--sigma", type=float, default=2.0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
a = ap.parse_args()
F = Frames(glob.glob(a.frames), thin=a.thin); kT = F.kT; D = S.D
xstar = np.load(a.xstar_npy)[a.index]
NPLUS = np.eye(4)[np.array([bin(k).count("1") for k in range(8)])]
OC = F.onehot @ NPLUS
G = F.weights(xstar) @ OC
RES = np.array([[(k >> (2 - i)) & 1 for i in range(3)] for k in range(8)], float)  # state -> per-residue '+'


def obj(x):
    w = F.weights(x); P = w @ OC; mB = w @ F.B
    mBs = (OC * w[:, None]).T @ F.B / np.maximum(P[:, None], 1e-300)
    return float((G * (np.log(G) - np.log(np.maximum(P, 1e-300)))).sum()), (G @ mBs - mB) / kT


rng = np.random.default_rng(3)
bnd = [(-a.box * kT, a.box * kT)] * D
sols = []
for _ in range(a.starts):
    r = minimize(obj, np.clip(rng.normal(0, a.sigma * kT, D), -a.box * kT, a.box * kT), jac=True, method="L-BFGS-B",
                 bounds=bnd, options=dict(maxiter=5000, gtol=1e-12, ftol=1e-16))
    plus = F.pops(r.x) @ RES
    sols.append(dict(x=r.x, L=float(r.fun), plus=plus, mode=int(np.argmax(plus))))
modes = sorted({s["mode"] for s in sols})
reps = {m: min([s for s in sols if s["mode"] == m], key=lambda s: s["L"]) for m in modes}


res = dict(G_count=G.tolist(), xstar_kT=(xstar / kT).tolist(), n_starts=a.starts,
           modes={str(m): dict(count=sum(s["mode"] == m for s in sols), L_best=reps[m]["L"],
                               plus_marginals=reps[m]["plus"].tolist(), ess=float(F.ess(reps[m]["x"])[0]), N=F.N,
                               x_kT=(reps[m]["x"] / kT).round(2).tolist())
                  for m in modes}, bands={})
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i, m1 in enumerate(modes):
    for m2 in modes[i + 1:]:
        bp = band_path(obj, reps[m1]["x"], reps[m2]["x"], kT)
        res["bands"][f"{m1}-{m2}"] = bp
        s_ = np.linspace(0, 1, 101)
        ax[0].semilogy(s_, [obj((1 - q) * reps[m1]["x"] + q * reps[m2]["x"])[0] + 1e-12 for q in s_],
                       label=f"line res{m1+1}-res{m2+1}; band max {bp['band_max_dense']:.3g}")
ax[0].set_xlabel("straight-line path coordinate"); ax[0].set_ylabel("KL(G_count || P_count(x))"); ax[0].legend(fontsize=7)
if len(modes) >= 2:
    xa, xb = reps[modes[0]]["x"], reps[modes[1]]["x"]
    e1 = (xb - xa) / np.linalg.norm(xb - xa); c = rng.normal(size=D); e2 = c - (c @ e1) * e1; e2 /= np.linalg.norm(e2)
    if len(modes) >= 3:
        xc = reps[modes[2]]["x"]; e2 = (xc - xa) - ((xc - xa) @ e1) * e1; e2 /= np.linalg.norm(e2)
    L0 = np.linalg.norm(xb - xa) / kT
    pr = np.array([[(reps[m]["x"] - xa) @ e1, (reps[m]["x"] - xa) @ e2] for m in modes]) / kT
    u = np.linspace(pr[:, 0].min() - 0.3 * L0, pr[:, 0].max() + 0.3 * L0, 61)
    v = np.linspace(pr[:, 1].min() - 0.3 * L0, pr[:, 1].max() + 0.3 * L0, 61)
    Z = np.array([[obj(xa + kT * (uu * e1 + vv * e2))[0] for uu in u] for vv in v])
    cs = ax[1].contourf(u, v, np.log10(Z + 1e-6), 30, cmap="viridis"); plt.colorbar(cs, ax=ax[1], label="log10 loss")
    for m in modes:
        p = (reps[m]["x"] - xa) / kT
        ax[1].plot(p @ e1, p @ e2, "r*", ms=11); ax[1].annotate(f"res {m+1} '+'", (p @ e1, p @ e2), color="w")
    ax[1].set_xlabel("e1 (kT)"); ax[1].set_ylabel("e2 (kT)")
ax[1].set_title(f"label-free target G_count={np.round(G, 3).tolist()}", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(a.out, f"mode_check_{a.index}.png"), dpi=130)
json.dump(res, open(os.path.join(a.out, f"mode_check_{a.index}.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "xstar_kT"}, indent=0))
