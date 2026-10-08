"""2-D slice of the state-population loss KL(G || P_x) through three of its global minimizers
(the hidden design x* and two multi-start L-BFGS solutions), to show the shape of the solution set."""
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
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--box", type=float, default=4.0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
a = ap.parse_args()
F = Frames(glob.glob(a.frames), thin=a.thin); kT = F.kT
rng = np.random.default_rng(a.seed)
xstar = rng.normal(0, kT, S.D); G = F.pops(xstar)


def obj(x):
    w = F.weights(x); P = w @ F.onehot; mB = w @ F.B
    mBs = (F.onehot * w[:, None]).T @ F.B / np.maximum(P[:, None], 1e-300)
    return float((G * (np.log(G) - np.log(np.maximum(P, 1e-300)))).sum()), (G @ mBs - mB) / kT


sols = []
for _ in range(12):
    r = minimize(obj, rng.normal(0, kT, S.D), jac=True, method="L-BFGS-B", bounds=[(-a.box * kT, a.box * kT)] * S.D)
    sols.append(r.x)
d = [np.linalg.norm(s - xstar) for s in sols]
xa = xstar; xb = sols[int(np.argmax(d))]
xc = sols[int(np.argmax([np.linalg.norm(s - xa) + np.linalg.norm(s - xb) for s in sols]))]
e1 = (xb - xa); e1 /= np.linalg.norm(e1)
e2 = (xc - xa) - ((xc - xa) @ e1) * e1; e2 /= np.linalg.norm(e2)
pts = np.array([[(p - xa) @ e1, (p - xa) @ e2] for p in (xa, xb, xc)]) / kT
lo, hi = pts.min(0) - 3, pts.max(0) + 3
u = np.linspace(lo[0], hi[0], 61); v = np.linspace(lo[1], hi[1], 61)
L = np.array([[obj(xa + kT * (uu * e1 + vv * e2))[0] for uu in u] for vv in v])
fig, ax = plt.subplots(figsize=(6, 5))
cs = ax.contourf(u, v, np.log10(L + 1e-6), 30, cmap="viridis"); plt.colorbar(cs, label="log10 KL(G || P_x)")
ax.plot(*pts.T, "r*", ms=12)
for p, nm in zip(pts, ["x*", "min A", "min B"]):
    ax.annotate(nm, p, color="w")
ax.set_xlabel("e1 (kT)"); ax.set_ylabel("e2 (kT)"); ax.set_title("state-population loss, plane through 3 global minimizers")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "slice_kl.png"), dpi=130)


bands = {"x*-A": band_path(obj, xa, xb, kT), "x*-B": band_path(obj, xa, xc, kT), "A-B": band_path(obj, xb, xc, kT)}
print(bands)
json.dump(dict(bands=bands, xstar=xstar.tolist(), G=G.tolist(), pts_kT=pts.tolist(), dist_kT=(np.array(d) / kT).tolist(),
               losses_at_minimizers=[obj(s)[0] for s in sols], L_grid_min=float(L.min()), L_grid_max=float(L.max())),
          open(os.path.join(a.out, "slice_kl.json"), "w"), indent=1)
print("dist of minimizers to x* (kT):", np.round(np.array(d) / kT, 2))
