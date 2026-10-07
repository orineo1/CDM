"""Exact (reweighted) landscape of the LABEL-FREE ensemble loss used by the CDM benchmark: MMD between the
canonical (residue-sorted) dihedral features at x and at x*, Gaussian kernel with 3 bandwidths on the
standardized features, by random Fourier features. Multi-start L-BFGS in the box; minimizers are labeled by
the residue that carries the most '+' population; reports the loss level of each mode."""
import argparse, glob, json, os, time
import numpy as np
from scipy.optimize import minimize
import system as S
from rw import Frames

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="/mnt/c/Code/Github/CDM/runs/trialanine/prod_x0_w*.npz")
ap.add_argument("--thin", type=int, default=4)
ap.add_argument("--xstar_npy", required=True)
ap.add_argument("--starts", type=int, default=24)
ap.add_argument("--box", type=float, default=4.0)
ap.add_argument("--sigma", type=float, default=2.0)
ap.add_argument("--R", type=int, default=600)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.time()
F = Frames(glob.glob(a.frames), thin=a.thin); kT = F.kT; D = S.D
C = S.canonical_features(F.phi, F.psi); C = (C - C.mean(0)) / C.std(0)
rng = np.random.default_rng(0)
W = np.concatenate([rng.normal(0, 1 / h, (a.R // 3, 12)) for h in (0.5, 1.0, 2.0)])
Z = np.concatenate([np.cos(C @ W.T), np.sin(C @ W.T)], 1) / np.sqrt(len(W))
RES = np.array([[(k >> (2 - i)) & 1 for i in range(3)] for k in range(8)], float)
NPLUS = np.eye(4)[np.array([bin(k).count("1") for k in range(8)])]
out = dict(args=vars(a), targets=[])
for xs in np.load(a.xstar_npy):
    g = F.weights(xs) @ Z
    Gc = F.pops(xs) @ NPLUS

    def obj(x):
        w = F.weights(x); m = w @ Z; mB = w @ F.B
        cov = (Z * w[:, None]).T @ F.B - np.outer(m, mB); r = m - g
        return float(r @ r), -2 * (r @ cov) / kT
    sols = []
    for _ in range(a.starts):
        x0 = np.clip(rng.normal(0, a.sigma * kT, D), -a.box * kT, a.box * kT)
        r = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=[(-a.box * kT, a.box * kT)] * D,
                     options=dict(maxiter=2000, gtol=1e-12, ftol=1e-15))
        P = F.pops(r.x)
        sols.append(dict(L=float(r.fun), mode=int(np.argmax(P @ RES)), plus=(P @ RES).round(4).tolist(),
                         count_tv=float(0.5 * np.abs(P @ NPLUS - Gc).sum()), ess=float(F.ess(r.x)[0]),
                         dist_kT=float(np.linalg.norm(r.x - xs) / kT)))
    Pst = F.pops(xs)
    rec = dict(xstar_kT=(xs / kT).round(2).tolist(), mode_star=int(np.argmax(Pst @ RES)), plus_star=(Pst @ RES).round(4).tolist(),
               G_count=Gc.round(4).tolist(), L_rand_median=float(np.median([obj(np.clip(rng.normal(0, a.sigma * kT, D), -a.box * kT, a.box * kT))[0] for _ in range(100)])),
               modes={})
    for m in range(3):
        q = [s for s in sols if s["mode"] == m]
        if q:
            b = min(q, key=lambda s: s["L"])
            rec["modes"][str(m)] = dict(n=len(q), L_best=b["L"], L_median=float(np.median([s["L"] for s in q])),
                                        count_tv_best=b["count_tv"], plus=b["plus"], ess=b["ess"], dist_kT=b["dist_kT"])
    out["targets"].append(rec)
    print(json.dumps(rec), flush=True)
out["wall_s"] = time.time() - t0
json.dump(out, open(a.out, "w"), indent=1)
