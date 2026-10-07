"""Stage 4: loss landscape L(x) = d(P(.|x), G) for reachable targets G = P(.|x*), x* ~ prior (hidden).
Losses: 'kl'  = KL(G || P_x) over the 8 phi states;
        'mmd' = MMD^2 on the 6 backbone dihedrals (phi, psi) with the product von Mises kernel
                k(a, b) = prod_i exp(kappa (cos(a_i - b_i) - 1)), estimated with R random Fourier features
                (integer frequencies n_i ~ Skellam(kappa/2, kappa/2), exact for this kernel in expectation).
Both are evaluated on the reweighted reference frames with analytic gradients
(d E_x[f] / dx = -Cov_x(f, B) / kT). Multi-start L-BFGS from prior draws; distinct minima are counted by
loss value and by barriers along straight-line paths between minimizers."""
import argparse, glob, json, os, time
import numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import system as S
from rw import Frames

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="/mnt/c/Code/Github/CDM/runs/trialanine/prod_x0_w*.npz")
ap.add_argument("--thin", type=int, default=2)
ap.add_argument("--sigma", type=float, default=1.0, help="prior sd in kT (hidden designs and starts)")
ap.add_argument("--targets", type=int, default=3)
ap.add_argument("--starts", type=int, default=48)
ap.add_argument("--random", type=int, default=2000)
ap.add_argument("--kappa", type=float, default=2.0)
ap.add_argument("--R", type=int, default=256)
ap.add_argument("--l2", type=float, default=0.0, help="optional Gaussian-prior penalty weight (0 = pure loss)")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--tag", default="")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
a = ap.parse_args()
t0 = time.time()
F = Frames(glob.glob(a.frames), thin=a.thin)
kT, D = F.kT, S.D
rng = np.random.default_rng(a.seed)
ang = np.concatenate([F.phi, F.psi], 1)
nfreq = rng.poisson(a.kappa / 2, (a.R, 6)) - rng.poisson(a.kappa / 2, (a.R, 6))
proj = ang @ nfreq.T
Phi = np.concatenate([np.cos(proj), np.sin(proj)], 1) / np.sqrt(a.R)  # (N, 2R); k ~ Phi Phi^T
Bc = F.B


def moments(x, feats):
    w = F.weights(x)
    m = w @ feats
    mB = w @ Bc
    cov = (feats * w[:, None]).T @ Bc - np.outer(m, mB)  # (K, D)
    return m, cov, w


def loss_kl(x, G):
    w = F.weights(x)
    P = w @ F.onehot
    mB = w @ Bc
    mBs = (F.onehot * w[:, None]).T @ Bc / np.maximum(P[:, None], 1e-300)
    L = float((G * (np.log(G) - np.log(np.maximum(P, 1e-300)))).sum())
    g = (G @ mBs - mB) / kT  # dL/dx = sum_s G_s (E[B|s] - E[B]) / kT
    return L, g


def loss_mmd(x, gfeat):
    m, cov, _ = moments(x, Phi)
    r = m - gfeat
    return float(r @ r), -2 * (r @ cov) / kT


def make_obj(kind, tgt):
    f = (lambda x: loss_kl(x, tgt)) if kind == "kl" else (lambda x: loss_mmd(x, tgt))
    if a.l2 > 0:
        return lambda x: tuple(np.add(f(x), (a.l2 * 0.5 * (x / (a.sigma * kT)) @ (x / (a.sigma * kT)),
                                             a.l2 * x / (a.sigma * kT) ** 2)))
    return f


def path_barrier(obj, xa, xb, n=25):
    t = np.linspace(0, 1, n)
    L = np.array([obj((1 - s) * xa + s * xb)[0] for s in t])
    return float(L.max() - max(L[0], L[-1])), L.tolist()


out = dict(args=vars(a), N=F.N, kT=kT, targets=[])
fig, axes = plt.subplots(2, a.targets, figsize=(4.2 * a.targets, 7.5), squeeze=False)
for ti in range(a.targets):
    xstar = rng.normal(0, a.sigma * kT, D)
    G = F.pops(xstar)
    gfeat = F.weights(xstar) @ Phi
    trec = dict(xstar=xstar.tolist(), G=G.tolist(), ess_xstar=F.ess(xstar)[0], losses={})
    for li, kind in enumerate(["kl", "mmd"]):
        obj = make_obj(kind, G if kind == "kl" else gfeat)
        Xr = rng.normal(0, a.sigma * kT, (a.random, D))
        Lr = np.array([obj(x)[0] for x in Xr])
        starts = rng.normal(0, a.sigma * kT, (a.starts, D))
        sols = []
        for x0 in starts:
            r = minimize(obj, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, gtol=1e-10, ftol=1e-15))
            sols.append(dict(x=r.x, L=float(r.fun), nit=int(r.nit), ess=float(F.ess(r.x)[0]),
                             dist_kT=float(np.linalg.norm(r.x) / kT), gnorm=float(np.linalg.norm(r.jac)),
                             tv_to_G=float(0.5 * np.abs(F.pops(r.x) - G).sum())))
        Ls = np.array([s["L"] for s in sols])
        Lmin = Ls.min()
        tol = max(1e-6, 1e-3 * np.median(Lr))
        glob_ok = Ls <= Lmin + tol
        # barriers between (up to 12) global-level minimizers, and from every non-global minimizer to the best
        gi = np.where(glob_ok)[0][:12]
        bars = []
        for i in range(len(gi)):
            for j in range(i + 1, len(gi)):
                b, _ = path_barrier(obj, sols[gi[i]]["x"], sols[gi[j]]["x"])
                bars.append(b)
        best = int(np.argmin(Ls))
        loc = []
        for i in np.where(~glob_ok)[0]:
            b, _ = path_barrier(obj, sols[i]["x"], sols[best]["x"])
            loc.append(dict(L=sols[i]["L"], barrier=b, gnorm=sols[i]["gnorm"]))
        # distinct non-global minima by loss value (rel. 1e-3) with a positive barrier
        nonglob = sorted({round(q["L"] / max(Lmin, 1e-12), 3) for q in loc if q["barrier"] > tol})
        Xg = np.array([sols[i]["x"] for i in np.where(glob_ok)[0]])
        spread = float(np.sqrt(np.mean(np.sum((Xg - Xg.mean(0)) ** 2, 1))) / kT) if len(Xg) > 1 else 0.0
        sv = np.linalg.svd((Xg - Xg.mean(0)) / kT, compute_uv=False).tolist() if len(Xg) > 1 else []
        rec = dict(L_random_median=float(np.median(Lr)), L_random_p05=float(np.percentile(Lr, 5)),
                   L_min=float(Lmin), n_starts=a.starts, n_global=int(glob_ok.sum()),
                   n_nonglobal=int((~glob_ok).sum()), nonglobal=loc, distinct_nonglobal_levels=len(nonglob),
                   max_barrier_between_global=float(max(bars)) if bars else 0.0,
                   median_barrier_between_global=float(np.median(bars)) if bars else 0.0,
                   global_rms_spread_kT=spread, global_singular_values_kT=sv,
                   dist_xstar_to_globals_kT=[float(np.linalg.norm(s["x"] - xstar) / kT) for s in sols if s["L"] <= Lmin + tol],
                   tv_globals=[s["tv_to_G"] for s in sols if s["L"] <= Lmin + tol],
                   ess_globals=[s["ess"] for s in sols if s["L"] <= Lmin + tol],
                   iters=[s["nit"] for s in sols], tol=tol)
        trec["losses"][kind] = rec
        ax = axes[li, ti]
        ax.hist(np.log10(Lr + 1e-12), bins=60, alpha=0.6, label="random x ~ prior")
        ax.hist(np.log10(Ls + 1e-12), bins=60, alpha=0.8, label="after L-BFGS")
        ax.set_title(f"target {ti} {kind}: {int(glob_ok.sum())}/{a.starts} reach min")
        ax.set_xlabel("log10 L"); ax.legend(fontsize=7)
        print(ti, kind, json.dumps({k: v for k, v in rec.items() if k not in ("nonglobal", "iters", "dist_xstar_to_globals_kT", "tv_globals", "ess_globals")}), flush=True)
    out["targets"].append(trec)
out["wall_s"] = time.time() - t0
fig.tight_layout(); fig.savefig(os.path.join(a.out, f"landscape{a.tag}.png"), dpi=130)
json.dump(out, open(os.path.join(a.out, f"landscape{a.tag}.json"), "w"), indent=1)
