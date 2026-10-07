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
ap.add_argument("--losses", default="kl,mmd,klcount")
ap.add_argument("--box", type=float, default=0.0, help="L-BFGS-B bounds |x_j| <= box * kT (0 = none)")
ap.add_argument("--xstar_npy", default=None, help="(T, 12) hidden designs in kJ/mol instead of prior draws")
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


NPLUS = np.eye(4)[np.array([bin(k).count("1") for k in range(8)])]  # 8 states -> number of '+' residues


def loss_kl_count(x, Gc):
    """KL(Gc || P_x) over the number of residues in the phi ~ +60 basin (0..3): a label-free observable."""
    w = F.weights(x)
    oc = F.onehot @ NPLUS
    P = w @ oc
    mB = w @ Bc
    mBs = (oc * w[:, None]).T @ Bc / np.maximum(P[:, None], 1e-300)
    return float((Gc * (np.log(Gc) - np.log(np.maximum(P, 1e-300)))).sum()), (Gc @ mBs - mB) / kT


def loss_mmd(x, gfeat):
    m, cov, _ = moments(x, Phi)
    r = m - gfeat
    return float(r @ r), -2 * (r @ cov) / kT


def make_obj(kind, tgt):
    f = {"kl": lambda x: loss_kl(x, tgt), "mmd": lambda x: loss_mmd(x, tgt),
         "klcount": lambda x: loss_kl_count(x, tgt)}[kind]
    if a.l2 > 0:
        return lambda x: tuple(np.add(f(x), (a.l2 * 0.5 * (x / (a.sigma * kT)) @ (x / (a.sigma * kT)),
                                             a.l2 * x / (a.sigma * kT) ** 2)))
    return f


def band_barrier(obj, xa, xb, n=16, k=1e-3):
    """Elastic band between two minimizers: minimize sum_k L(x_k) + k/2 sum |x_{k+1} - x_k|^2 / kT^2 over the
    interior points (endpoints fixed), from the straight line. Returns the max loss along the relaxed band
    minus the larger endpoint loss (about 0 when both lie in one connected low-loss set)."""
    t = np.linspace(0, 1, n)[1:-1, None]
    z0 = ((1 - t) * xa + t * xb).ravel()

    def f(z):
        X = np.vstack([xa, z.reshape(-1, D), xb])
        vals, grads = zip(*[obj(x) for x in X[1:-1]])
        dX = np.diff(X, axis=0) / kT
        E = sum(vals) + 0.5 * k * (dX ** 2).sum()
        g = np.array(grads) + k * (dX[:-1] - dX[1:]) / kT
        return E, g.ravel()
    bnd = [(-a.box * kT, a.box * kT)] * len(z0) if a.box > 0 else None
    r = minimize(f, z0, jac=True, method="L-BFGS-B", bounds=bnd, options=dict(maxiter=500))
    X = np.vstack([xa, r.x.reshape(-1, D), xb])
    L = np.array([obj(x)[0] for x in X])
    return float(L.max() - max(L[0], L[-1]))


def path_barrier(obj, xa, xb, n=25):
    t = np.linspace(0, 1, n)
    L = np.array([obj((1 - s) * xa + s * xb)[0] for s in t])
    return float(L.max() - max(L[0], L[-1])), L.tolist()


out = dict(args=vars(a), N=F.N, kT=kT, targets=[])
XSTAR = np.load(a.xstar_npy) if a.xstar_npy else None
if XSTAR is not None:
    a.targets = len(XSTAR)
LOSSES = a.losses.split(",")
fig, axes = plt.subplots(len(LOSSES), a.targets, figsize=(4.2 * a.targets, 3.7 * len(LOSSES)), squeeze=False)
for ti in range(a.targets):
    xstar = rng.normal(0, a.sigma * kT, D)
    if XSTAR is not None:
        xstar = XSTAR[ti]
    G = F.pops(xstar)
    gfeat = F.weights(xstar) @ Phi
    trec = dict(xstar=xstar.tolist(), G=G.tolist(), ess_xstar=F.ess(xstar)[0], losses={})
    trec["G_count"] = (G @ NPLUS).tolist()
    for li, kind in enumerate(LOSSES):
        obj = make_obj(kind, {"kl": G, "mmd": gfeat, "klcount": G @ NPLUS}[kind])
        Xr = rng.normal(0, a.sigma * kT, (a.random, D))
        Lr = np.array([obj(x)[0] for x in Xr])
        starts = rng.normal(0, a.sigma * kT, (a.starts, D))
        sols = []
        for x0 in starts:
            bnd = [(-a.box * kT, a.box * kT)] * D if a.box > 0 else None
            if bnd:
                x0 = np.clip(x0, -a.box * kT, a.box * kT)
            r = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bnd, options=dict(maxiter=3000, gtol=1e-10, ftol=1e-15))
            sols.append(dict(x=r.x, L=float(r.fun), nit=int(r.nit), ess=float(F.ess(r.x)[0]),
                             dist_kT=float(np.linalg.norm(r.x) / kT), gnorm=float(np.linalg.norm(r.jac)),
                             at_box=int(np.sum(np.abs(r.x) >= a.box * kT * (1 - 1e-6))) if a.box > 0 else 0,
                             tv_to_G=float(0.5 * np.abs(F.pops(r.x) - G).sum())))
        Ls = np.array([s["L"] for s in sols])
        Lmin = Ls.min()
        tol = max(1e-6, 1e-3 * np.median(Lr))
        glob_ok = Ls <= Lmin + tol
        # connected components of the minimizers: i ~ j if the straight line between them stays within tol of
        # the larger endpoint loss; components that remain separate are tested with an elastic band.
        idx = list(np.argsort(Ls)[:24])
        parent = {i: i for i in idx}
        find = lambda i: i if parent[i] == i else find(parent[i])
        bars, bands = [], []
        for ii, i in enumerate(idx):
            for j in idx[ii + 1:]:
                if find(i) == find(j):
                    continue
                b, _ = path_barrier(obj, sols[i]["x"], sols[j]["x"])
                bars.append(b)
                if b <= tol:
                    parent[find(j)] = find(i)
        reps = sorted({find(i) for i in idx})
        for ii, i in enumerate(reps):
            for j in reps[ii + 1:]:
                if find(i) == find(j):
                    continue
                bb = band_barrier(obj, sols[i]["x"], sols[j]["x"])
                bands.append(dict(Li=sols[i]["L"], Lj=sols[j]["L"], band_barrier=bb))
                if bb <= tol:
                    parent[find(j)] = find(i)
        comps = {}
        for i in idx:
            comps.setdefault(find(i), []).append(i)
        comp_list = sorted([dict(size=len(v), L_min=float(min(Ls[v])), L_max=float(max(Ls[v])),
                                 x_mean_kT=(np.mean([sols[k]["x"] for k in v], 0) / kT).round(2).tolist())
                            for v in comps.values()], key=lambda c: c["L_min"])
        best = int(np.argmin(Ls))
        loc = [dict(L=sols[i]["L"], gnorm=sols[i]["gnorm"]) for i in np.where(~glob_ok)[0]]
        Xg = np.array([sols[i]["x"] for i in np.where(glob_ok)[0]])
        spread = float(np.sqrt(np.mean(np.sum((Xg - Xg.mean(0)) ** 2, 1))) / kT) if len(Xg) > 1 else 0.0
        sv = np.linalg.svd((Xg - Xg.mean(0)) / kT, compute_uv=False).tolist() if len(Xg) > 1 else []
        rec = dict(L_random_median=float(np.median(Lr)), L_random_p05=float(np.percentile(Lr, 5)),
                   L_min=float(Lmin), n_starts=a.starts, n_global=int(glob_ok.sum()),
                   n_nonglobal=int((~glob_ok).sum()), nonglobal=loc,
                   n_components=len(comp_list), components=comp_list,
                   max_line_barrier=float(max(bars)) if bars else 0.0,
                   band_barriers=bands, max_band_barrier=float(max([q["band_barrier"] for q in bands])) if bands else 0.0,
                   global_rms_spread_kT=spread, global_singular_values_kT=sv,
                   dist_xstar_to_globals_kT=[float(np.linalg.norm(s["x"] - xstar) / kT) for s in sols if s["L"] <= Lmin + tol],
                   tv_globals=[s["tv_to_G"] for s in sols if s["L"] <= Lmin + tol],
                   ess_globals=[s["ess"] for s in sols if s["L"] <= Lmin + tol],
                   ess_all_min=[s["ess"] for s in sols], at_box=[s["at_box"] for s in sols],
                   iters=[s["nit"] for s in sols], tol=tol)
        trec["losses"][kind] = rec
        ax = axes[li, ti]
        ax.hist(np.log10(Lr + 1e-12), bins=60, alpha=0.6, label="random x ~ prior")
        ax.hist(np.log10(Ls + 1e-12), bins=60, alpha=0.8, label="after L-BFGS")
        ax.set_title(f"target {ti} {kind}: {int(glob_ok.sum())}/{a.starts} reach min")
        ax.set_xlabel("log10 L"); ax.legend(fontsize=7)
        print("   components:", [(c["size"], round(c["L_min"], 5)) for c in comp_list], flush=True)
        print(ti, kind, json.dumps({k: v for k, v in rec.items() if k not in ("nonglobal", "iters", "dist_xstar_to_globals_kT", "tv_globals", "ess_globals", "band_barriers", "components", "ess_all_min", "at_box")}), flush=True)
    out["targets"].append(trec)
out["wall_s"] = time.time() - t0
fig.tight_layout(); fig.savefig(os.path.join(a.out, f"landscape{a.tag}.png"), dpi=130)
json.dump(out, open(os.path.join(a.out, f"landscape{a.tag}.json"), "w"), indent=1)
