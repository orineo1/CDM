"""True-loss landscape of a charge design by reweighting a pool of simulated ensembles (MBAR); ruggedness test.

With only the charges free (everything else fixed), the energy of a frame y under design q is
    U_q(y) = U_0(y) + A sum_{i<j} q_i q_j phi(r_ij(y)),   phi(r) = exp(-r / debye) / r  (r < rcut),
so the equilibrium ensemble of ANY q is a reweighting of frames simulated at other designs. MBAR on a pool
(sim_ensembles.py at K designs drawn from the prior) gives the free energies f_k and per-frame weights
w_n(q) ~ exp(-beta U_q(y_n) + beta U_0(y_n)) / sum_k N_k exp(f_k - beta U_k(y_n) + beta U_0(y_n)): the loss of any q
against a target is then a cheap, differentiable function of q (no networks, no new simulation).

Loss: MMD^2 with a multi-bandwidth Gaussian kernel (bandwidths 0.25, 0.5, 1, 2 x the median distance of the
pool's standardized features, common.raw_to_feat), by random Fourier features; self terms removed (U-statistic
form), so a small effective sample size (ESS) does not inflate it.

For each target (a design q* simulated directly, 64 replicas):
  * floor   : the loss between two halves of the target's replicas; MBAR check: the loss and the basin TV of the
              reweighted ensemble at q* against the target (both should be near the floor);
  * random  : the loss of --n-random prior draws (how rare good designs are);
  * descent : projected Adam on q from --n-starts prior draws (the ideal single-start gradient method, on the
              true loss); endpoints are clustered (q and -q have identical ensembles: distance min |a -+ b|), every
              cluster is a local minimum found by descent; 'solved' = the endpoint reaches the global basin
              (loss <= the loss at q* + --solve-tol x the median random loss);
  * barrier : MBAR loss along the straight line from each spurious minimum to the nearer of +-q*;
  * --confirm: the line scans of the best spurious minima re-simulated directly (no reweighting) and scored by the
              same MMD, energy distance and basin TV: direct evidence of the barrier.
Diagnostics (MBAR iterations and convergence, per-start loss trajectories, times) go into the JSON.

    python experiments/peptide/pipeline/reweight.py --pool runs/rugged2/c0/pool.pt --targets runs/rugged2/c0/targets.pt \
        --out runs/rugged2/c0/landscape.json [--confirm --n-steps 4000000 ...]
    python experiments/peptide/pipeline/reweight.py --analyze runs/rugged2/c0/landscape.json   (table + plots)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common   # noqa: E402
from simulate import Design, TEMP, simulate   # noqa: E402


def positions(Y, n):
    """raw internal coordinates [m, 2N - 3] -> bead positions [m, N, 2] (bead 0 at the origin, bond 1 along x)"""
    r, th = Y[:, :n - 1], Y[:, n - 1:]
    x = torch.zeros(Y.shape[0], n, 2, dtype=Y.dtype, device=Y.device)
    x[:, 1, 0] = r[:, 0]
    for k in range(1, n - 1):
        back = x[:, k - 1] - x[:, k]
        ang = torch.atan2(back[:, 1], back[:, 0]) + th[:, k - 1]
        x[:, k + 1] = x[:, k] + r[:, k:k + 1] * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    return x


def pair_kernel(Y, design):
    """[m, P] phi(r_ij) of every pair (design.pairs() order): the electrostatic energy per unit A q_i q_j"""
    x = positions(Y.double(), design.n_beads)
    P = design.pairs()
    r = (x[:, [j for _, j, _ in P]] - x[:, [i for i, _, _ in P]]).norm(dim=-1)
    el = design.elec
    return (torch.exp(-r / el["debye"]) / r * (r < el["rcut"])).float()


class Reweighter:
    """MBAR over a pool of ensembles of a charge design; weights and losses of any design q."""

    def __init__(self, pool, design, device, max_frames=None, tol=1e-7, max_iter=5000, seed=0):
        self.design, self.dev = design, device
        Y = pool["Y"]                                    # [K, M, dy]
        K, M, dy = Y.shape
        g = torch.Generator().manual_seed(seed)
        if max_frames and M > max_frames:                # the same subset size per design keeps N_k equal
            idx = torch.stack([torch.randperm(M, generator=g)[:max_frames] for _ in range(K)])
            Y = torch.gather(Y, 1, idx[..., None].expand(-1, -1, dy))
            M = max_frames
        self.K, self.M = K, M
        self.Y = Y.reshape(K * M, dy)
        self.Xk = pool["eps"].float()
        self.beta = 1.0 / (design.temp if design.temp is not None else TEMP)
        self.Phi = pair_kernel(self.Y, design).to(device)                   # [n, P]
        self.coef = self.beta * design.elec["A"]
        t0 = time.time()
        uk = self._u(self.Xk.to(device))                                    # [K, n]
        logN = math.log(M)
        f = torch.zeros(K, dtype=torch.float64, device=device)
        hist = []
        for it in range(max_iter):
            logden = torch.logsumexp((logN + f)[:, None] - uk.double(), 0)   # [n]
            fn = -torch.logsumexp(-uk.double() - logden[None], 1)
            fn = fn - fn[0]
            d = float((fn - f).abs().max())
            f = fn
            if it % 50 == 0 or d < tol:
                hist.append([it, d])
            if d < tol:
                break
        self.f, self.logden = f, torch.logsumexp((logN + f)[:, None] - uk.double(), 0).float()
        self.mbar = {"iterations": it + 1, "last_change": d, "history": hist, "time_s": time.time() - t0,
                     "K": K, "frames_per_design": M}
        print(f"[reweight] MBAR: K={K}, {K * M} frames, {it + 1} iterations, last change {d:.2e}, {time.time() - t0:.0f}s",
              flush=True)

    def _u(self, X):
        """reduced electrostatic energies [B, n] of all pool frames under designs X [B, p]"""
        qq = self.design.pair_charge(X)                                    # [B, P]
        return self.coef * qq @ self.Phi.T

    def logw(self, X):
        lw = -self._u(X) - self.logden[None]
        return lw - torch.logsumexp(lw, 1, keepdim=True)

    def ess(self, X, chunk=64):
        out = []
        for c in range(0, X.shape[0], chunk):
            lw = self.logw(X[c:c + chunk].to(self.dev))
            out.append(torch.exp(-torch.logsumexp(2 * lw, 1)))
        return torch.cat(out).cpu()


class RFF:
    """random Fourier features of the kernel sum_l exp(-|u|^2 / (2 h_l^2)) on standardized features"""

    def __init__(self, mean, std, hs, d_per=256, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.mean, self.std = mean.to(device), std.to(device)
        dz = mean.shape[0]
        self.W = torch.cat([torch.randn(dz, d_per, generator=g) / h for h in hs], 1).to(device)
        self.b = (2 * math.pi * torch.rand(self.W.shape[1], generator=g)).to(device)
        self.scale = math.sqrt(2.0 / d_per)            # each block approximates one Gaussian kernel (k(0) = 1)

    def __call__(self, Yraw):
        z = (common.raw_to_feat(Yraw.to(self.mean.device)) - self.mean) / self.std
        return self.scale * torch.cos(z @ self.W + self.b)


def mmd_u_target(psi):
    """mean embedding and the U-statistic ||mu||^2 of an unweighted sample's features"""
    n = psi.shape[0]
    mu = psi.mean(0)
    sq = (n * n * (mu @ mu) - (psi * psi).sum()) / (n * (n - 1))
    return mu, sq


def weighted_loss(lw, Psi, sqnorm, muT, sqT):
    """MMD^2 between the reweighted pool (log weights lw [B, n], normalized) and the target (muT, ||muT||^2_U), with the
    self terms of the weighted sample removed: (||sum w psi||^2 - sum w^2 |psi|^2) / (1 - sum w^2)."""
    w = torch.exp(lw)
    mu = w @ Psi                                                           # [B, D]
    w2 = (w * w).sum(1)
    self_t = (w * w) @ sqnorm
    xx = ((mu * mu).sum(1) - self_t) / (1 - w2).clamp_min(1e-6)
    return xx - 2 * mu @ muT + sqT


def basin_probs_weighted(w, codes, n_classes):
    """basin populations of the reweighted pool: w [n], codes [n] class ids"""
    return torch.zeros(n_classes, dtype=w.dtype, device=w.device).index_add_(0, codes, w)


def sign_dist(a, b):
    """distance between designs up to the global charge flip (q and -q have the same energy)"""
    return torch.minimum((a - b).norm(dim=-1), (a + b).norm(dim=-1))


def cluster(ends, losses, radius):
    """greedy clustering of endpoints in order of loss; returns [(center index, member indices)]"""
    order = torch.argsort(losses)
    free = torch.ones(len(ends), dtype=torch.bool)
    out = []
    for i in order.tolist():
        if not free[i]:
            continue
        m = (sign_dist(ends, ends[i][None]) < radius) & free
        free &= ~m
        out.append((i, m.nonzero().flatten().tolist()))
    return out


def run(a):
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pool = torch.load(a.pool, weights_only=False)
    tens = torch.load(a.targets, weights_only=False)
    design = Design.from_dict(pool["design"])
    assert Design.from_dict(tens["design"]).to_dict() == design.to_dict(), "pool and targets designs differ"
    assert all(p.startswith("charge") for p in design.free), "reweighting needs a design whose free parameters are charges"
    common.set_design(design)
    t0 = time.time()
    rw = Reweighter(pool, design, dev, max_frames=a.max_frames, seed=a.seed)
    # features: standardized by the pool; bandwidths from the median distance of 2,000 pool frames
    F = common.raw_to_feat(rw.Y)
    mean, std = F.mean(0), F.std(0).clamp_min(1e-6)
    g = torch.Generator().manual_seed(a.seed)
    sub = ((F[torch.randperm(F.shape[0], generator=g)[:2000]] - mean) / std)
    med = float(torch.cdist(sub, sub).median())
    hs = [m * med for m in a.bw_mult]
    rff = RFF(mean, std, hs, a.rff_dim, seed=a.seed, device=dev)
    Psi = torch.cat([rff(rw.Y[c:c + 200000]) for c in range(0, rw.Y.shape[0], 200000)])     # [n, D]
    sqnorm = (Psi * Psi).sum(1)
    codes = None
    names = common.basin_classes(design.n_beads)[1]
    st = common.angle_states(rw.Y)
    na = st.shape[1]
    class_of = common.basin_classes(design.n_beads)[0]
    codes = class_of[(st * (4 ** torch.arange(na - 1, -1, -1))).sum(1)].to(dev)
    lo, hi = design.bounds()
    lo_t, hi_t = torch.tensor(lo, device=dev), torch.tensor(hi, device=dev)

    def loss_of(X, muT, sqT, chunk=a.chunk):
        out = []
        for c in range(0, X.shape[0], chunk):
            out.append(weighted_loss(rw.logw(X[c:c + chunk].to(dev)), Psi, sqnorm, muT, sqT))
        return torch.cat(out)

    ids = list(range(tens["eps"].shape[0])) if a.target_ids is None else \
        [int(t) for t in a.target_ids.split(",")] if "," in a.target_ids or "-" not in a.target_ids else \
        list(range(int(a.target_ids.split("-")[0]), int(a.target_ids.split("-")[1]) + 1))
    res = {"args": vars(a), "design": design.to_dict(), "mbar": rw.mbar, "bandwidths": hs, "median_dist": med,
           "basin_names": names, "targets": []}
    gt = torch.Generator().manual_seed(a.seed + 1)
    for tid in ids:
        tt = time.time()
        T = tens["Y"][tid]
        qs = tens["eps"][tid].float()
        psiT = rff(T)
        muT, sqT = mmd_u_target(psiT)
        h = T.shape[0] // 2
        perm = torch.randperm(T.shape[0], generator=gt)
        mu1, sq1 = mmd_u_target(rff(T[perm[:h]]))
        mu2, sq2 = mmd_u_target(rff(T[perm[h:]]))
        floor = float(sq1 + sq2 - 2 * mu1 @ mu2)
        pT = common.basin_probs(T).to(dev)
        # MBAR at q* (and -q*: the same energy)
        Xs = torch.stack([qs, -qs]).to(dev)
        L_star = loss_of(Xs, muT, sqT).tolist()
        w_star = torch.exp(rw.logw(qs[None].to(dev)))[0]
        tv_star = 0.5 * float((basin_probs_weighted(w_star, codes, len(names)) - pT).abs().sum())
        ess_star = float(rw.ess(qs[None])[0])
        # random designs
        R = design.sample(a.n_random, generator=gt).to(dev)
        LR = loss_of(R, muT, sqT)
        # multi-start projected Adam on the reweighted loss
        X0 = design.sample(a.n_starts, generator=gt).to(dev)
        X = X0.clone().requires_grad_(True)
        opt = torch.optim.Adam([X], lr=a.lr)
        traj = []
        for it in range(a.steps):
            opt.zero_grad()
            L = loss_of(X, muT, sqT)
            L.sum().backward()
            opt.step()
            with torch.no_grad():
                X.copy_(torch.max(torch.min(X, hi_t), lo_t))
            if it % 10 == 0 or it == a.steps - 1:
                traj.append(L.detach().cpu().tolist())
        with torch.no_grad():
            ends = X.detach()
            Lend = loss_of(ends, muT, sqT)
            ess_end = rw.ess(ends)
        ends_c, Lend_c = ends.cpu(), Lend.cpu()
        dstar = sign_dist(ends_c, qs[None])
        medR = float(LR.median())
        thr = min(L_star) + a.solve_tol * medR
        solved = (Lend_c <= thr)
        cl = cluster(ends_c, Lend_c, a.radius)
        clusters = []
        for ci, (c, mem) in enumerate(cl):
            q0 = ends_c[c]
            # straight line from the cluster centre to the nearer of +-q*, MBAR loss
            tgt = qs if (q0 - qs).norm() <= (q0 + qs).norm() else -qs
            lam = torch.linspace(0, 1, a.line_points)
            line = (1 - lam[:, None]) * q0[None] + lam[:, None] * tgt[None]
            with torch.no_grad():
                Ll = loss_of(line.to(dev), muT, sqT).cpu()
                el = rw.ess(line)
            # local-minimum test: random perturbations of the centre
            with torch.no_grad():
                pert = q0[None] + a.pert_radius * torch.randn(64, design.dim_x, generator=gt)
                pert = torch.max(torch.min(pert, torch.tensor(hi)), torch.tensor(lo))
                Lp = loss_of(pert.to(dev), muT, sqT).cpu()
            clusters.append({"center": q0.tolist(), "loss": float(Lend_c[c]), "count": len(mem),
                             "members_solved": int(solved[mem].sum()), "dist_to_xstar": float(dstar[c]),
                             "ess": float(ess_end[c]), "line_lam": lam.tolist(), "line_loss": Ll.tolist(),
                             "line_ess": el.tolist(), "barrier": float(Ll.max() - Ll[0]),
                             "pert_frac_higher": float((Lp > Lend_c[c]).float().mean()),
                             "global": bool(Lend_c[c] <= thr)})
        rec = {"target_id": tid, "x_star": qs.tolist(), "floor": floor, "L_star": L_star, "tv_star_mbar": tv_star,
               "ess_star": ess_star, "target_basins": pT.cpu().tolist(),
               "random": {"x": R.cpu().tolist(), "loss": LR.cpu().tolist(), "ess": rw.ess(R).tolist()},
               "descent": {"x0": X0.cpu().tolist(), "x_end": ends_c.tolist(), "loss_end": Lend_c.tolist(),
                           "ess_end": ess_end.tolist(), "dist_end": dstar.tolist(), "loss_traj_every10": traj,
                           "steps": a.steps, "lr": a.lr},
               "solve_threshold": thr, "frac_solved": float(solved.float().mean()), "clusters": clusters,
               "n_minima": len([c for c in clusters if c["count"] >= a.min_count]),
               "n_spurious": len([c for c in clusters if c["count"] >= a.min_count and not c["global"]]),
               "time_s": time.time() - tt}
        print(f"[reweight] t{tid}: floor {floor:.2e}, L(q*) {L_star[0]:.2e} (ESS {ess_star:.0f}, basin TV {tv_star:.3f}), "
              f"median random {medR:.2e}, solved {rec['frac_solved']:.2f}, minima {rec['n_minima']} "
              f"(spurious {rec['n_spurious']}), {time.time() - tt:.0f}s", flush=True)
        for c in clusters[:6]:
            print(f"    cluster loss {c['loss']:.2e} n={c['count']} d*={c['dist_to_xstar']:.2f} ESS={c['ess']:.0f} "
                  f"barrier {c['barrier']:.2e} pert_higher {c['pert_frac_higher']:.2f} global={c['global']} q={np.round(c['center'], 2).tolist()}")
        res["targets"].append(rec)
    if a.confirm:
        confirm(a, res, design, tens, rff, dev)
    res["time_s"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(f"[reweight] wrote {a.out} ({time.time() - t0:.0f}s)")


def confirm(a, res, design, tens, rff, dev):
    """direct simulation of the straight line from the best spurious minima to q*: no reweighting"""
    from landscape import energy_distance
    X, meta = [], []
    for rec in res["targets"]:
        sp = [c for c in rec["clusters"] if not c["global"] and c["count"] >= a.min_count][:a.confirm_top]
        qs = torch.tensor(rec["x_star"])
        for k, c in enumerate(sp):
            q0 = torch.tensor(c["center"])
            tgt = qs if (q0 - qs).norm() <= (q0 + qs).norm() else -qs
            lam = torch.linspace(0, 1, a.confirm_points)
            X.append((1 - lam[:, None]) * q0[None] + lam[:, None] * tgt[None])
            meta.append((rec["target_id"], k, lam.tolist()))
    if not X:
        res["confirm"] = []
        return
    X = torch.cat(X)
    t0 = time.time()
    Ys = []
    for c in range(0, X.shape[0], a.sim_chunk):
        Ys += list(simulate(X[c:c + a.sim_chunk], n_rep=a.confirm_rep, n_steps=a.n_steps, stride=a.stride, burn=a.burn,
                            design=design, seed=a.seed + 99 + c, verbose=False))
    gs = torch.Generator().manual_seed(0)
    out, i = [], 0
    for tid, k, lam in meta:
        T = tens["Y"][tid]
        muT, sqT = mmd_u_target(rff(T))
        fT = common.raw_to_feat(T)
        mu, sd = fT.mean(0), fT.std(0).clamp_min(1e-6)
        pT = common.basin_probs(T)
        rows = []
        for l in lam:
            Y = Ys[i]; i += 1
            mG, sG = mmd_u_target(rff(Y))
            rows.append({"lam": l, "mmd": float(sG + sqT - 2 * mG @ muT),
                         "ed": energy_distance((common.raw_to_feat(Y) - mu) / sd, (fT - mu) / sd, gen=gs),
                         "tv": 0.5 * float((common.basin_probs(Y) - pT).abs().sum())})
        out.append({"target_id": tid, "rank": k, "rows": rows})
        m = [r["mmd"] for r in rows]
        print(f"[confirm] t{tid} spurious #{k}: direct MMD along the line " + " ".join(f"{v:.2e}" for v in m), flush=True)
    res["confirm"] = out
    res["confirm_time_s"] = time.time() - t0


def analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(a.analyze))
    base = os.path.splitext(a.analyze)[0]
    L = [f"Reweighted (MBAR) landscape {a.analyze}: design {d['design']}", "",
         f"MBAR: K = {d['mbar']['K']} pool designs x {d['mbar']['frames_per_design']} frames, {d['mbar']['iterations']} "
         f"iterations (last change {d['mbar']['last_change']:.1e}). Loss = MMD^2 (U-statistic), bandwidths "
         f"{', '.join(f'{h:.2f}' for h in d['bandwidths'])}.", "",
         "| target | floor | L(q*) MBAR | ESS(q*) | basin TV(q*) MBAR | median L random | best random / L(q*) | starts solved | local minima (>= %d starts) | spurious | best spurious L / L(q*) | its barrier (MBAR line) | perturbations higher |" % d["args"]["min_count"],
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d["targets"]:
        ls = r["L_star"][0]
        lr = np.array(r["random"]["loss"])
        sp = [c for c in r["clusters"] if not c["global"] and c["count"] >= d["args"]["min_count"]]
        b = sp[0] if sp else None
        L.append(f"| {r['target_id']} | {r['floor']:.1e} | {ls:.1e} | {r['ess_star']:.0f} | {r['tv_star_mbar']:.3f} | {np.median(lr):.1e} | "
                 f"{lr.min() / max(ls, 1e-9):.1f} | {r['frac_solved']:.2f} | {r['n_minima']} | {r['n_spurious']} | "
                 + (f"{b['loss'] / max(ls, 1e-9):.1f} (n={b['count']}) | {b['barrier']:.1e} | {b['pert_frac_higher']:.2f} |" if b else "- | - | - |"))
    fs = [r["frac_solved"] for r in d["targets"]]
    L += ["", f"Mean fraction of descent starts that reach the global basin: {np.mean(fs):.2f} "
          f"(targets with >= 1 spurious minimum: {sum(r['n_spurious'] > 0 for r in d['targets'])}/{len(d['targets'])})."]
    if d.get("confirm"):
        L += ["", "Direct simulation (no reweighting) along the line spurious minimum (lam 0) -> q* (lam 1): MMD^2 / energy distance / basin TV", "",
              "| target | spurious # | lam: MMD^2 | max interior - endpoint (MMD^2) | TV at lam 0 | TV at lam 1 |", "|---|---|---|---|---|---|"]
        for c in d["confirm"]:
            m = np.array([x["mmd"] for x in c["rows"]])
            L.append(f"| {c['target_id']} | {c['rank']} | " + " ".join(f"{v:.1e}" for v in m) +
                     f" | {m[1:-1].max() - m[0]:.1e} | {c['rows'][0]['tv']:.3f} | {c['rows'][-1]['tv']:.3f} |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    # plots: MBAR line scans of target 0..3 and the direct confirmations
    nt = min(4, len(d["targets"]))
    fig, ax = plt.subplots(1, nt + (1 if d.get("confirm") else 0), figsize=(3.3 * (nt + 1), 3), squeeze=False)
    for k in range(nt):
        r = d["targets"][k]
        for c in r["clusters"]:
            if c["count"] >= d["args"]["min_count"] and not c["global"]:
                ax[0, k].plot(c["line_lam"], c["line_loss"], lw=1, label=f"n={c['count']}")
        ax[0, k].axhline(r["L_star"][0], color="k", ls=":", lw=0.8)
        ax[0, k].set_yscale("log"); ax[0, k].set_title(f"target {r['target_id']}: MBAR lines", fontsize=9)
        ax[0, k].set_xlabel("spurious min -> q*")
    if d.get("confirm"):
        for c in d["confirm"]:
            ax[0, -1].plot([x["lam"] for x in c["rows"]], [x["mmd"] for x in c["rows"]], marker="o", ms=3, lw=1,
                           label=f"t{c['target_id']}#{c['rank']}")
        ax[0, -1].set_yscale("symlog", linthresh=1e-4); ax[0, -1].set_title("direct simulation", fontsize=9)
        ax[0, -1].legend(fontsize=6); ax[0, -1].set_xlabel("spurious min -> q*")
    fig.tight_layout()
    fig.savefig(base + "_lines.png", dpi=140)
    print("wrote", base + ".md", base + "_lines.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analyze", default=None, help="a landscape JSON to summarize (table + plots)")
    p.add_argument("--pool", default=None)
    p.add_argument("--targets", default=None)
    p.add_argument("--target-ids", default=None, help="e.g. 0-7 or 0,3 (default all)")
    p.add_argument("--max-frames", type=int, default=2000, help="pool frames per design used")
    p.add_argument("--bw-mult", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--rff-dim", type=int, default=256, help="random features per bandwidth")
    p.add_argument("--n-random", type=int, default=1000)
    p.add_argument("--n-starts", type=int, default=128)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--radius", type=float, default=0.2, help="clustering radius of endpoints (raw design units)")
    p.add_argument("--min-count", type=int, default=2, help="a local minimum must collect this many starts")
    p.add_argument("--solve-tol", type=float, default=0.05, help="solved: L <= L(q*) + tol x median random L")
    p.add_argument("--line-points", type=int, default=41)
    p.add_argument("--pert-radius", type=float, default=0.05)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--confirm", action="store_true", help="re-simulate the line scans of the best spurious minima")
    p.add_argument("--confirm-top", type=int, default=2)
    p.add_argument("--confirm-points", type=int, default=11)
    p.add_argument("--confirm-rep", type=int, default=32)
    p.add_argument("--sim-chunk", type=int, default=200, help="designs per simulation batch (--confirm)")
    p.add_argument("--n-steps", type=int, default=4_000_000)
    p.add_argument("--burn", type=int, default=50_000)
    p.add_argument("--stride", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    analyze(a) if a.analyze else run(a)


if __name__ == "__main__":
    main()
