"""Exploration: summaries that break the residue symmetry of the count target weakly, so the residue modes have
unequal depth (deceptive local minima). Exact reweighted loss, GB-FESO-style Adam from x0 and random starts;
reports the distribution of end losses and which residue mode each start lands in."""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rw import Frames
import system as S

KARPLUS = (6.51, -1.76, 1.60)   # 3J(HN,HA) = A cos^2(phi - 60) + B cos(phi - 60) + C  (Vuister & Bax 1993), Hz


def jcoup(phi):
    t = phi - np.deg2rad(60.0)
    return KARPLUS[0] * np.cos(t) ** 2 + KARPLUS[1] * np.cos(t) + KARPLUS[2]


class Exact:
    def __init__(self, frames, thin, scale, dev="cpu"):
        F = Frames(glob.glob(frames), thin=thin)
        d = torch.float64
        self.B = torch.tensor(F.B, dtype=d); self.base = torch.tensor(F.base, dtype=d)
        self.cnt = torch.tensor(S.count_plus(F.phi)); self.kT, self.scale = F.kT, scale
        plus = (np.degrees(F.phi) > 0) & (np.degrees(F.phi) < 120)
        self.sign = torch.tensor(plus.astype(int))                       # (N, 3)
        self.J = torch.tensor(jcoup(F.phi), dtype=d)                      # (N, 3)
        self.N = F.N

    def w(self, xs):
        return torch.softmax(self.base - self.B @ (xs.double() * self.scale) / self.kT, 0)

    def summ(self, xs, k):
        w = self.w(xs)
        pc = torch.zeros(4, dtype=w.dtype).index_add(0, self.cnt, w)
        pj = torch.zeros(8, dtype=w.dtype).index_add(0, self.cnt * 2 + self.sign[:, k], w)
        J = w @ self.J
        plus = w @ self.sign.double()
        return pc, pj, J, plus, w


def kl(G, P):
    m = G > 1e-12
    return (G[m] * (torch.log(G[m]) - torch.log(P[m].clamp_min(1e-300)))).sum()


def make_loss(E, kind, tgt, k, wJ, sJ, wP=0.1):
    pc_t, pj_t, J_t, p_t = tgt

    def L(xs):
        pc, pj, J, plus, _ = E.summ(xs, k)
        if kind == "joint":
            return kl(pj_t, pj)
        if kind == "countP":
            return kl(pc_t, pc) + wP * (plus[k] - p_t[k]) ** 2
        return kl(pc_t, pc) + wJ * ((J[k] - J_t[k]) / sJ) ** 2
    return L


def adam(L, x0, box=2.0, steps=201, lr=0.05, noise=0.05):
    x = x0.clone().clamp(-box, box).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr); best, bx = 1e9, x.detach().clone()
    for s in range(steps + 1):
        opt.zero_grad(); l = L(x); v = float(l)
        if v < best:
            best, bx = v, x.detach().clone()
        if s == steps:
            break
        l.backward(); opt.step()
        with torch.no_grad():
            x.add_(noise * (1 - s / steps) * torch.randn_like(x)); x.clamp_(-box, box)
    return bx, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default="runs/trialanine/cdm/meta.json")
    p.add_argument("--frames", default="runs/trialanine/prod_x0_w*.npz")
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--kind", choices=["joint", "countJ", "countP"], default="countJ")
    p.add_argument("--wP", type=float, default=0.1)
    p.add_argument("--ids", default=None, help="comma list of target ids (overrides --targets)")
    p.add_argument("--wJ", type=float, default=0.3)
    p.add_argument("--sJ", type=float, default=0.5)
    p.add_argument("--starts", type=int, default=16)
    p.add_argument("--targets", default="0-29")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    meta = json.load(open(a.meta)); scale = meta["x_scale_kJ"]
    E = Exact(a.frames, a.thin, scale)
    lo, hi = map(int, a.targets.split("-"))
    res = []
    for t in meta["targets"]:
        if t["id"] is None or not (lo <= t["id"] <= hi) or (a.ids and t["id"] not in [int(v) for v in a.ids.split(",")]):
            continue
        xs_star = torch.tensor(t["x_star_std"])
        k = t["mode_star"]
        with torch.no_grad():
            pc_t, pj_t, J_t, plus_t, _ = E.summ(xs_star, k)
        L = make_loss(E, a.kind, (pc_t, pj_t, J_t, plus_t), k, a.wJ, a.sJ, a.wP)
        g = torch.Generator().manual_seed(t["id"])
        starts = [torch.zeros(12)] + [torch.randn(12, generator=g).clamp(-2, 2) for _ in range(a.starts)]
        ends = []
        for i, x0 in enumerate(starts):
            torch.manual_seed(1000 * t["id"] + i)
            xb, lb = adam(L, x0)
            with torch.no_grad():
                pc, pj, J, plus, w = E.summ(xb, k)
            ends.append(dict(L=lb, count_tv=float(0.5 * (pc - pc_t).abs().sum()), joint_tv=float(0.5 * (pj - pj_t).abs().sum()),
                             dJ=float(J[k] - J_t[k]), dP=float(plus[k] - plus_t[k]), mode=int(torch.argmax(plus)), ess=float(1 / (w ** 2).sum())))
        Ls = np.array([e["L"] for e in ends])
        rec = dict(id=t["id"], kind=t["kind"], k=k, G_count=t["G_count"], plus_star=t["plus_marginals"],
                   L_min=float(Ls.min()), x0=ends[0], frac_bad=float(np.mean(Ls > Ls.min() + 0.01)),
                   frac_bad_tv05=float(np.mean([e["joint_tv"] > 0.05 for e in ends[1:]])),
                   frac_bad_summary=float(np.mean([max(e["count_tv"], abs(e["dP"])) > 0.05 for e in ends[1:]])),
                   modes=[e["mode"] for e in ends], L=[round(v, 4) for v in Ls.tolist()], ends=ends)
        res.append(rec)
        print(f"t{t['id']:2d} {t['kind']:12s} k={k} Lmin={Ls.min():.4f} x0: L={ends[0]['L']:.3f} jointTV={ends[0]['joint_tv']:.3f} "
              f"dP={ends[0]['dP']:.3f} mode={ends[0]['mode']} | bad(L>min+0.01)={rec['frac_bad']:.2f} jointTV>0.05={rec['frac_bad_tv05']:.2f} summary>0.05={rec['frac_bad_summary']:.2f} "
              f"modes={np.bincount(rec['modes'], minlength=3).tolist()}", flush=True)
    json.dump(dict(args=vars(a), targets=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
