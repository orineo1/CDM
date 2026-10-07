"""Reweighting core: frames sampled at design x_s with frozen bias V -> any design x.
log w_n(x) = -(x - x_s).B_n / kT + V_n / kT   (energy linear in x, so this is exact importance weighting).
"""
import glob, json
import numpy as np
from scipy.special import logsumexp
import system as S


class Frames:
    def __init__(self, paths, thin=1):
        phi, psi, V, xs, walker = [], [], [], [], []
        for k, p in enumerate(sorted(paths)):
            d = np.load(p)
            phi.append(d["phi"][::thin]); psi.append(d["psi"][::thin]); V.append(d["V"][::thin])
            xs.append(np.repeat(d["x"][None], len(phi[-1]), 0)); walker.append(np.full(len(phi[-1]), k))
            self.T = float(d["T"])
        self.phi, self.psi, self.V = np.concatenate(phi), np.concatenate(psi), np.concatenate(V)
        self.xs, self.walker = np.concatenate(xs), np.concatenate(walker)
        self.kT = S.KB * self.T
        self.B = S.basis(self.phi)
        self.s = S.states(self.phi)
        self.N = len(self.phi)
        self.base = (self.V + (self.xs * self.B).sum(1)) / self.kT  # log w(x) = base - x.B/kT
        self.onehot = np.eye(8)[self.s]

    def logw(self, x, mask=None):
        lw = self.base - self.B @ np.asarray(x) / self.kT
        if mask is not None:
            lw = np.where(mask, lw, -np.inf)
        return lw - logsumexp(lw)

    def weights(self, x, mask=None):
        return np.exp(self.logw(x, mask))

    def pops(self, x, mask=None):
        return self.weights(x, mask) @ self.onehot

    def ess(self, x, mask=None):
        w = self.weights(x, mask)
        tot = 1.0 / (w ** 2).sum()
        per = []
        for s in range(8):
            ws = w[self.s == s]
            per.append(ws.sum() ** 2 / (ws ** 2).sum() if ws.sum() > 0 else 0.0)
        return tot, np.array(per)

    def pops_blocks(self, x, nblocks=None):
        """Populations per walker (block), for error bars from between-walker spread."""
        out = []
        for k in np.unique(self.walker):
            out.append(self.pops(x, self.walker == k))
        return np.array(out)
