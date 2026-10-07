"""Probe of a design: how the equilibrium ensemble responds to one design parameter (stage 1 of the rugged model).

For each of --n-base base designs drawn from the design's prior, sweeps one parameter (e.g. sigma4 or eta1) over
--values with all others fixed, simulates every design, and reports per design the conformation (basin)
populations (`common.basin_probs`; chain reversal merged only for reversal-symmetric designs) and, for every
non-bonded pair, the contact fraction (distance < CONTACT * sigma_ij). A bump-shaped or switching response of
the populations to one parameter is what makes the loss over designs multimodal.

    python experiments/peptide/pipeline/probe_residue.py --n-beads 8 --free sigma,eta --fixed eps=0.5 \
        --param sigma4 --values 0.7:1.4:15 --n-base 3 --n-rep 64 --n-steps 4000000 --out probe_sigma4.json
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common   # noqa: E402
from simulate import simulate, Design   # noqa: E402

CONTACT = 1.3          # contact: r_ij < 1.3 sigma_ij (the LJ minimum is at 2^(1/6) sigma ~ 1.12 sigma)


def positions(Y, n):
    """raw internal coordinates [m, 2N - 3] -> bead positions [m, N, 2] (bead 0 at the origin, bond 1 along x)."""
    r, th = Y[:, :n - 1], Y[:, n - 1:]
    x = torch.zeros(Y.shape[0], n, 2)
    x[:, 1, 0] = r[:, 0]
    for k in range(1, n - 1):
        back = x[:, k - 1] - x[:, k]                    # i -> i-1
        ang = torch.atan2(back[:, 1], back[:, 0]) + th[:, k - 1]
        x[:, k + 1] = x[:, k] + r[:, k:k + 1] * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    return x


def contacts(Y, sig_pair, design):
    """contact fraction of every non-bonded pair; sig_pair [n_pairs] in design.pairs() order"""
    n = design.n_beads
    x = positions(Y, n)
    out = {}
    for k, (i, j, d) in enumerate(design.pairs()):
        if d < 2:
            continue
        dist = (x[:, i] - x[:, j]).norm(dim=-1)
        out[f"{i + 1}-{j + 1}"] = float((dist < CONTACT * float(sig_pair[k])).float().mean())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-beads", type=int, default=8)
    p.add_argument("--free", default="sigma,eta")
    p.add_argument("--fixed", default="eps=0.5")
    p.add_argument("--box", default=None)
    p.add_argument("--prior", default="uniform")
    p.add_argument("--param", required=True, help="the swept parameter, e.g. sigma4 (must be free)")
    p.add_argument("--values", default="0.7:1.4:15", help="start:stop:count")
    p.add_argument("--n-base", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temp", type=float, default=None)
    p.add_argument("--elec", default=None, help="screened Coulomb of charged designs, e.g. A=1,debye=2,rcut=6")
    p.add_argument("--n-rep", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=4_000_000)
    p.add_argument("--burn", type=int, default=400_000)
    p.add_argument("--stride", type=int, default=400)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    design = Design.from_args(a.n_beads, a.free, a.fixed, a.box, a.prior, a.temp, a.elec)
    common.set_design(design)
    assert a.param in design.free, f"{a.param} is not free in {design}"
    k = design.free.index(a.param)
    lo, hi, cnt = a.values.split(":")
    values = np.linspace(float(lo), float(hi), int(cnt))
    base = design.sample(a.n_base, generator=torch.Generator().manual_seed(a.seed))
    X = []
    for b in range(a.n_base):
        for v in values:
            x = base[b].clone(); x[k] = float(v); X.append(x)
    X = torch.stack(X)
    t0 = time.time()
    Y = simulate(X, n_rep=a.n_rep, n_steps=a.n_steps, stride=a.stride, burn=a.burn, device=a.device, design=design,
                 temp=a.temp, verbose=True)
    sig = design.pair_sigma(X)
    names = common.basin_classes(design.n_beads)[1]
    rows = []
    for i in range(X.shape[0]):
        pr = common.basin_probs(Y[i]).tolist()
        rows.append({"base": i // len(values), "value": float(X[i, k]), "x": X[i].tolist(), "basins": pr,
                     "contacts": contacts(Y[i], sig[i] if sig is not None else torch.ones(len(design.pairs())), design)})
    json.dump({"args": vars(a), "design": design.to_dict(), "param": a.param, "basin_names": names, "rows": rows,
               "time_s": time.time() - t0}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
