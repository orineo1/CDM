"""Probe of the residue design (`simulate.ResidueDesign`): how the ensemble responds to one bead's size.

Sweeps sigma of one bead with the other beads fixed, simulates each design, and reports for every non-bonded
pair the contact fraction (distance < CONTACT * sigma_ij) and the populations of the conformations (per-angle
folded/open states, merged with their mirror images). A non-monotone (bump-shaped) response of the contacts
to sigma is what makes the loss over designs multimodal: a target contact level is then reached at two sizes.

    python experiments/peptide/pipeline/probe_residue.py --n-beads 4 --bead 3 --sigmas 0.6:1.8:13 \
        --base-sigma 1.0 --base-eps 0.8 --n-steps 1000000 --out probe_N4_bead3.json
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
from simulate import simulate, ResidueDesign   # noqa: E402

CONTACT = 1.3          # contact: r_ij < 1.3 sigma_ij (the LJ minimum is at 2^(1/6) sigma ~ 1.12 sigma)
FOLD_MAX = 1.55        # per-angle folded / open threshold of common.py


def positions(Y, n):
    """raw internal coordinates [m, 2N - 3] -> bead positions [m, N, 2] (bead 0 at the origin, bond 1 along x)."""
    r, th = Y[:, :n - 1], Y[:, n - 1:]
    m = Y.shape[0]
    x = torch.zeros(m, n, 2)
    x[:, 1, 0] = r[:, 0]
    for k in range(1, n - 1):
        back = x[:, k - 1] - x[:, k]                    # i -> i-1
        ang = torch.atan2(back[:, 1], back[:, 0]) + th[:, k - 1]
        x[:, k + 1] = x[:, k] + r[:, k:k + 1] * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    return x


def summarize(Y, sig, n):
    x = positions(Y, n)
    out = {"contacts": {}, "conformations": {}}
    for i, j in itertools.combinations(range(n), 2):
        if j - i < 2:
            continue
        s_ij = 0.5 * (sig[i] + sig[j])
        d = (x[:, i] - x[:, j]).norm(dim=-1)
        out["contacts"][f"{i + 1}-{j + 1}"] = float((d < CONTACT * s_ij).float().mean())
    th = Y[:, n - 1:]
    st = (th < 0).long() + 2 * (th.abs() >= FOLD_MAX).long()      # 0 +f, 1 -f, 2 +o, 3 -o
    sym, flip = "fFoO", {0: 1, 1: 0, 2: 3, 3: 2}
    counts = {}
    for row in st.tolist():
        t = tuple(row)
        key = min(t, tuple(flip[s] for s in t))                     # mirror image: same population
        counts[key] = counts.get(key, 0) + 1
    tot = sum(counts.values())
    for k, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        out["conformations"]["".join(sym[s] for s in k)] = c / tot
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-beads", type=int, default=4)
    p.add_argument("--bead", type=int, default=3, help="0-based index of the bead whose size is swept")
    p.add_argument("--sigmas", default="0.6:1.8:13", help="start:stop:count")
    p.add_argument("--base-sigma", type=float, default=1.0)
    p.add_argument("--base-eps", type=float, default=0.8)
    p.add_argument("--n-rep", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=1_000_000)
    p.add_argument("--burn", type=int, default=100_000)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lo, hi, k = a.sigmas.split(":")
    sigmas = np.linspace(float(lo), float(hi), int(k))
    n = a.n_beads
    design = ResidueDesign(n)
    X = []
    for s in sigmas:
        sig = [a.base_sigma] * n
        sig[a.bead] = float(s)
        X.append([v for i in range(n) for v in (sig[i], a.base_eps)])
    t0 = time.time()
    Y = simulate(X, n_rep=a.n_rep, n_steps=a.n_steps, burn=a.burn, device=a.device, design=design, verbose=True)
    rows = []
    for s, x, y in zip(sigmas, X, Y):
        r = summarize(y, x[0::2], n)
        r["sigma"] = float(s)
        rows.append(r)
    pairs = list(rows[0]["contacts"])
    print(f"\nN = {n}, bead {a.bead + 1} swept, other sigma = {a.base_sigma}, all eps = {a.base_eps}; "
          f"contact fraction (r < {CONTACT} sigma_ij) and the top conformations")
    print("sigma  " + "  ".join(f"{q:>6s}" for q in pairs) + "   conformations")
    for r in rows:
        top = ", ".join(f"{c} {v:.2f}" for c, v in list(r["conformations"].items())[:3])
        print(f"{r['sigma']:5.2f}  " + "  ".join(f"{r['contacts'][q]:6.3f}" for q in pairs) + f"   {top}")
    json.dump({"args": vars(a), "design": design.to_dict(), "rows": rows, "time_s": time.time() - t0},
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
