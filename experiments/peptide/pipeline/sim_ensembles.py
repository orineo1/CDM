"""
Simulate equilibrium ensembles Y | eps with the batched Langevin simulator (simulate.py) and save them.

    # training set: 300 random eps in the box
    python experiments/peptide/pipeline/sim_ensembles.py --n-eps 300 --seed 1 --keep 8000 --out runs/peptide/train_ens.pt
    # held-out targets
    python experiments/peptide/pipeline/sim_ensembles.py --n-eps 40 --seed 2 --keep 20000 --out runs/peptide/targets_ens.pt
    # ensembles at given eps (a .npy [n, K] or a .json list of K-vectors)
    python experiments/peptide/pipeline/sim_ensembles.py --eps-file eps_hat.npy --keep 20000 --out runs/peptide/eval_ens.pt
    # N beads: --n-beads 6 (all 5 eps_d free by default); --free-eps 1,2 --fixed-eps 3=0.5,4=0.5,5=0.5 for others
    # (--design-from <ens.pt> copies the design of an existing ensemble file)

Output .pt: {"eps": [E, K] free eps, "Y": [E, keep, 2N - 3] raw (N - 1 bond lengths, N - 2 angles; N = 4: r12, r23,
             r34, theta123, theta234), "design": Design.to_dict(), "protocol": {...}, "temperature": kinetic T,
             "time_s": wall time}. With --no-subsample, Y holds every frame in order: replica r is the block
             Y[:, r * F:(r + 1) * F] of F = (n_steps - burn) / stride frames (the equilibration checks use this).
Every ensemble uses the same protocol (n_rep replicas from the straight chain, n_steps steps each, frames
every `stride` steps after `burn`), so training data, targets and evaluation share one definition of P(Y | eps).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import simulate, EPS_BOX, Design   # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-eps", type=int, default=None, help="draw this many eps vectors uniformly in the box")
    p.add_argument("--eps-file", default=None, help=".npy [n, K] or .json list of K-vectors (K free eps)")
    p.add_argument("--n-beads", type=int, default=4)
    p.add_argument("--free-eps", "--free", dest="free_eps", default=None,
                   help="free parameters: sequence distances 1,2 or names/groups, e.g. sigma,eta or eps1,sigma3 "
                        "(default: N = 4 -> 1,2; else all eps_d)")
    p.add_argument("--fixed-eps", "--fixed", dest="fixed_eps", default=None,
                   help="values of the others, e.g. 3=0.5 or eps=0.5,eta4=2 (default: N = 4 -> 3=0.5)")
    p.add_argument("--box", default=None, help="ranges, e.g. sigma=0.8:1.3,eta=0.5:1.5 (default simulate.BOXES)")
    p.add_argument("--prior", default="uniform", help="design prior for --n-eps draws: uniform | band:<w>")
    p.add_argument("--temp", type=float, default=None, help="Langevin temperature (default simulate.TEMP = 0.09); "
                   "a non-legacy design keeps it, so later scripts (--design-from) simulate at the same temperature")
    p.add_argument("--elec", default=None, help="screened Coulomb of charged designs, e.g. A=1,debye=2,rcut=6 (simulate.ELEC)")
    p.add_argument("--design-from", default=None, help="take the design from this ensemble .pt (overrides the three above)")
    p.add_argument("--no-subsample", action="store_true",
                   help="keep all frames in replica-major time order (for the equilibration checks)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-rep", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=4_000_000)
    p.add_argument("--stride", type=int, default=200)
    p.add_argument("--burn", type=int, default=50_000)
    p.add_argument("--keep", type=int, default=8000, help="frames kept per eps (random subset of all replicas' frames)")
    p.add_argument("--chunk", type=int, default=400, help="eps per simulation batch")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.design_from:
        design = Design.from_dict(torch.load(a.design_from, weights_only=False).get("design"))
    else:
        legacy = Design.from_args(a.n_beads, a.free_eps, a.fixed_eps, a.box, a.prior).is_legacy
        # the temperature joins the design except for the legacy (eps-only) designs, whose files stay as before
        design = Design.from_args(a.n_beads, a.free_eps, a.fixed_eps, a.box, a.prior, None if legacy else a.temp, a.elec)
    K, dy = design.dim_x, design.dim_y

    if a.eps_file:
        eps = np.load(a.eps_file) if a.eps_file.endswith(".npy") else np.array(json.load(open(a.eps_file)))
        eps = torch.as_tensor(eps, dtype=torch.float32).view(-1, K)
    else:
        g = torch.Generator().manual_seed(a.seed)
        eps = design.sample(a.n_eps, generator=g)       # legacy designs: the same draws as before
    print(f"[sim_ensembles] {design}: {eps.shape[0]} eps, {a.n_rep} replicas x {a.n_steps:,} steps, "
          f"keep {'all' if a.no_subsample else a.keep}/eps", flush=True)
    t0 = time.time()
    Ys, Ts = [], []
    g = torch.Generator().manual_seed(a.seed + 12345)
    for c in range(0, eps.shape[0], a.chunk):
        Y, T = simulate(eps[c:c + a.chunk], n_rep=a.n_rep, n_steps=a.n_steps, stride=a.stride, burn=a.burn,
                        seed=a.seed * 1000 + c, return_temperature=True, design=design, temp=a.temp)
        if a.no_subsample:
            Ys.append(Y)
        else:
            idx = torch.stack([torch.randperm(Y.shape[1], generator=g)[:a.keep] for _ in range(Y.shape[0])])
            Ys.append(torch.gather(Y, 1, idx[..., None].expand(-1, -1, dy)))
        Ts.append(T)
        print(f"[sim_ensembles] chunk {c // a.chunk}: T_kin = {T:.4f} (target 0.09), {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"eps": eps, "Y": torch.cat(Ys), "temperature": float(np.mean(Ts)), "time_s": time.time() - t0,
                "design": design.to_dict(),
                "protocol": {k: getattr(a, k) for k in ("n_rep", "n_steps", "stride", "burn", "keep", "seed", "no_subsample", "temp")}},
               a.out)
    print(f"[sim_ensembles] wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
