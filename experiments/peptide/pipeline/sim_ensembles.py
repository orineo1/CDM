"""
Simulate equilibrium ensembles Y | eps with the batched Langevin simulator (simulate.py) and save them.

    # training set: 300 random eps in the box
    python experiments/peptide/pipeline/sim_ensembles.py --n-eps 300 --seed 1 --keep 8000 --out runs/peptide/train_ens.pt
    # held-out targets
    python experiments/peptide/pipeline/sim_ensembles.py --n-eps 40 --seed 2 --keep 20000 --out runs/peptide/targets_ens.pt
    # ensembles at given eps (a .npy [n, 2] or a .json list of pairs)
    python experiments/peptide/pipeline/sim_ensembles.py --eps-file eps_hat.npy --keep 20000 --out runs/peptide/eval_ens.pt

Output .pt: {"eps": [E, 2], "Y": [E, keep, 5] raw (r12, r23, r34, theta123, theta234), "protocol": {...},
             "temperature": kinetic T, "time_s": wall time}
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
from simulate import simulate, EPS_BOX   # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-eps", type=int, default=None, help="draw this many eps pairs uniformly in the box")
    p.add_argument("--eps-file", default=None, help=".npy [n, 2] or .json list of pairs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-rep", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=4_000_000)
    p.add_argument("--stride", type=int, default=200)
    p.add_argument("--burn", type=int, default=50_000)
    p.add_argument("--keep", type=int, default=8000, help="frames kept per eps (random subset of all replicas' frames)")
    p.add_argument("--chunk", type=int, default=400, help="eps per simulation batch")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if a.eps_file:
        eps = np.load(a.eps_file) if a.eps_file.endswith(".npy") else np.array(json.load(open(a.eps_file)))
        eps = torch.as_tensor(eps, dtype=torch.float32).view(-1, 2)
    else:
        g = torch.Generator().manual_seed(a.seed)
        eps = EPS_BOX[0] + (EPS_BOX[1] - EPS_BOX[0]) * torch.rand(a.n_eps, 2, generator=g)
    print(f"[sim_ensembles] {eps.shape[0]} eps, {a.n_rep} replicas x {a.n_steps:,} steps, keep {a.keep}/eps", flush=True)
    t0 = time.time()
    Ys, Ts = [], []
    g = torch.Generator().manual_seed(a.seed + 12345)
    for c in range(0, eps.shape[0], a.chunk):
        Y, T = simulate(eps[c:c + a.chunk], n_rep=a.n_rep, n_steps=a.n_steps, stride=a.stride, burn=a.burn,
                        seed=a.seed * 1000 + c, return_temperature=True)
        idx = torch.stack([torch.randperm(Y.shape[1], generator=g)[:a.keep] for _ in range(Y.shape[0])])
        Ys.append(torch.gather(Y, 1, idx[..., None].expand(-1, -1, 5)))
        Ts.append(T)
        print(f"[sim_ensembles] chunk {c // a.chunk}: T_kin = {T:.4f} (target 0.09), {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"eps": eps, "Y": torch.cat(Ys), "temperature": float(np.mean(Ts)), "time_s": time.time() - t0,
                "protocol": {k: getattr(a, k) for k in ("n_rep", "n_steps", "stride", "burn", "keep", "seed")}}, a.out)
    print(f"[sim_ensembles] wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
