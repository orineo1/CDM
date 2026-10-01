"""
Grid-search baseline over eps in the box, on the same surrogate and loss as our method: for every grid
point, the MMD between --n-cond conditional samples of our trained sampler and the target (search_mog's
own make_cfg / make_loss / _final_loss), then the --select-top best points are re-estimated with
--select-n samples x --select-repeats (the reselection search_mog uses) and the minimum is returned.

    python experiments/peptide/pipeline/grid_search.py --prior runs/peptide/models/prior/model_unconditional.pt \
        --sampler runs/peptide/models/diff/model_conditional.pt --targets-dir runs/peptide/targets \
        --target-ids 0-39 --grid 30 --out runs/peptide/results/grid_0.json
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "synthetic", "pipeline"))
from common import x_to_std, x_from_std   # noqa: E402
from simulate import EPS_BOX   # noqa: E402
import search_mog as fx   # noqa: E402
from cdm.guidance.core import _final_loss   # noqa: E402
from gbfeso import parse_ids   # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prior", required=True)
    p.add_argument("--sampler", required=True)
    p.add_argument("--targets-dir", required=True)
    p.add_argument("--target-ids", default="0-39")
    p.add_argument("--grid", type=int, default=30)
    p.add_argument("--n-cond", type=int, default=1000)
    p.add_argument("--select-top", type=int, default=5)
    p.add_argument("--select-n", type=int, default=2000)
    p.add_argument("--select-repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lin = torch.linspace(*EPS_BOX, a.grid)
    E = torch.cartesian_prod(lin, lin)
    X = x_to_std(E).to(dev)
    eps_star = json.load(open(os.path.join(a.targets_dir, "..", "meta.json")))["eps_star"]
    records = []
    for tid in parse_ids(a.target_ids, len(eps_star)):
        tgt = os.path.join(a.targets_dir, f"target_{tid:03d}.npy")
        sys.argv = ["search_mog", "--prior", a.prior, "--sampler", a.sampler, "--target", tgt, "--device", dev]
        args = fx.parse_args()
        if not records:
            prior, _ = fx.load_model(a.prior, dev)
            sampler, sampler_type = fx.load_model(a.sampler, dev)
            loss_fn = fx.make_loss(args)
        gen, _ = fx.load_target(tgt, dev)
        cfg = fx.make_cfg(args, prior, sampler_type, gen, a.seed)
        torch.manual_seed(a.seed + tid)
        torch.cuda.synchronize() if dev == "cuda" else None
        t0 = time.time()
        c = dataclasses.replace(cfg, final_n_cond=a.n_cond)
        L = torch.stack([_final_loss(sampler, loss_fn, c, X[i:i + 1]).reshape(()) for i in range(X.shape[0])]).float().cpu()
        top = torch.argsort(L)[:a.select_top].tolist()
        cs = dataclasses.replace(cfg, final_n_cond=a.select_n)
        sel = [float(sum(_final_loss(sampler, loss_fn, cs, X[i:i + 1]) for _ in range(a.select_repeats)) / a.select_repeats)
               for i in top]
        best = top[min(range(len(top)), key=lambda j: sel[j])]
        torch.cuda.synchronize() if dev == "cuda" else None
        dt = time.time() - t0
        rec = {"target_id": tid, "eps_star": eps_star[tid], "eps_hat": E[best].tolist(), "surrogate_loss": min(sel),
               "time_s": dt, "n_loss_evals": int(X.shape[0] + a.select_top * a.select_repeats)}
        records.append(rec)
        print(f"[grid] target {tid}: eps*={[round(v, 3) for v in eps_star[tid]]} eps_hat={[round(v, 3) for v in rec['eps_hat']]} "
              f"loss={rec['surrogate_loss']:.5f} ({dt:.1f}s)", flush=True)
        json.dump({"method": "grid", "args": vars(a), "runs": records}, open(a.out, "w"))


if __name__ == "__main__":
    main()
