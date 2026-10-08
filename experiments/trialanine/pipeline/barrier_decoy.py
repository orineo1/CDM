"""Barriers between the global minimizer and the deceptive local minima, on the exact loss: for each target, the
best and up to --n-bad worst-basin endpoints of the equal-compute random multistart (GBFESO-exact-randEq) are joined
by an elastic band (paths.band_path, dense evaluation of the relaxed path). Reports per-start basin statistics
(fraction of single local runs ending with summary distance D > 0.05) and the barrier height above the bad minimum."""
import argparse, glob, json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from search_exact_tri import ExactCountLoss
from paths import band_path

p = argparse.ArgumentParser()
p.add_argument("--dir", default="runs/trialanine/decoy")
p.add_argument("--frames", default="runs/trialanine/prod_x0_w*.npz")
p.add_argument("--n-bad", type=int, default=2)
p.add_argument("--out", default="experiments/trialanine/results/decoy/barriers.json")
a = p.parse_args()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
meta = json.load(open(os.path.join(a.dir, "targets.json"))); wP = meta["wP"]
out = []
for t in meta["targets"]:
    f = os.path.join(a.dir, "results", f"GBFESO-exact-randEq_t{t['id']:03d}.json")
    if not os.path.exists(f):
        continue
    J = json.load(open(f)); rs = J["restarts"]
    k = t["mode_star"]
    L = ExactCountLoss(a.frames, t["G_count"], meta["x_scale_kJ"], k=k, p_star=t["plus_marginals"][k], wP=wP, thin=2)

    def obj(x):
        xt = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        v = L(xt); v.backward()
        return float(v), xt.grad.numpy().astype(float)
    Ls = np.array([q["final_loss"] for q in rs]); best = int(np.argmin(Ls))
    bad = [i for i in np.argsort(-Ls) if Ls[i] > Ls[best] + 0.005][:a.n_bad]
    rec = dict(id=t["id"], n_starts=len(rs), L_best=float(Ls[best]), frac_starts_L_gap_gt_0005=float(np.mean(Ls > Ls[best] + 0.005)),
               L_bad=[float(Ls[i]) for i in bad], bands=[])
    for i in bad:
        bp = band_path(obj, np.array(rs[best]["x_hat"]), np.array(rs[i]["x_hat"]), 1.0, n=16, springs=(1e-3, 1e-1), maxiter=300,
                       bounds=[(-2.0, 2.0)] * 12)
        rec["bands"].append(dict(L_bad=float(Ls[i]), **bp, barrier_above_bad=bp["band_max_dense"] - float(Ls[i])))
    out.append(rec)
    print(t["id"], json.dumps({k_: v for k_, v in rec.items() if k_ != "bands"}), [round(b["barrier_above_bad"], 4) for b in rec["bands"]], flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
