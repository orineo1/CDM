"""Candidate hidden designs for the deceptive-target benchmark: x* ~ N(0, (sigma kT)^2) truncated to the box,
kept when one residue carries most of the '+' population (max per-residue '+' marginal >= --min-plus), plus the
engineered single- and double-'+' designs of make_data_tri.py. Writes a meta JSON in make_data_tri's format
(targets with id, x_star_std, mode_star, G_count, plus_marginals, ess) for decoy_explore.py."""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import system as S
from rw import Frames

p = argparse.ArgumentParser()
p.add_argument("--meta", default="runs/trialanine/cdm/meta.json", help="for the design scale and box")
p.add_argument("--frames", default="runs/trialanine/prod_x0_w*.npz")
p.add_argument("--n", type=int, default=48)
p.add_argument("--sigma", type=float, default=1.5)
p.add_argument("--min-plus", type=float, default=0.5)
p.add_argument("--min-ess", type=float, default=1000)
p.add_argument("--seed", type=int, default=11)
p.add_argument("--out", default="runs/trialanine/decoy/candidates.json")
a = p.parse_args()
meta = json.load(open(a.meta)); scale, box = meta["x_scale_kJ"], meta["box_std"]
F = Frames(glob.glob(a.frames), thin=2); kT = F.kT
RES = np.array([[(k >> (2 - i)) & 1 for i in range(3)] for k in range(8)], float)
NPLUS = np.eye(4)[np.array([bin(k).count("1") for k in range(8)])]
rng = np.random.default_rng(a.seed)
cands = [(f"eng_plus{i + 1}", 0) for i in range(3)]
out, tries = [], 0
eng = []
for i in range(3):
    x = np.zeros(S.D); x[4 * i + 1] = -3.5 * kT; eng.append((f"eng_plus{i + 1}", x))
while len(out) < a.n and tries < 20000:
    tries += 1
    if eng:
        kind, x = eng.pop(0)
    else:
        kind = "random"
        x = rng.normal(0, a.sigma * kT, S.D)
        if (np.abs(x / scale) > box).any():
            continue
    w = F.weights(x); P = w @ F.onehot; plus = P @ RES; ess = float(1 / (w ** 2).sum())
    if plus.max() < a.min_plus or ess < a.min_ess:
        continue
    out.append(dict(id=len(out), kind=kind, x_star_kJ=x.tolist(), x_star_std=(x / scale).tolist(), G_count=(P @ NPLUS).tolist(),
                    G_states=P.tolist(), plus_marginals=plus.tolist(), mode_star=int(np.argmax(plus)), ess=ess))
m2 = {k: v for k, v in meta.items() if k != "targets"}
m2.update(targets=out, candidate_args=vars(a), candidate_tries=tries)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(m2, open(a.out, "w"), indent=1)
print(len(out), "candidates from", tries, "draws; modes", np.bincount([t["mode_star"] for t in out], minlength=3).tolist())
