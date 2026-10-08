"""Per-trajectory statistics of the deceptive benchmark: for every individual start / restart (GB-FESO random
starts of the equal-compute row, GB-FESO from x0, MLGD's 10 guided restarts after refinement), the summary
distance D and whether it ended in the global basin (mode residue = x*'s). Shows whether MLGD's guided
trajectories land in the global basin more often than plain local descent from prior draws."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rw import Frames
RES = np.array([[(k >> (2 - i)) & 1 for i in range(3)] for k in range(8)], float)
NPLUS = np.eye(4)[np.array([bin(k).count("1") for k in range(8)])]
D0 = "runs/trialanine/decoy"
meta = json.load(open(f"{D0}/targets.json")); scale = meta["x_scale_kJ"]
F = Frames(glob.glob("runs/trialanine/prod_x0_w*.npz"), thin=2)
stat = {"GB-FESO random start (randEq)": [], "MLGD restart": [], "GB-FESO x0 start": []}
for t in meta["targets"]:
    k = t["mode_star"]; G = np.array(t["G_count"]); tid = t["id"]

    def D(x):
        w = F.weights(np.array(x) * scale); P = w @ F.onehot
        return max(0.5 * np.abs(P @ NPLUS - G).sum(), abs((P @ RES)[k] - t["plus_marginals"][k])), int(np.argmax(P @ RES)) == k
    for name, f, key in [("GB-FESO random start (randEq)", "GBFESO-exact-randEq", "restarts"), ("GB-FESO x0 start", "GBFESO-exact-x0", "restarts"),
                         ("MLGD restart", "MLGD-exact", None)]:
        J = json.load(open(f"{D0}/results/{f}_t{tid:03d}.json"))
        rs = J["restarts"] if key else J["attempts"][0]["restarts"]
        for q in rs:
            d, g = D(q["x_hat"]); stat[name].append(dict(target=tid, D=d, glob=g))
out = {}
for n, v in stat.items():
    Dv = np.array([q["D"] for q in v]); g = np.array([q["glob"] for q in v])
    out[n] = dict(n=len(v), frac_D_le_002=float(np.mean(Dv <= 0.02)), frac_D_le_005=float(np.mean(Dv <= 0.05)),
                  frac_global_basin=float(g.mean()), frac_wrong_basin=float(1 - g.mean()))
    print(f"{n:32s} n={len(v):4d}  D<=0.02 {out[n]['frac_D_le_002']:.0%}  D<=0.05 {out[n]['frac_D_le_005']:.0%}  global basin {g.mean():.0%}")
json.dump(dict(summary=out, per_start=stat), open("experiments/trialanine/results/decoy/per_start.json", "w"), indent=1)
