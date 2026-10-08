"""Exact evaluation of the deceptive-target benchmark. Target summary = (count distribution G_count, '+' population
p_k* of the x*-mode residue k). Per run: count TV, |p_k - p_k*|, summary distance D = max of the two (solved at
D <= 0.02 / 0.05), loss L = KL(count) + wP (p_k - p_k*)^2, global basin found (x_hat's mode residue = k),
|x - x*| (kT), ESS, loss evaluations, time."""
import argparse, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rw import Frames

RES = np.array([[(k >> (2 - i)) & 1 for i in range(3)] for k in range(8)], float)
NPLUS = np.eye(4)[np.array([bin(k).count("1") for k in range(8)])]
ORDER = ["oracle (x*)", "GBFESO-exact-x0", "GBFESO-exact-rand4", "GBFESO-exact-randEq", "MLGD-exact", "MLGD-exact+polish"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default="runs/trialanine/decoy/targets.json")
    p.add_argument("--results", default="runs/trialanine/decoy/results")
    p.add_argument("--frames", default="runs/trialanine/prod_x0_w*.npz")
    p.add_argument("--out", default="experiments/trialanine/results/decoy")
    a = p.parse_args()
    meta = json.load(open(a.meta)); wP = meta["wP"]; scale = meta["x_scale_kJ"]
    F = Frames(glob.glob(a.frames), thin=1); kT = F.kT
    T = {t["id"]: t for t in meta["targets"]}

    def score(x_kJ, t):
        w = F.weights(x_kJ); P = w @ F.onehot; pc = P @ NPLUS; plus = P @ RES; k = t["mode_star"]
        G = np.array(t["G_count"]); m = G > 0
        ctv = 0.5 * np.abs(pc - G).sum(); dp = abs(plus[k] - t["plus_marginals"][k])
        L = float((G[m] * (np.log(G[m]) - np.log(np.maximum(pc[m], 1e-300)))).sum() + wP * dp ** 2)
        return dict(count_tv=float(ctv), dp=float(dp), D=float(max(ctv, dp)), loss=L, mode=int(np.argmax(plus)),
                    global_basin=bool(int(np.argmax(plus)) == k), dist_kT=float(np.linalg.norm(x_kJ - np.array(t["x_star_kJ"])) / kT),
                    ess=float(1 / (w ** 2).sum()))
    rows, ev = [], {}
    for i, t in T.items():
        rows.append(dict(method="oracle (x*)", target=i, time_s=0.0, loss_evals=0, **score(np.array(t["x_star_kJ"]), t)))
    for f in sorted(glob.glob(os.path.join(a.results, "*_t*.json"))):
        m = re.match(r"(.+)_t(\d+)\.json", os.path.basename(f)); name, i = m.group(1), int(m.group(2))
        J = json.load(open(f))
        x = np.array(J["attempts"][0]["x_hat"] if "attempts" in J else J["best"]["x_hat"])
        t_s = J["attempts"][0]["time_s"] if "attempts" in J else J["time_s"]
        ev[(name, i)] = (t_s, J["n_loss_evals"])
        rows.append(dict(method=name, target=i, time_s=t_s, loss_evals=J["n_loss_evals"], n_starts=J.get("n_starts"),
                         x_hat_std=x.tolist(), **score(x * scale, T[i])))
    for r in rows:
        if r["method"].endswith("+polish"):
            t0, e0 = ev.get((r["method"][:-7], r["target"]), (np.nan, 0))
            r["time_s"] += t0; r["loss_evals"] += e0
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(wP=wP, rows=rows), open(os.path.join(a.out, "per_run.json"), "w"), indent=1)
    names = [n for n in ORDER if any(r["method"] == n for r in rows)]
    L = ["| method | targets | solved (D <= 0.02) | solved (D <= 0.05) | global basin found | median D | median loss | "
         "median abs(x - x*) (kT) | median loss evals | median time (s) | ESS < 500 |", "|" + "---|" * 11]
    for n in names:
        R = [r for r in rows if r["method"] == n]; D = np.array([r["D"] for r in R])
        L.append(f"| {n} | {len(R)} | {np.mean(D <= 0.02):.0%} | {np.mean(D <= 0.05):.0%} | {np.mean([r['global_basin'] for r in R]):.0%} | "
                 f"{np.median(D):.3f} | {np.median([r['loss'] for r in R]):.4f} | {np.median([r['dist_kT'] for r in R]):.2f} | "
                 f"{np.median([r['loss_evals'] for r in R]):.0f} | {np.median([r['time_s'] for r in R]):.0f} | {sum(r['ess'] < 500 for r in R)} |")
    open(os.path.join(a.out, "results_table.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
