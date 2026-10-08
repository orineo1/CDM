"""Select the deceptive targets from the decoy_explore.py checks of the candidates at one restraint weight:
keep candidates where the best start reaches the global optimum (summary distance D <= 0.02) and a fraction
in [--lo, --hi] of the random starts ends in a strictly worse local minimum (D > 0.05). Writes the benchmark
meta (make_data_tri format + wP + deceptiveness statistics) and a dummy target sample for search_mog."""
import argparse, glob, json, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dir", default="runs/trialanine/decoy")
p.add_argument("--wP", type=float, required=True)
p.add_argument("--lo", type=float, default=0.3)
p.add_argument("--hi", type=float, default=0.8)
p.add_argument("--max", type=int, default=30)
a = p.parse_args()
cand = json.load(open(os.path.join(a.dir, "candidates.json")))
C = {t["id"]: t for t in cand["targets"]}
sel, stats = [], []
for f in sorted(glob.glob(os.path.join(a.dir, "explore", f"c*_w{a.wP}.json"))):
    for r in json.load(open(f))["targets"]:
        ends = r["ends"]; D = np.array([max(e["count_tv"], abs(e["dP"])) for e in ends]); L = np.array([e["L"] for e in ends])
        bad = D[1:] > 0.05
        st = dict(cand_id=r["id"], frac_bad=float(bad.mean()), reach=bool(D.min() <= 0.02), x0_bad=bool(D[0] > 0.05),
                  L_global=float(L.min()), L_bad_median=float(np.median(L[1:][bad])) if bad.any() else None,
                  D_bad_median=float(np.median(D[1:][bad])) if bad.any() else None,
                  modes_of_bad=np.bincount([e["mode"] for e, b in zip(ends[1:], bad) if b], minlength=3).tolist())
        stats.append(st)
        if st["reach"] and a.lo <= st["frac_bad"] <= a.hi:
            sel.append(st)
sel = sel[:a.max]
targets = []
for j, st in enumerate(sel):
    t = dict(C[st["cand_id"]]); t.update(id=j, cand_id=st["cand_id"], deceptive=st); targets.append(t)
meta = {k: v for k, v in cand.items() if k != "targets"}
meta.update(targets=targets, wP=a.wP, selection=vars(a), all_candidates=stats)
json.dump(meta, open(os.path.join(a.dir, "targets.json"), "w"), indent=1)
np.save(os.path.join(a.dir, "dummy_target.npy"), np.zeros((16, 12), np.float32))
fb = [s["frac_bad"] for s in stats]
print(f"wP={a.wP}: {len(stats)} candidates checked, {len(sel)} selected; frac_bad over candidates: median {np.median(fb):.2f}, "
      f"in window {sum(a.lo <= v <= a.hi for v in fb)}, reach {sum(s['reach'] for s in stats)}; "
      f"x0 fails on {sum(s['x0_bad'] for s in sel)}/{len(sel)} selected; median frac_bad selected "
      f"{np.median([s['frac_bad'] for s in sel]) if sel else float('nan'):.2f}")
