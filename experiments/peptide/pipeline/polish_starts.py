"""Starts file for GB-FESO polishing of MLGD's answers (gbfeso.py opt --starts-file).

Per target: 'm1' = the design MLGD returned, 'm2'.. = the next-best distinct restarts (by their best loss), mapped
from the whitened coordinates of the search to eps in the box [0.21, 1.55] (uniform prior: eps = mid + x * half / sqrt 3).
    python experiments/peptide/pipeline/polish_starts.py --results 'runs/peptide_parity/N5/results/r10_kl_box/*.json' \
        --k 3 --out runs/peptide_parity/N5/polish_starts_r10_kl_box.json
"""
import argparse
import glob
import json
import math
import re

LO, HI = 0.21, 1.55


def to_eps(x):
    mid, half = (LO + HI) / 2, (HI - LO) / 2
    return [min(HI, max(LO, mid + v * half / math.sqrt(3))) for v in x]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--k", type=int, default=3, help="starts per target")
    p.add_argument("--min-dist", type=float, default=0.05, help="min distance (whitened) between starts")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    S = {}
    for f in sorted(glob.glob(a.results)):
        tid = int(re.search(r"_t(\d+)\.json$", f).group(1))
        b = json.load(open(f))["best"]
        xs = [b["x_hat"]]
        for r in sorted(b.get("restarts", []), key=lambda r: r.get("best_loss", r.get("final_loss"))):
            if len(xs) >= a.k:
                break
            if all(math.dist(r["x_hat"], q) > a.min_dist for q in xs):
                xs.append(r["x_hat"])
        S[str(tid)] = {f"m{i + 1}": to_eps(x) for i, x in enumerate(xs)}
    json.dump(S, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(S)} targets, {sum(len(v) for v in S.values())} starts")


if __name__ == "__main__":
    main()
