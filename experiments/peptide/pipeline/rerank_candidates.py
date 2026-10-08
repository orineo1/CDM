"""Inference-time re-ranking of MLGD's candidates with the true simulator.

build: MLGD's top-k distinct candidates per target (as polish_starts.py) -> one results file in GB-FESO's format
(method '<name>', one run per candidate 'c1'.., m1 = MLGD's own pick) for evaluate.py collect / slurm_eval.sh.
table: given two independent evaluations of the candidates (simulation seeds A and B), pick per target the
candidate with the lowest simulated KDE-KL to the target in A (the loss, from simulation instead of the
surrogate) and report its dF error in B; also MLGD's own pick (m1) and the best candidate by dF in B (oracle).
    python rerank_candidates.py build --results 'runs/peptide_parity/N5/results/r10_kl_box/*.json' --k 5 --out X.json
    python rerank_candidates.py table --sel per_run_A.json --score per_run_B.json --held-out 10
"""
import argparse
import json
import math

import numpy as np

import polish_starts


def cmd_build(a):
    import sys
    sys.argv = ["polish_starts", "--results", a.results, "--k", str(a.k), "--out", a.out + ".starts"]
    polish_starts.main()
    S = json.load(open(a.out + ".starts"))
    runs = [{"target_id": int(t), "start": c, "start_eps": e, "eps_hat": e, "surrogate_loss": 0.0, "time_s": 0.0}
            for t, cs in S.items() for c, e in cs.items()]
    json.dump({"method": a.name, "runs": runs}, open(a.out, "w"))
    print(f"wrote {a.out}: {len(runs)} candidates")


def rows(path, name):
    R = {}
    for r in json.load(open(path)):
        if r["method"].startswith(name + " ["):
            R.setdefault(r["target_id"], {})[r["method"][len(name) + 2:-1]] = r
    return R


def cmd_table(a):
    A, B = rows(a.sel, a.name), rows(a.score, a.name)
    out = {"MLGD's pick (m1)": [], "re-ranked by simulated KL (A), scored in B": [], "best candidate by dF in B (oracle)": []}
    for t in sorted(set(A) & set(B)):
        if not B[t]["m1"].get("multi_basin", True):
            continue
        cand = [c for c in A[t] if c in B[t]]
        pick = min(cand, key=lambda c: A[t][c]["kl"])
        out["MLGD's pick (m1)"].append((t, B[t]["m1"]["df_median_abs_error"]))
        out["re-ranked by simulated KL (A), scored in B"].append((t, B[t][pick]["df_median_abs_error"]))
        out["best candidate by dF in B (oracle)"].append((t, min(B[t][c]["df_median_abs_error"] for c in cand)))
    print("| method | held-out mean | test mean | test median | test <=0.3 | all mean | n |")
    print("|---|---|---|---|---|---|---|")
    for k, v in out.items():
        h = [e for t, e in v if t < a.held_out]; s = [e for t, e in v if t >= a.held_out]; al = [e for _, e in v]
        print(f"| {k} | {np.mean(h):.3f} | {np.mean(s):.3f} | {np.median(s):.3f} | {np.mean(np.array(s) <= 0.3):.0%} | {np.mean(al):.3f} | {len(al)} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--results", required=True); b.add_argument("--k", type=int, default=5)
    b.add_argument("--name", default="MLGD_r10_kl_box rerank"); b.add_argument("--out", required=True)
    t = sub.add_parser("table"); t.add_argument("--sel", required=True); t.add_argument("--score", required=True)
    t.add_argument("--name", default="MLGD_r10_kl_box rerank"); t.add_argument("--held-out", type=int, default=10)
    a = p.parse_args()
    {"build": cmd_build, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
