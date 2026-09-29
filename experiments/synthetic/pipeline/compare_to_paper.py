"""Print the reproduced synthetic tables and figures next to the paper's numbers.

Reads the outputs of the full tiers (or the committed results/ folders with --results)
and, for every cell the paper prints, shows "ours / paper" and flags a cell with "!"
when it lies outside the paper's mean +- std (or, where the paper gives no std, when the
relative difference exceeds --tol).

    python experiments/synthetic/pipeline/compare_to_paper.py             # from */outputs/
    python experiments/synthetic/pipeline/compare_to_paper.py --results   # from */results/

Covered: Table 1 (main_grid), Table 5 (model_quality), Table 8 (ablations), Table 9
(infeasible_target), Table 10 (training_times), Figure 3's TV values (smc_correction) and
Figure 1's basin hits (toy_example). A missing file prints "not run".
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_numbers as P  # noqa: E402

FLAGS = {"n": 0, "bad": 0}


def cell(ours, paper, tol):
    """'ours / paper' with a '!' when ours is outside the paper's range."""
    if ours is None:
        return f"-- / {fmt(paper)}"
    if isinstance(paper, tuple):
        mean, std = paper
        bad = abs(ours - mean) > max(std, 0.005)   # 0.005: the paper's rounding to two decimals
    else:
        bad = abs(ours - paper) > tol * max(abs(paper), 1e-9) + 0.0005
    FLAGS["n"] += 1
    FLAGS["bad"] += bad
    return f"{ours:.3f} / {fmt(paper)}" + (" !" if bad else "")


def fmt(v):
    if not isinstance(v, tuple):
        return f"{v:g}"
    nd = 3 if any(round(x, 2) != round(x, 3) for x in v) else 2   # the paper prints 2 or 3 decimals
    return f"{v[0]:.{nd}f}±{v[1]:.{nd}f}"


def table(title, header, rows):
    print(f"\n## {title}\n")
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header, ["-" * w for w in widths]] + rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def d(exp, sub, results):
    return os.path.join(SYN, exp, "results" if results else "outputs", sub)


def table1(results, tol):
    t = load_json(d("main_grid", "table1.json", results))
    if t is None:
        print("\n## Table 1: not run (no main_grid table1.json)")
        return
    ours = {(r["setting"], r["method"]): r["tv_mean"] for r in t["rows"]}
    rows = []
    for s in P.SCENARIOS:
        key = "x" + s.replace("x", "_y")
        rows.append([s] + [cell(ours.get((key, m)), P.TABLE1[s][m], tol) for m in P.METHODS])
    table("Table 1: TV(P(Y | x_hat), G), ours / paper (mean±std over 5 runs)", ["(dX,dY)"] + P.METHODS, rows)


def table5(results, tol):
    path = d("model_quality", "table5.csv", results)
    if not os.path.exists(path):
        print("\n## Table 5: not run (no model_quality table5.csv)")
        return
    ours = {r["scenario"]: r for r in csv.DictReader(open(path))}
    rows = []
    for s in P.SCENARIOS:
        key = "x" + s.replace("x", "_y")
        r = ours.get(key)
        rows.append([s] + [cell(float(r[c]) if r else None, v, tol) for c, v in zip(P.TABLE5_COLUMNS, P.TABLE5[s])])
    table("Table 5: network quality, ours / paper", ["(dX,dY)"] + P.TABLE5_COLUMNS, rows)


def table8(results, tol):
    path = d("ablations", "table8.csv", results)
    if not os.path.exists(path):
        print("\n## tab:sim_full: not run (no ablations table8.csv)")
        return
    ours = {(r["scenario"], r["method"]): r for r in csv.DictReader(open(path))}
    cols = ["dist", "l2rel", "norefine", "annealed", "oracle"]
    rows = []
    for s, methods in P.SIM_FULL.items():
        key = "x" + s.replace("x", "_y")
        for m, paper in methods.items():
            r = ours.get((key, m))
            row = [s, m]
            for c in cols:
                if c not in paper:
                    row.append("")
                    continue
                v = float(r[f"{c}_mean"]) if r and r.get(f"{c}_mean") not in (None, "") else None
                row.append(cell(v, paper[c], tol))
            rows.append(row)
    table("tab:sim_full: ||x_hat-x*|| and relative L2 after refinement, TV without refinement per arm, ours / paper",
          ["(dX,dY)", "method", "||x_hat-x*||", "relative L2", "no refinement", "annealed", "oracle"], rows)


def table9(results, tol):
    path = d("infeasible_target", "table9.csv", results)
    if not os.path.exists(path):
        print("\n## Table 9: not run (no infeasible_target table9.csv)")
        return
    ours = {r["arm"]: r for r in csv.DictReader(open(path))}
    rows = []
    for arm, (pct, tm, ts) in P.TABLE9.items():
        r = ours.get(arm)
        rows.append([arm, cell(float(r["pct_at_x_bi"]) if r else None, pct, tol),
                     cell(float(r["time_mean"]) if r else None, (tm, ts), tol)])
    table("Table 9: infeasible target, ours / paper (time depends on the GPU)", ["arm", "% at x_bi", "time (s)"], rows)


def table10(results, tol):
    path = d("training_times", "timings.csv", results)
    if not os.path.exists(path):
        print("\n## Training times: not run (the table is commented out in the current paper)")
        return
    ours = {(r["params"], r["network"]): float(r["seconds"]) / 60 for r in csv.DictReader(open(path))}
    rows = [[pf, net, cell(ours.get((pf, net)), minutes, 0.5)] for (pf, net), minutes in P.TABLE10.items()]
    table("Training minutes, ours / paper (the table is commented out in the current paper; NVIDIA L4, approximate; flagged beyond 50%)",
          ["params", "network", "minutes"], rows)


def figure3(results, tol):
    j = load_json(d("smc_correction", "fig_smc_hard.json", results))
    if j is None:
        print("\n## Figure 3: not run (no smc_correction fig_smc_hard.json)")
        return
    rows = []
    for b, (plain, corr) in P.FIGURE3.items():
        r = j["per_beta"].get(b) or j["per_beta"].get(str(float(b)))
        rows.append([b, cell(r["tv_plain_vs_Q"] if r else None, plain, tol), cell(r["tv_corrected_vs_Q"] if r else None, corr, tol)])
    table("Figure 3: TV of the beta-sweep histogram to Q_beta, ours / paper", ["beta", "plain", "corrected"], rows)


def figure1(results, tol):
    j = load_json(d("toy_example", "fig1_hard_sim.json", results))
    if j is None:
        print("\n## Figure 1: not run (no toy_example fig1_hard_sim.json)")
        return
    rows = []
    for m, hits in P.FIGURE1_HITS.items():
        r = j.get(m)
        n = r.get("hits_xstar_basin") if isinstance(r, dict) else None
        rows.append([m, cell(float(n) if n is not None else None, float(hits), tol),
                     f"of {r.get('n_restarts')}" if isinstance(r, dict) else ""])
    table("Figure 1 (a)/(c): restarts (of ten) that reach the x* basin, ours / paper", ["method", "hits", ""], rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", action="store_true", help="read the committed results/ folders instead of outputs/")
    p.add_argument("--tol", type=float, default=0.10, help="relative tolerance where the paper gives no std")
    args = p.parse_args()
    print(f"# Synthetic experiments vs. the paper ({'results/' if args.results else 'outputs/'})")
    for fn in (table1, table5, table8, table9, table10, figure3, figure1):
        fn(args.results, args.tol)
    print(f"\n{FLAGS['bad']} of {FLAGS['n']} compared cells outside the paper's range (marked '!').")


if __name__ == "__main__":
    main()
