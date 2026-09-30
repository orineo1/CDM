"""
Compare the synthetic-experiment outputs of a local/cluster run with the numbers printed in the paper.

    python experiments/synthetic/paper_comparison/compare_to_paper.py
    python experiments/synthetic/paper_comparison/compare_to_paper.py --out my_report      # writes my_report.md / .csv

Reads (every input is optional; a missing one is reported as "not run" and skipped):
  Table 1  TV after refinement            <runs>/<scenario>/{MLGD,MLGD-F,PMLGD,PMLGD-F}.json
  Table 4  networks vs analytic mixture   model_quality/outputs/checks__<scenario>.json, else <runs>/<scenario>/check.json
  Table 5  ||x_hat-x*||, relative L2      same JSONs as Table 1
           TV no refinement, plain        <runs>/<scenario>/<M>_norefine.json (ablation arm), else the pre-refinement
                                          point ('x_before') recorded in the Table 1 JSONs
           TV no refinement, annealed     <runs>/<scenario>/{PMLGD,PMLGD-F}_annealed.json, pre-refinement point
           TV no refinement, oracle       <runs>/<scenario>/{MLGD-X,PMLGD-X}_oracle.json
  Table 6  no-star target                 infeasible_target/outputs/arm_{MMD_n250,L2_n250,L2_n1}.json
  Fig. 1   toy basin hits                 toy_example/outputs/fig1_hard_sim.json
  SMC      TV to Q_beta                   smc_correction/outputs/fig_smc_hard.json

The TV / relative-L2 are computed with make_table1.scale_free_distances, the function that produced the
paper's tables, so a number here is exactly what `make_table1.py --collect` would print.

Verdict per cell (mean +- std over runs): with d = |ours - paper| and s = sqrt((sd_ours^2 + sd_paper^2) / n),
  OK    d <= 0.02 (the paper's rounding) or d <= 2 s
  NEAR  2 s < d <= 3 s
  DIFF  d > 3 s
For Table 4 and the SMC TVs (single numbers) the verdict is on the ratio ours / paper: OK within x1.5 (or
|d| <= 0.02), NEAR within x2, DIFF otherwise. Seeds are fixed, but trained weights differ across GPUs / CUDA /
torch versions, so expect agreement within noise, not to the last digit.
"""
import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SYN))
sys.path.insert(0, os.path.join(SYN, "pipeline"))

SCENARIOS = ["x1_y1", "x1_y5", "x1_y10", "x5_y1", "x5_y5", "x5_y10", "x10_y1", "x10_y5", "x10_y10"]
METHODS = ["MLGD", "MLGD-F", "PMLGD", "PMLGD-F"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper", default=os.path.join(HERE, "paper_numbers.json"))
    p.add_argument("--runs", default=os.path.join(SYN, "main_grid", "outputs", "runs"),
                   help="runs dir of the grid (Table 1 and the ablation arms)")
    p.add_argument("--checks", default=os.path.join(SYN, "model_quality", "outputs"))
    p.add_argument("--infeasible", default=os.path.join(SYN, "infeasible_target", "outputs"))
    p.add_argument("--toy-sim", default=os.path.join(SYN, "toy_example", "outputs", "fig1_hard_sim.json"))
    p.add_argument("--smc", default=os.path.join(SYN, "smc_correction", "outputs", "fig_smc_hard.json"))
    p.add_argument("--out", default=os.path.join(HERE, "outputs", "comparison"),
                   help="stem of the report: <out>.md and <out>.csv")
    p.add_argument("--n-mc", type=int, default=20000, help="Monte-Carlo draws for TV (make_table1 uses 20000)")
    return p.parse_args()


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def pm(s):
    """'0.02 +- 0.01' -> (0.02, 0.01); None -> None."""
    if s is None:
        return None
    m = re.match(r"\s*([-\d.]+)\s*\+-\s*([-\d.]+)", s)
    return float(m.group(1)), float(m.group(2))


def fmt(ms, nd=3):
    return "--" if ms is None else f"{ms[0]:.{nd}f} +- {ms[1]:.{nd}f}"


def verdict_ms(ours, paper, n=5):
    if ours is None or paper is None:
        return ""
    d = abs(ours[0] - paper[0])
    s = math.sqrt((ours[1] ** 2 + paper[1] ** 2) / n)
    if d <= 0.02 or d <= 2 * s:
        return "OK"
    return "NEAR" if d <= 3 * s else "DIFF"


def verdict_ratio(ours, paper):
    if ours is None or paper is None:
        return ""
    if abs(ours - paper) <= 0.02:
        return "OK"
    r = max(ours, 1e-9) / max(paper, 1e-9)
    r = max(r, 1 / r)
    return "OK" if r <= 1.5 else ("NEAR" if r <= 2 else "DIFF")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


_MIX = {}


def mixture(scn, runs):
    if scn not in _MIX:
        from make_table1 import _load_mixture
        _MIX[scn] = _load_mixture(os.path.join(runs, scn, "data.pt"))
    return _MIX[scn]


def tv_stats(scn, runs, points, n_mc, key="tv"):
    """mean/std over runs of TV (or l2_rel) between P(Y|X=x) and G for each x in points."""
    from make_table1 import scale_free_distances
    data = mixture(scn, runs)
    if data is None or not points:
        return None
    v = [scale_free_distances(data, x, n_mc=n_mc)[key] for x in points]
    return float(np.mean(v)), float(np.std(v))


def pre_refine_points(res):
    """What each attempt would have returned without refinement: among its trajectories (restarts or
    particles), the pre-refinement point with the smallest pre-refinement loss. None if the run did not refine."""
    pts = []
    for a in res["attempts"]:
        if a.get("restarts"):                                   # lgd: one refine record per restart
            cand = [((q.get("refine") or {}).get("x_before"), (q.get("refine") or {}).get("final_loss_before"))
                    for q in a["restarts"]]
        else:                                                   # particles: one record for the population
            rf = a.get("refine") or {}
            xb, lb = rf.get("x_before"), rf.get("final_losses_before")
            if xb is None:
                return None
            xb = np.asarray(xb, dtype=float).reshape(len(lb) if lb else 1, -1)
            cand = list(zip(xb.tolist(), lb or [0.0]))
        if not cand or any(x is None for x, _ in cand):
            return None
        x, _ = min(cand, key=lambda c: float("inf") if c[1] is None else c[1])
        pts.append(np.asarray(x, dtype=float).reshape(-1).tolist())
    return pts


# ----------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.md, self.rows, self.counts = [], [], {"OK": 0, "NEAR": 0, "DIFF": 0}

    def h(self, s):
        self.md += ["", s, ""]

    def table(self, header, lines):
        self.md.append("| " + " | ".join(header) + " |")
        self.md.append("|" + "---|" * len(header))
        self.md += ["| " + " | ".join(str(c) for c in ln) + " |" for ln in lines]

    def row(self, table, scenario, method, column, paper, ours, v):
        if v in self.counts:
            self.counts[v] += 1
        self.rows.append({"table": table, "scenario": scenario, "method": method, "column": column,
                          "paper": paper, "ours": ours, "verdict": v})


def table1_and_5(P, R, args):
    runs = args.runs
    t1_lines, t5_lines = [], []
    for scn in SCENARIOS:
        for m in METHODS:
            res = load(os.path.join(runs, scn, f"{m}.json"))
            paper_tv = pm(P["table1"][scn][m])
            if res is None:
                t1_lines.append([scn, m, fmt(paper_tv, 2), "not run", ""])
                continue
            pts = [a["x_hat"] for a in res["attempts"]]
            tv = tv_stats(scn, runs, pts, args.n_mc)
            v = verdict_ms(tv, paper_tv)
            R.row("Table 1", scn, m, "TV refined", fmt(paper_tv, 2), fmt(tv), v)
            t1_lines.append([scn, m, fmt(paper_tv, 2), fmt(tv), v])

            # Table 5: ||x_hat - x*||, relative L2 (refined)
            p5 = P["table5"][scn][m]
            xs = res.get("x_star")
            dist = None
            if xs is not None:
                dd = [float(np.linalg.norm(np.asarray(x) - np.asarray(xs))) for x in pts]
                dist = (float(np.mean(dd)), float(np.std(dd)))
            l2 = tv_stats(scn, runs, pts, args.n_mc, key="l2_rel")
            cells = [scn, m]
            for col, ours, paper in (("||x_hat-x*||", dist, pm(p5[0])), ("relative L2", l2, pm(p5[1]))):
                v = verdict_ms(ours, paper)
                R.row("Table 5", scn, m, col, fmt(paper), fmt(ours), v)
                cells += [f"{fmt(paper)} / {fmt(ours)}", v]
            # plain, no refinement
            plain, src = None, ""
            nr = load(os.path.join(runs, scn, f"{m}_norefine.json"))
            if nr is not None:
                plain, src = tv_stats(scn, runs, [a["x_hat"] for a in nr["attempts"]], args.n_mc), "arm"
            else:
                xb = pre_refine_points(res)
                if xb is not None:
                    plain, src = tv_stats(scn, runs, xb, args.n_mc), "x_before"
            v = verdict_ms(plain, pm(p5[2]))
            R.row("Table 5", scn, m, f"TV plain ({src or 'not run'})", fmt(pm(p5[2]), 2), fmt(plain), v)
            cells += [f"{fmt(pm(p5[2]), 2)} / {fmt(plain) if plain else 'not run'}{' ('+src+')' if src else ''}", v]
            # annealed resampling (particle methods), no refinement
            ann = None
            if p5[3] is not None:
                ar = load(os.path.join(runs, scn, f"{m}_annealed.json"))
                if ar is not None:
                    xb = pre_refine_points(ar) or [a["x_hat"] for a in ar["attempts"]]
                    ann = tv_stats(scn, runs, xb, args.n_mc)
                v = verdict_ms(ann, pm(p5[3]))
                R.row("Table 5", scn, m, "TV annealed", fmt(pm(p5[3]), 2), fmt(ann), v)
                cells += [f"{fmt(pm(p5[3]), 2)} / {fmt(ann) if ann else 'not run'}", v]
            else:
                cells += ["--", ""]
            # oracle sampler
            orc = None
            if p5[4] is not None:
                o = load(os.path.join(runs, scn, f"{m.replace('-F', '')}-X_oracle.json"))
                if o is not None:
                    orc = tv_stats(scn, runs, [a["x_hat"] for a in o["attempts"]], args.n_mc)
                v = verdict_ms(orc, pm(p5[4]))
                R.row("Table 5", scn, m, "TV oracle", fmt(pm(p5[4]), 2), fmt(orc), v)
                cells += [f"{fmt(pm(p5[4]), 2)} / {fmt(orc) if orc else 'not run'}", v]
            else:
                cells += ["--", ""]
            t5_lines.append(cells)
    R.h("## Table 1 -- TV(P(Y|X=x_hat), G) after refinement (mean +- std over runs)")
    R.table(["scenario", "method", "paper", "ours", "verdict"], t1_lines)
    R.h("## Table 5 -- appendix companion (each cell: paper / ours)")
    R.md.append("`plain` comes from the `_norefine` ablation arm when it exists, otherwise from the pre-refinement "
                "point stored in the Table 1 run (`x_before`); both are the returned x_hat before refinement.")
    R.md.append("")
    R.table(["scenario", "method", "||x_hat-x*||", "", "relative L2", "", "TV plain", "", "TV annealed", "",
             "TV oracle", ""], t5_lines)


def table4(P, R, args):
    cols = P["table4"]["_columns"]
    lines = []
    for scn in SCENARIOS:
        c = load(os.path.join(args.checks, f"checks__{scn}.json")) or load(os.path.join(args.runs, scn, "check.json"))
        paper = P["table4"][scn]
        if c is None:
            lines.append([scn] + [f"{p:.3f} / not run" for p in paper])
            continue
        try:
            fisher = max(o["rel_fisher"] for o in c["score"] if o["t"] > 1)
        except (KeyError, ValueError):
            fisher = None

        def pt(k):
            try:
                return c[k]["points"]["x*"]["mean"]
            except (KeyError, TypeError):
                return None

        ours = [fisher, (c.get("prior") or {}).get("swd_norm"),
                pt("cond_diffusion"), (c.get("cond_diffusion") or {}).get("swd_norm_random_x_mean"),
                pt("cond_cm"), (c.get("cond_cm") or {}).get("swd_norm_random_x_mean")]
        cells = [scn]
        for col, p, o in zip(cols, paper, ours):
            v = verdict_ratio(o, p)
            R.row("Table 4", scn, "", col, f"{p:.3f}", "--" if o is None else f"{o:.3f}", v)
            cells.append(f"{p:.3f} / {'--' if o is None else f'{o:.3f}'} {v}")
        lines.append(cells)
    R.h("## Table 4 -- trained networks vs the analytic mixture (each cell: paper / ours verdict)")
    R.table(["scenario"] + cols, lines)


def table6(P, R, args):
    lines = []
    for arm, p in ((k, v) for k, v in P["table6"].items() if not k.startswith("_")):
        res = load(os.path.join(args.infeasible, f"arm_{arm}.json"))
        pt = pm(p["time_s"])
        if res is None:
            lines.append([arm, f"{p['pct_xbi']}%", "not run", "", fmt(pt, 2), "not run"])
            continue
        xs = [float(np.asarray(a["x_hat"]).reshape(-1)[0]) for a in res["attempts"]]
        n = len(xs)
        pct = 100.0 * sum(abs(x + 2) < abs(x - 2) for x in xs) / n
        t = [a["time_s"] for a in res["attempts"]]
        tm = (float(np.mean(t)), float(np.std(t)))
        # % of n restarts: binomial sd of the paper's rate
        sd = 100 * math.sqrt(max(p["pct_xbi"] / 100 * (1 - p["pct_xbi"] / 100), 1 / n) / n)
        d = abs(pct - p["pct_xbi"])
        v = "OK" if d <= 2 * sd else ("NEAR" if d <= 3 * sd else "DIFF")
        R.row("Table 6", arm, "MLGD-F", "% at x_bi", f"{p['pct_xbi']}%", f"{pct:.0f}% (n={n})", v)
        R.row("Table 6", arm, "MLGD-F", "time (s)", fmt(pt, 2), fmt(tm, 2), "info")
        lines.append([arm, f"{p['pct_xbi']}%", f"{pct:.0f}% (n={n})", v, fmt(pt, 2), fmt(tm, 2)])
    R.h("## Table 6 -- no-star target (time depends on the GPU; not scored)")
    R.table(["arm", "paper % at x_bi", "ours", "verdict", "paper time (s)", "ours time (s)"], lines)


def toy(P, R, args):
    sim = load(args.toy_sim)
    R.h("## Figure 1 -- toy example, trajectories in the x* basin")
    if sim is None:
        R.md.append(f"not run (`{os.path.relpath(args.toy_sim, REPO)}` missing)")
        return
    lines = []
    for m, p in P["fig1_toy"]["hits_out_of_10"].items():
        r = sim.get(m)
        if r is None:
            lines.append([m, f"{p}/10", "not run", ""])
            continue
        n = r.get("n_restarts") or r.get("n_particles")
        h = r.get("hits_xstar_basin")
        # one run of ten trajectories: only the rate is comparable, within binomial noise
        rate_p, rate_o = p / 10, h / n
        sd = math.sqrt(max(rate_p * (1 - rate_p), 0.05) / n)
        v = "OK" if abs(rate_o - rate_p) <= 2 * sd else ("NEAR" if abs(rate_o - rate_p) <= 3 * sd else "DIFF")
        R.row("Fig 1", "toy hard", m, "hits in x* basin", f"{p}/10", f"{h}/{n}", v)
        lines.append([m, f"{p}/10", f"{h}/{n}", v])
    R.table(["method", "paper", "ours", "verdict"], lines)


def smc(P, R, args):
    out = load(args.smc)
    R.h("## Twisted-SMC correction -- TV(histogram, Q_beta)")
    if out is None:
        R.md.append(f"not run (`{os.path.relpath(args.smc, REPO)}` missing)")
        return
    per = {float(k): v for k, v in out.get("per_beta", {}).items()}
    lines = []
    for b, p in ((k, v) for k, v in P["smc"].items() if not k.startswith("_")):
        o = per.get(float(b))
        if o is None:
            lines.append([b, p["plain"], "not run", "", p["corrected"], "not run", ""])
            continue
        v1 = verdict_ratio(o["tv_plain_vs_Q"], p["plain"])
        v2 = verdict_ratio(o["tv_corrected_vs_Q"], p["corrected"])
        R.row("SMC", f"beta={b}", "", "plain", p["plain"], f"{o['tv_plain_vs_Q']:.3f}", v1)
        R.row("SMC", f"beta={b}", "", "corrected", p["corrected"], f"{o['tv_corrected_vs_Q']:.3f}", v2)
        lines.append([b, p["plain"], f"{o['tv_plain_vs_Q']:.3f}", v1, p["corrected"], f"{o['tv_corrected_vs_Q']:.3f}", v2])
    R.table(["beta", "paper plain", "ours plain", "", "paper corrected", "ours corrected", ""], lines)


def main():
    args = parse_args()
    P = load(args.paper)
    R = Report()
    R.md.append("# Synthetic experiments: paper vs this run")
    table1_and_5(P, R, args)
    table4(P, R, args)
    table6(P, R, args)
    toy(P, R, args)
    smc(P, R, args)
    c = R.counts
    summary = f"**Summary:** {c['OK']} OK, {c['NEAR']} NEAR, {c['DIFF']} DIFF (cells that were run)."
    R.md[1:1] = ["", summary, "", "OK = within 2 standard errors of the paper (or within its rounding); "
                 "NEAR = within 3; DIFF = further. See the docstring of compare_to_paper.py."]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(R.md) + "\n")
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table", "scenario", "method", "column", "paper", "ours", "verdict"])
        w.writeheader()
        w.writerows(R.rows)
    print("\n".join(R.md))
    print(f"\nwrote {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
