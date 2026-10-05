"""
results/table1_gbfeso_d256.{json,md,tex}: MLGD* against GB-FESO (KL, MMD) on the retrained d256 conditional
samplers (the .json and .tex as make_table1.py --collect writes them). One result file per run (seed):
<runs-dir>/<cell>/<method>_s<seed>.json.

  MLGDstar_s42..46.json    the MLGD* runs (search_mog.py, see experiments/synthetic/README.md, "MLGD*")
  GBFESO-KL_s42..46.json   search_gbfeso.py --loss kl, 30 starts, MLGD*'s sample counts and selection
  GBFESO-MMD_s42..46.json  the same with --loss mmd

Per run: the exact TV between P(Y | x_hat) and G, the relative and raw L2 (make_table1.scale_free_distances,
20,000 Monte Carlo draws, seed 0, as make_table1.py --collect) and ||x_hat - x*|| (sd units of X); per cell and
method: mean +- std over the runs, the trajectories (30 per run) with TV < 0.5 and those in the x* basin.
Run from the repository root, with the cells' data.pt files under --data-root:

    python experiments/synthetic/pipeline/make_table1_d256.py --runs-dir runs/gbfeso_d256 \\
        --data-root runs/table1v2 --out results/table1_gbfeso_d256
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_table1 import _load_mixture, scale_free_distances   # noqa: E402

CELLS = ["x1_y1", "x1_y5", "x1_y10", "x5_y1", "x5_y5", "x5_y10", "x10_y1", "x10_y5", "x10_y10"]
METHODS = ["MLGDstar", "GBFESO-KL", "GBFESO-MMD"]
LABEL = {"MLGDstar": "MLGD*", "GBFESO-KL": "GB-FESO (KL)", "GBFESO-MMD": "GB-FESO (MMD)"}
TEX = {"MLGDstar": "\\mlgd{}$^\\star$", "GBFESO-KL": "GB-FESO (KL)", "GBFESO-MMD": "GB-FESO (\\mmd{})"}

HEADER = """# Table 1 (MLGD*) with the GB-FESO competitor, on the retrained samplers

TV(P(Y | x̂), G), mean ± std over 5 runs (seeds 42–46, 30 trajectories each); lower is better; **bold**: the best in
the row. All three methods use the same retrained conditional sampler per cell (diffusion, 256 units, 4 blocks)
and the Table 1 data and target G (`runs/table1v2/<cell>/data.pt`). MLGD*: zeroth-order guidance, 30 restarts,
refinement 100 steps × 3,000 samples keeping the best iterate, the 5 lowest-loss restarts re-scored 5 × 5,000.
GB-FESO (Zick & Mendels, arXiv:2607.06421): 30 starts from random training designs, no prior over X, their optimizer (201 + 201 Adam steps), MLGD*'s budget: 250 samples per phase-0 step,
3,000 per phase-1 step, each start scored on 1,000, and MLGD*'s selection (the 5 lowest re-scored 5 × 5,000).
GB-FESO (MMD) uses the Table 1 loss, so MLGD* vs GB-FESO (MMD) differs only in the search; GB-FESO (KL) is their
method with their KDE-KL loss.
Setup and commands: `../README.md`, section "MLGD* and GB-FESO". Written by
`pipeline/make_table1_d256.py`.
"""


def dims(cell):
    x, y = cell[1:].split("_y")
    return int(x), int(y)


def seed_of(f):
    return int(re.search(r"_s(\d+)\.json$", f).group(1))


def collect_cell(cell, data, runs_dir):
    """make_table1.py --collect's rows and scenario entry for one cell, from one result file per run."""
    rows, entry = [], {"setting": {"name": cell, "dim_x": dims(cell)[0], "dim_y": dims(cell)[1]}}
    for m in METHODS:
        files = sorted(glob.glob(os.path.join(runs_dir, cell, f"{m}_s*.json")), key=seed_of)
        if not files:
            print(f"  missing: {cell} {m}")
            continue
        atts, trajs, x_star = [], [], None
        for f in files:
            r = json.load(open(f))
            a = r["attempts"][0]
            x_star = r.get("x_star", x_star)
            rec = {"seed": seed_of(f), "x_hat": a["x_hat"], "final_loss": a.get("final_loss"),
                   "select_loss": a.get("select_loss"), "time_s": a.get("time_s"), "sampler": r["args"].get("sampler")}
            rec.update(scale_free_distances(data, a["x_hat"]))
            if x_star is not None:
                rec["dist"] = float(np.linalg.norm(np.array(a["x_hat"]) - np.array(x_star)))
            atts.append(rec)
            trajs += [q["x_hat"] for q in a.get("restarts", [a])]
        tvs = [scale_free_distances(data, p, n_mc=3000, seed=i)["tv"] for i, p in enumerate(trajs)]
        hits = None
        if x_star is not None:   # closer to x* than half the distance to the nearest other cluster centre
            xs = np.array(x_star); c = len(xs)
            centres = np.unique(np.round(np.array([mu[:c] for mu in data["mu_list"]]), 6), axis=0)
            others = [o for o in centres if not np.allclose(o, xs)]
            half = 0.5 * min(np.linalg.norm(o - xs) for o in others) if others else np.inf
            hits = f"{int(sum(np.linalg.norm(np.array(p) - xs) < half for p in trajs))}/{len(trajs)}"
        succ = f"{sum(t < 0.5 for t in tvs)}/{len(tvs)}"

        def ms(k):
            v = [q[k] for q in atts if q.get(k) is not None]
            return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} if v else \
                {"mean": float("nan"), "std": float("nan"), "n": 0}
        summ = {k: ms(k) for k in ("tv", "l2_rel", "l2", "dist", "time_s", "final_loss")}
        entry[m] = {"summary": summ, "budget": 30, "n_attempts": len(atts), "attempts": atts,
                    "hits": hits, "success_tv": succ}
        dx, dy = dims(cell)
        rows.append({"setting": cell, "dim_x": dx, "dim_y": dy, "method": m,
                     "tv_mean": summ["tv"]["mean"], "tv_std": summ["tv"]["std"],
                     "l2rel_mean": summ["l2_rel"]["mean"], "l2rel_std": summ["l2_rel"]["std"],
                     "dist_mean": summ["dist"]["mean"], "dist_std": summ["dist"]["std"],
                     "l2_mean": summ["l2"]["mean"], "l2_std": summ["l2"]["std"], "n": len(atts),
                     "time_mean": summ["time_s"]["mean"], "hits": hits, "success": succ, "resamplings": None})
        print(f"  {cell:8s} {m:11s} n={len(atts)}  TV {summ['tv']['mean']:.3f} +- {summ['tv']['std']:.3f}"
              f"  ({summ['time_s']['mean'] / 60:.0f} min/run)", flush=True)
    return rows, entry


def write_tex(rows, path):
    """make_table1.py's .tex rows: one block per cell, TV | ||x_hat - x*|| | relative L2 | time (s)."""
    tex, last = [], None
    for r in rows:
        key = (r["dim_x"], r["dim_y"])
        if key != last:
            if last is not None:
                tex.append("\\midrule")
            n = sum(1 for q in rows if (q["dim_x"], q["dim_y"]) == key)
            tex.append(f"\\multirow{{{n}}}{{*}}{{$\\dim X={r['dim_x']}$, $\\dim Y={r['dim_y']}$}}")
            last = key
        tex.append(f" & {TEX[r['method']]} & ${r['tv_mean']:.3f} \\pm {r['tv_std']:.3f}$ & ${r['dist_mean']:.3f} \\pm {r['dist_std']:.3f}$ "
                   f"& ${r['l2rel_mean']:.3f} \\pm {r['l2rel_std']:.3f}$ & ${r['time_mean']:.0f}$ \\\\")
    open(path, "w").write("\n".join(tex) + "\n")


def write_md(rows, path):
    """TV per cell and method, then every column."""
    get = {(r["setting"], r["method"]): r for r in rows}
    cells = [c for c in CELLS if any((c, m) in get for m in METHODS)]
    L = [HEADER, "| dim X × dim Y | " + " | ".join(LABEL[m] for m in METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
    for c in cells:
        vals = [get.get((c, m)) for m in METHODS]
        best = min(v["tv_mean"] for v in vals if v is not None)
        cs = []
        for v in vals:
            if v is None:
                cs.append("-"); continue
            s = f"{v['tv_mean']:.3f} ± {v['tv_std']:.3f}"
            cs.append(f"**{s}**" if v["tv_mean"] == best else s)
        L.append("| {}×{} | ".format(*dims(c)) + " | ".join(cs) + " |")
    L.append("| time / run (s), mean | " + " | ".join(
        f"{np.mean([r['time_mean'] for r in rows if r['method'] == m]):.0f}" if any(r["method"] == m for r in rows) else "-"
        for m in METHODS) + " |")
    L += ["", "## All columns", "",
          "| dim X | dim Y | method | TV mean ± std | ‖x̂ − x*‖ (sd units) | relative L2 | time / run (s) | trajectories with TV<0.5 | in x* basin |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dim_x']} | {r['dim_y']} | {LABEL[r['method']]} | {r['tv_mean']:.3f} ± {r['tv_std']:.3f} | "
                 f"{r['dist_mean']:.3f} ± {r['dist_std']:.3f} | {r['l2rel_mean']:.3f} ± {r['l2rel_std']:.3f} | "
                 f"{r['time_mean']:.0f} | {r['success'] or ''} | {r['hits'] or ''} |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs/gbfeso_d256")
    p.add_argument("--data-root", required=True, help="directory holding <cell>/data.pt")
    p.add_argument("--out", default="results/table1_gbfeso_d256")
    a = p.parse_args()

    rows, scenarios = [], {}
    for cell in CELLS:
        if not glob.glob(os.path.join(a.runs_dir, cell, "*_s*.json")):
            print(f"  missing: {cell} (no result files)")
            continue
        data = _load_mixture(os.path.join(a.data_root, cell, "data.pt"))
        if data is None:
            raise SystemExit(f"cannot read {a.data_root}/{cell}/data.pt")
        r, e = collect_cell(cell, data, a.runs_dir)
        rows += r
        scenarios[cell] = e

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump({"rows": rows, "scenarios": scenarios, "args": vars(a)}, f, indent=2)
    write_tex(rows, a.out + ".tex")
    write_md(rows, a.out + ".md")
    print(f"[table] wrote {a.out}.json / .md / .tex")


if __name__ == "__main__":
    main()
