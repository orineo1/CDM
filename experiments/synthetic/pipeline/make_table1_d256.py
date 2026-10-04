"""
results/table1_gbfeso_d256.{json,md,tex}: MLGD* against GB-FESO (KL, MMD) on the retrained d256 conditional
samplers, one result file per run (seed), as Or's runs/night/grid files: <runs-dir>/<cell>/<method>_s<seed>.json.

  MLGDstar_s42..46.json    symlinks to Or's runs: runs/night/grid/<cell>/G_zo_s*.json; at (10, 1)
                           runs/night/x10_y1/E6_sel5000_n3000_s*.json (the same recipe)
  GBFESO-KL_s42..46.json   search_gbfeso.py --loss kl, 30 starts, MLGD*'s sample counts and selection
  GBFESO-MMD_s42..46.json  the same with --loss mmd

Per run: the exact TV between P(Y | x_hat) and G (make_table1.scale_free_distances, 20,000 Monte Carlo draws,
seed 0, as make_table1.py --collect and summarize_runs.py), ||x_hat - x*|| (sd units of X), relative L2 and
the run time; per cell and method: mean +- std over the runs, and the share of all trajectories (30 per run)
with TV < 0.5. Run from the repository root, with the cells' data.pt files under --data-root:

    python experiments/synthetic/pipeline/make_table1_d256.py --runs-dir runs/gbfeso_d256 \\
        --data-root /sci/labs/orzuk/orzuk/github/CDM/runs/table1v2 --out experiments/synthetic/results/table1_gbfeso_d256
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
METHODS = [("MLGDstar", "MLGD*", "\\mlgd{}$^\\star$"), ("GBFESO-KL", "GB-FESO (KL)", "GB-FESO (KL)"),
           ("GBFESO-MMD", "GB-FESO (MMD)", "GB-FESO (\\mmd{})")]


def dims(cell):
    x, y = cell[1:].split("_y")
    return int(x), int(y)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs/gbfeso_d256")
    p.add_argument("--data-root", required=True, help="directory holding <cell>/data.pt")
    p.add_argument("--out", default="experiments/synthetic/results/table1_gbfeso_d256")
    a = p.parse_args()

    rows, runs_out = [], {}
    for cell in CELLS:
        if not glob.glob(os.path.join(a.runs_dir, cell, "*_s*.json")):
            print(f"  missing: {cell} (no result files)")
            continue
        data = _load_mixture(os.path.join(a.data_root, cell, "data.pt"))
        if data is None:
            raise SystemExit(f"cannot read {a.data_root}/{cell}/data.pt")
        for m, _, _ in METHODS:
            files = sorted(glob.glob(os.path.join(a.runs_dir, cell, f"{m}_s*.json")),
                           key=lambda f: int(re.search(r"_s(\d+)\.json$", f).group(1)))
            if not files:
                print(f"  missing: {cell} {m}")
                continue
            recs, traj_tv = [], []
            for f in files:
                r = json.load(open(f))
                att = r["attempts"][0]
                d = scale_free_distances(data, att["x_hat"])
                xs = r.get("x_star")
                recs.append({"seed": int(re.search(r"_s(\d+)\.json$", f).group(1)), "x_hat": att["x_hat"],
                             "tv": d["tv"], "l2_rel": d["l2_rel"], "time_s": att.get("time_s"),
                             "dist": None if xs is None else float(np.linalg.norm(np.array(att["x_hat"]) - np.array(xs))),
                             "sampler": r["args"].get("sampler")})
                traj_tv += [scale_free_distances(data, q["x_hat"], n_mc=3000, seed=i)["tv"]
                            for i, q in enumerate(att.get("restarts", [att]))]
            def ms(k):
                v = [q[k] for q in recs if q[k] is not None]
                return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))
            (tv_m, tv_s), (d_m, d_s), (l_m, l_s), (t_m, _) = ms("tv"), ms("dist"), ms("l2_rel"), ms("time_s")
            dx, dy = dims(cell)
            rows.append({"setting": cell, "dim_x": dx, "dim_y": dy, "method": m, "n": len(recs),
                         "tv_mean": tv_m, "tv_std": tv_s, "dist_mean": d_m, "dist_std": d_s,
                         "l2rel_mean": l_m, "l2rel_std": l_s, "time_mean": t_m,
                         "traj_tv_lt_0.5": f"{sum(t < 0.5 for t in traj_tv)}/{len(traj_tv)}"})
            runs_out.setdefault(cell, {})[m] = recs
            print(f"  {cell:8s} {m:11s} n={len(recs)}  TV {tv_m:.3f} +- {tv_s:.3f}  ({t_m / 60:.0f} min/run)", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump({"rows": rows, "runs": runs_out, "args": vars(a)}, f, indent=2)

    get = lambda c, m: next((r for r in rows if r["setting"] == c and r["method"] == m), None)   # noqa: E731
    md = ["# MLGD* and GB-FESO on the retrained conditional samplers", "",
          "Exact TV(P(Y | x̂), G), mean ± std over 5 runs (seeds 42–46); lower is better; **bold**: the best in the row.",
          "All three use the same retrained conditional sampler per cell (diffusion, 256 units, 4 blocks; `runs/night/grid/<cell>/d256`,",
          "at (10, 1) `runs/night/x10_y1/models/d256`) and the Table 1 data and target (`runs/table1v2/<cell>/data.pt`).",
          "MLGD*: zeroth-order guidance, 30 restarts, refinement 100 steps × 3,000 samples keeping the best iterate, the 5",
          "lowest-loss restarts re-scored 5 × 5,000 (Or's `G_zo` runs; `E6_sel5000_n3000` at (10, 1), the same recipe).",
          "GB-FESO (Zick & Mendels, arXiv:2607.06421): 30 starts from random training designs, their optimizer (201 + 201",
          "Adam steps), 250 samples per phase-0 step, 3,000 per phase-1 step, each start scored on 1,000, and MLGD*'s",
          "selection (top 5 re-scored 5 × 5,000); KL: their KDE-KL loss, MMD: the Table 1 loss.",
          "Written by `pipeline/make_table1_d256.py`; setup and commands: `../README.md`, section \"GB-FESO competitor\".", "",
          "| dim X × dim Y | " + " | ".join(lbl for _, lbl, _ in METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
    tex = []
    for c in CELLS:
        vals = [get(c, m) for m, _, _ in METHODS]
        if all(v is None for v in vals):
            continue
        best = min(v["tv_mean"] for v in vals if v is not None)
        cm, ct = [], []
        for v in vals:
            if v is None:
                cm.append("-"); ct.append("--"); continue
            s = f"{v['tv_mean']:.3f} ± {v['tv_std']:.3f}"
            cm.append(f"**{s}**" if v["tv_mean"] == best else s)
            t = f"{v['tv_mean']:.2f} \\pm {v['tv_std']:.2f}"
            ct.append(f"$\\mathbf{{{t}}}$" if v["tv_mean"] == best else f"${t}$")
        dx, dy = dims(c)
        md.append(f"| {dx}×{dy} | " + " | ".join(cm) + " |")
        tex.append(f"$({dx}, {dy})$ & " + " & ".join(ct) + " \\\\")
    md.append("| time / run (min), mean | " + " | ".join(
        f"{np.mean([r['time_mean'] for r in rows if r['method'] == m]) / 60:.0f}" if any(r["method"] == m for r in rows) else "-"
        for m, _, _ in METHODS) + " |")
    md += ["", "## All columns", "",
           "| dim X | dim Y | method | TV mean ± std | ‖x̂ − x*‖ (sd units) | relative L2 | time / run (min) | trajectories with TV < 0.5 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['dim_x']} | {r['dim_y']} | {dict((m, l) for m, l, _ in METHODS)[r['method']]} | "
                  f"{r['tv_mean']:.3f} ± {r['tv_std']:.3f} | {r['dist_mean']:.3f} ± {r['dist_std']:.3f} | "
                  f"{r['l2rel_mean']:.3f} ± {r['l2rel_std']:.3f} | {r['time_mean'] / 60:.0f} | {r['traj_tv_lt_0.5']} |")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    open(a.out + ".tex", "w").write("% (d_X, d_Y) & " + " & ".join(t for _, _, t in METHODS)
                                    + " -- TV mean +- std over 5 runs, written by make_table1_d256.py\n"
                                    + "\n".join(tex) + "\n")
    print(f"[table] wrote {a.out}.json / .md / .tex")


if __name__ == "__main__":
    main()
