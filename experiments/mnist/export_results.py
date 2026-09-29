#!/usr/bin/env python
"""Copy a finished MNIST run into the committed results/ folders.

Reads the run records of `table_mlgdf/run.sh full` (outputs/full/*.pkl) and the
figures of `figures/run.sh full` (figures/outputs/full/*.png) and writes files
that are safe and small to commit, so the results can be read without re-running:

    table_mlgdf/results/table2.md       Table 2
    table_mlgdf/results/seeds.csv       one row per target and seed: SWD, time, label, confidence
    table_mlgdf/results/images.npz      the recovered images (numpy, no pickle; see below)
    table_mlgdf/results/run_info.txt    commit, classifier, GPU and settings of the run
    figures/results/*.png               the figures

images.npz holds, per target T: T_images (15, 28, 28) float32 as optimised (not
clamped), T_seed, T_swd, T_time_s, T_label (-1 = below the classifier threshold)
and T_confidence. Load it with np.load("images.npz") (no allow_pickle needed).

    python experiments/mnist/export_results.py
"""

import argparse
import csv
import glob
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["uniform", "bimodal", "unimodal"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", default=os.path.join(HERE, "table_mlgdf", "outputs", "full"))
    p.add_argument("--figures-dir", default=os.path.join(HERE, "figures", "outputs", "full"))
    p.add_argument("--no-figures", action="store_true", help="export the table and data only")
    args = p.parse_args()

    records = {}
    for path in sorted(glob.glob(os.path.join(args.run_dir, "*.pkl"))):
        with open(path, "rb") as fh:
            rec = pickle.load(fh)
        records[rec["target"]] = rec
    if not records:
        raise SystemExit(f"no run records (.pkl) in {args.run_dir}; run table_mlgdf/run.sh full first")
    targets = [t for t in ORDER if t in records] + sorted(set(records) - set(ORDER))

    table_res = os.path.join(HERE, "table_mlgdf", "results")
    os.makedirs(table_res, exist_ok=True)

    arrays, rows = {}, []
    for t in targets:
        rec = records[t]
        n = len(rec["loss_log"])
        labels = rec.get("predicted_labels") or [None] * n
        confs = rec.get("predicted_confidences") or [float("nan")] * n
        arrays[f"{t}_images"] = np.stack([np.asarray(x, dtype=np.float32).reshape(28, 28)
                                          for x in rec["results"]])
        arrays[f"{t}_seed"] = np.asarray(rec["seed_log"], dtype=np.int64)
        arrays[f"{t}_swd"] = np.asarray(rec["loss_log"], dtype=np.float64)
        arrays[f"{t}_time_s"] = np.asarray(rec["time_log"], dtype=np.float64)
        arrays[f"{t}_label"] = np.asarray([-1 if x is None else int(x) for x in labels], dtype=np.int64)
        arrays[f"{t}_confidence"] = np.asarray(confs, dtype=np.float64)
        top = set(np.argsort(rec["loss_log"])[:5].tolist())
        for k in range(n):
            rows.append({"target": t, "run": rec["experiment_name"], "seed": int(rec["seed_log"][k]),
                         "swd": round(float(rec["loss_log"][k]), 6),
                         "time_s": round(float(rec["time_log"][k]), 2),
                         "in_top5": k in top,
                         "label": "None" if labels[k] is None else int(labels[k]),
                         "confidence": round(float(confs[k]), 4)})

    np.savez_compressed(os.path.join(table_res, "images.npz"), **arrays)
    with open(os.path.join(table_res, "seeds.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    subprocess.run([sys.executable, os.path.join(HERE, "table_mlgdf", "make_table.py"), args.run_dir],
                   check=True, stdout=subprocess.DEVNULL)
    shutil.copy(os.path.join(args.run_dir, "table2.md"), os.path.join(table_res, "table2.md"))
    info = os.path.join(args.run_dir, "run_info.txt")
    if os.path.exists(info):
        shutil.copy(info, os.path.join(table_res, "run_info.txt"))
    else:
        print(f"note: no {info} (run made before run.sh recorded it); run_info.txt not exported")

    n_png = 0
    if not args.no_figures:
        pngs = sorted(glob.glob(os.path.join(args.figures_dir, "*.png")))
        if not pngs:
            print(f"note: no figures in {args.figures_dir}; run figures/run.sh full, or pass --no-figures")
        fig_res = os.path.join(HERE, "figures", "results")
        os.makedirs(fig_res, exist_ok=True)
        for old in glob.glob(os.path.join(fig_res, "*.png")):
            os.remove(old)
        for png in pngs:
            shutil.copy(png, fig_res)
        n_png = len(pngs)

    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(table_res, "*")))
    print(f"exported {len(targets)} targets ({', '.join(targets)}): table2.md, seeds.csv ({len(rows)} rows), "
          f"images.npz, run_info.txt -> {table_res} ({size / 1024:.0f} KB); {n_png} figures -> figures/results/")
    print("commit with:  git add experiments/mnist/table_mlgdf/results experiments/mnist/figures/results")


if __name__ == "__main__":
    main()
