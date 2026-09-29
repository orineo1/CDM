#!/usr/bin/env python
"""Rank the finished grid-search configurations of one target.

Reads outputs/<target>/*/metrics.json, sorts by the classifier on the top-5 seeds
(top5_clf_score, highest first; ties broken by lower top-5 SWD), prints the table
and writes outputs/<target>/ranking.md. Each row names the configuration's folder,
which holds <name>_top5.png and <name>_all_seeds.png.

    python experiments/mnist/grid_search/summarize.py --target uniform
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target", choices=["uniform", "bimodal"], required=True)
    p.add_argument("--out", default=os.path.join(HERE, "outputs"))
    p.add_argument("--top", type=int, default=0, help="show only the best N configurations (0: all)")
    args = p.parse_args()

    root = os.path.join(args.out, args.target)
    rows = []
    for path in glob.glob(os.path.join(root, "*", "metrics.json")):
        with open(path) as fh:
            rows.append(json.load(fh))
    if not rows:
        raise SystemExit(f"no finished configurations under {root}")
    rows.sort(key=lambda m: (-m["top5_clf_score"], m["swd_top5_mean"]))
    if args.top:
        rows = rows[:args.top]

    head = ["rank", "configuration", "top5 score", "top5 labelled", "top5 conf", "top5 labels",
            "SWD top5", "SWD all", "labelled /seeds", "time/seed (s)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, m in enumerate(rows, 1):
        lines.append("| " + " | ".join([
            str(i), m["run_name"], f"{m['top5_clf_score']:.3f}", f"{m['top5_n_labelled']}/5",
            f"{m['top5_conf_mean']:.3f}", " ".join(str(x) for x in m["top5_labels"]),
            f"{m['swd_top5_mean']:.4f} ± {m['swd_top5_std']:.4f}",
            f"{m['swd_all_mean']:.4f} ± {m['swd_all_std']:.4f}",
            str(m["n_labelled"]), f"{m['time_mean_s']:.1f}"]) + " |")
    lines += ["", f"top5 score = (# of the 5 lowest-SWD seeds the classifier labels) + (their mean "
              f"confidence); threshold {rows[0].get('threshold')}. Figures: {root}/<configuration>/"]
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(root, "ranking.md"), "w") as fh:
        fh.write(table + "\n")
    print(f"\nwrote {os.path.join(root, 'ranking.md')}")


if __name__ == "__main__":
    main()
