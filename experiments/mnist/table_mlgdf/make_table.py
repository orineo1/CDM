"""Build paper Table 2 from the run records written by run_mlgdf.py.

Reads every `.pkl` in a results directory (one per target) and prints one row
per target in the layout of the paper's Table 2:

    SWD_all    mean +- std of the final SWD over all seeds
    SWD_top5   mean +- std over the 5 seeds with the lowest final SWD
    Time (s)   mean +- std of the wall-clock time per seed
    0 ... 9    % of the *classified* seeds recovered as each digit
    None       number of seeds the classifier left unlabelled (confidence below
               the threshold); they are excluded from the percentages, as in the paper

std is the population standard deviation (numpy default, ddof=0).
The table is also written to <results-dir>/table2.md.

    python experiments/mnist/table_mlgdf/make_table.py experiments/mnist/table_mlgdf/outputs/full
"""

import argparse
import glob
import os
import pickle

import numpy as np

ORDER = ["uniform", "bimodal", "unimodal"]


def pm(x, digits=3):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.{digits}f} ± {x.std():.{digits}f}"


def row(rec):
    loss = np.asarray(rec["loss_log"], dtype=float)
    time_s = np.asarray(rec["time_log"], dtype=float)
    top5 = np.sort(loss)[:5]
    cells = [rec["target"], pm(loss), pm(top5), pm(time_s, 1)]
    preds = rec.get("predicted_labels")
    if preds is None:
        cells += ["n/a"] * 10 + ["n/a"]
        return cells
    labelled = [p for p in preds if p is not None]
    for d in range(10):
        n = sum(p == d for p in labelled)
        cells.append(f"{100 * n / len(labelled):.0f}" if n and labelled else "--")
    cells.append(str(len(preds) - len(labelled)))
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results_dir", help="directory with the .pkl run records (e.g. outputs/full)")
    args = p.parse_args()

    recs = {}
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.pkl"))):
        with open(path, "rb") as fh:
            rec = pickle.load(fh)
        if "loss_log" in rec and "target" in rec:
            recs[rec["target"]] = rec
    if not recs:
        raise SystemExit(f"no run records (.pkl) in {args.results_dir}")

    header = ["Target", "SWD_all", "SWD_top5", "Time (s)"] + [str(d) for d in range(10)] + ["None"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name in ORDER + sorted(set(recs) - set(ORDER)):
        if name in recs:
            lines.append("| " + " | ".join(row(recs[name])) + " |")
    n_seeds = {name: len(r["loss_log"]) for name, r in recs.items()}
    lines.append("")
    lines.append(f"Seeds per target: {n_seeds}. Digit columns: % of classified seeds; "
                 "None: seeds below the classifier threshold (excluded from the %).")
    table = "\n".join(lines)

    print(table)
    out = os.path.join(args.results_dir, "table2.md")
    with open(out, "w") as fh:
        fh.write(table + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
