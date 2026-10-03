"""
One table of every GB-FESO comparison arm next to MLGD: TV(P(Y | x_hat), G), mean +- std over the runs.

Reads the tables written by `make_table1.py --collect` (results/table1_*.json) and the Table 1 run
(experiments/synthetic/results/table1.json); a missing file or row prints as "-". Run from the repository root:

    python experiments/synthetic/pipeline/compare_gbfeso.py                 # TV
    python experiments/synthetic/pipeline/compare_gbfeso.py --metric dist   # ||x_hat - x*|| (sd units)

Writes results/compare_gbfeso_<metric>.md as well.
"""
import argparse
import json
import os

# column: (table, method); the method of a row must match exactly, after the tag is removed
COLUMNS = [
    ("MLGD MMD 250 (T1)",   "experiments/synthetic/results/table1.json", "MLGD"),
    ("MLGD-F MMD 250 (T1)", "experiments/synthetic/results/table1.json", "MLGD-F"),
    ("MLGD KL 250",         "results/table1_mlgd_kl.json", "MLGD"),
    ("GB-FESO KL 250",      "results/table1_gbfeso_main.json", "GBFESO-KL"),
    ("GB-FESO MMD 250",     "results/table1_gbfeso_main.json", "GBFESO-MMD"),
    ("MLGD MMD 1k",         "results/table1_mlgd_1k.json", "MLGD"),
    ("MLGD-F MMD 1k",       "results/table1_mlgd_1k.json", "MLGD-F"),
    ("GB-FESO KL 1k",       "results/table1_gbfeso_1k.json", "GBFESO-KL"),
    ("GB-FESO MMD 1k",      "results/table1_gbfeso_1k.json", "GBFESO-MMD"),
    ("MLGD MMD 3.5k",       "results/table1_mlgd_3500.json", "MLGD"),
    ("MLGD-F MMD 3.5k",     "results/table1_mlgd_3500.json", "MLGD-F"),
    ("MLGD KL 3.5k",        "results/table1_mlgd_kl3500.json", "MLGD"),
    ("MLGD-F KL 3.5k",      "results/table1_mlgd_kl3500.json", "MLGD-F"),
    ("GB-FESO KL 3.5k",     "results/table1_gbfeso_3500.json", "GBFESO-KL"),
    ("GB-FESO MMD 3.5k",    "results/table1_gbfeso_3500.json", "GBFESO-MMD"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", choices=["tv", "dist", "l2rel", "time"], default="tv")
    a = p.parse_args()
    cell, cols = {}, []
    for col, path, method in COLUMNS:
        if not os.path.exists(path):
            continue
        rows = json.load(open(path))["rows"]
        tag = json.load(open(path)).get("args", {}).get("tag", "")
        hit = False
        for r in rows:
            m = r["method"][:-len(tag)] if tag and r["method"].endswith(tag) else r["method"]
            if m != method:
                continue
            hit = True
            mean, std = r.get(f"{a.metric}_mean"), r.get(f"{a.metric}_std")
            if mean is None:
                continue
            cell[(r["setting"], col)] = f"{mean:.0f}" if a.metric == "time" else f"{mean:.3f} ± {std:.3f}"
        if hit:
            cols.append(col)
    settings = sorted({s for s, _ in cell}, key=lambda s: tuple(int(v) for v in s[1:].split("_y")))
    lines = ["| scenario | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for s in settings:
        name = s[1:].replace("_y", "×")
        lines.append(f"| {name} | " + " | ".join(cell.get((s, c), "-") for c in cols) + " |")
    out = "\n".join(lines) + "\n"
    print(out)
    os.makedirs("results", exist_ok=True)
    open(f"results/compare_gbfeso_{a.metric}.md", "w").write(out)
    print(f"[compare] wrote results/compare_gbfeso_{a.metric}.md")


if __name__ == "__main__":
    main()
