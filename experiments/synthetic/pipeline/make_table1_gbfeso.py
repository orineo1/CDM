"""
results/table1_gbfeso.md: Table 1 with the GB-FESO competitor, from two make_table1.py --collect tables:
the Table 1 methods (results/table1.json) and GB-FESO at the Table 1 budget (--tag _main --methods
GBFESO-KL GBFESO-MMD). Run from the repository root:

    python experiments/synthetic/pipeline/make_table1_gbfeso.py \\
        --table1 experiments/synthetic/results/table1.json --gbfeso experiments/synthetic/results/table1_gbfeso.json \\
        --out experiments/synthetic/results/table1_gbfeso.md
"""
import argparse
import json

OURS = ["MLGD", "MLGD-F", "PMLGD", "PMLGD-F"]
THEIRS = [("GBFESO-KL", "GB-FESO (KL)"), ("GBFESO-MMD", "GB-FESO (MMD)")]

HEADER = """# Table 1 with the GB-FESO competitor

TV(P(Y | x̂), G), mean ± std over 5 runs (10 trajectories each); lower is better; **bold**: the best in the row.
The Table 1 methods are `table1.json` (this folder); GB-FESO (Zick & Mendels, arXiv:2607.06421) runs on the
same networks and data at the same sample budget: 250 conditional samples per step, 1,000 per refinement step
(GB-FESO: phase 1), scored on 1,000. GB-FESO (MMD) uses the Table 1 loss, so MLGD vs GB-FESO (MMD) differs only
in the search; GB-FESO (KL) is their method with their KDE-KL loss. Setup and command: `../README.md`,
section "GB-FESO competitor". Written by `pipeline/make_table1_gbfeso.py`.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table1", default="experiments/synthetic/results/table1.json")
    p.add_argument("--gbfeso", default="experiments/synthetic/results/table1_gbfeso.json")
    p.add_argument("--out", default="experiments/synthetic/results/table1_gbfeso.md")
    a = p.parse_args()
    rows = {(r["setting"], r["method"]): r for f in (a.table1, a.gbfeso) for r in json.load(open(f))["rows"]}
    settings = sorted({s for s, _ in rows}, key=lambda s: tuple(int(v) for v in s[1:].split("_y")))
    methods = [(m, m) for m in OURS] + THEIRS

    def name(s):
        x, y = s[1:].split("_y")
        return f"{x}×{y}"

    L = [HEADER, "| dim X × dim Y | " + " | ".join(lbl for _, lbl in methods) + " |", "|---" * (len(methods) + 1) + "|"]
    for s in settings:
        vals = [rows.get((s, m)) for m, _ in methods]
        best = min(r["tv_mean"] for r in vals if r is not None)
        cells = []
        for r in vals:
            if r is None:
                cells.append("-")
                continue
            c = f"{r['tv_mean']:.3f} ± {r['tv_std']:.3f}"
            cells.append(f"**{c}**" if r["tv_mean"] == best else c)
        L.append(f"| {name(s)} | " + " | ".join(cells) + " |")
    times = []
    for m, _ in methods:
        t = [rows[(s, m)]["time_mean"] for s in settings if (s, m) in rows]
        times.append(f"{sum(t) / len(t):.0f}" if t else "-")
    L.append("| time / run (s), mean | " + " | ".join(times) + " |")

    L += ["", "## GB-FESO, all columns", "",
          "| dim X | dim Y | method | TV mean ± std | ‖x̂ − x*‖ (sd units) | relative L2 | time / run (s) | trajectories with TV<0.5 | in x* basin |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in settings:
        for m, _ in THEIRS:
            r = rows.get((s, m))
            if r is None:
                continue
            L.append(f"| {r['dim_x']} | {r['dim_y']} | {m} | {r['tv_mean']:.3f} ± {r['tv_std']:.3f} | "
                     f"{r['dist_mean']:.3f} ± {r['dist_std']:.3f} | {r['l2rel_mean']:.3f} ± {r['l2rel_std']:.3f} | "
                     f"{r['time_mean']:.0f} | {r['success'] or ''} | {r['hits'] or ''} |")
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"[table1_gbfeso] wrote {a.out}")


if __name__ == "__main__":
    main()
