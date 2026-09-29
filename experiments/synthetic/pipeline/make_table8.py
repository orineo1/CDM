"""The appendix table tab:sim_full (ablations) from the make_table1.py JSONs of its four arms.

Per scenario and method: ||x_hat - x*|| (sd units) and the relative L2 after refinement
(from the refined arm), then the TV without refinement of the plain search, the annealed
resampling and the oracle sampler. The TV of the refined arm (Table 1) is kept as a column
for reading, the paper prints it in Table 1 only. The four arms:

    no refinement        <ablations>/table8_norefine.json   (MLGD, MLGD-F, PMLGD, PMLGD-F)
    refined              <main_grid>/table1.json            (= Table 1)
    annealed resampling  <ablations>/table8_annealed.json   (PMLGD, PMLGD-F)
    oracle f_phi         <ablations>/table8_oracle.json     (MLGD-X -> MLGD row, PMLGD-X -> PMLGD row)

Written to <out>.md and <out>.csv; a missing arm or cell prints as "--".

    python experiments/synthetic/pipeline/make_table8.py experiments/synthetic/ablations/outputs \
        --refined experiments/synthetic/main_grid/outputs/table1.json
"""

import argparse
import csv
import json
import os

SCENARIOS = ["x1_y1", "x1_y5", "x1_y10", "x5_y1", "x5_y5", "x5_y10", "x10_y1", "x10_y5", "x10_y10"]
METHODS = ["MLGD", "MLGD-F", "PMLGD", "PMLGD-F"]
ARMS = ["norefine", "refined", "annealed", "oracle"]
HEADER = ["||x_hat-x*|| (sd)", "relative L2", "TV refined (Table 1)", "TV no refinement", "TV annealed", "TV oracle f_phi"]
ORACLE = {"MLGD-X": "MLGD", "PMLGD-X": "PMLGD"}


def load_rows(path):
    """{(scenario, method): row dict} from a make_table1.py JSON, or {} if absent."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    for r in json.load(open(path))["rows"]:
        out[(r["setting"], ORACLE.get(r["method"], r["method"]))] = r
    return out


def pm(r, key):
    return "--" if r is None or r.get(f"{key}_mean") is None else f"{r[f'{key}_mean']:.2f} ± {r[f'{key}_std']:.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ablations_dir", help="directory with table8_{norefine,annealed,oracle}.json")
    p.add_argument("--refined", default=None, help="main_grid's table1.json (the refined arm)")
    p.add_argument("--out", default=None, help="output stem (default <ablations_dir>/table8)")
    args = p.parse_args()
    out = args.out or os.path.join(args.ablations_dir, "table8")

    arms = {a: load_rows(os.path.join(args.ablations_dir, f"table8_{a}.json")) for a in ("norefine", "annealed", "oracle")}
    arms["refined"] = load_rows(args.refined)
    present = [a for a in ARMS if arms[a]]
    if not present:
        raise SystemExit(f"no table8_*.json under {args.ablations_dir}")
    print("arms found:", ", ".join(present))

    keys = {k for a in arms.values() for k in a}
    names = [s for s in SCENARIOS if any(k[0] == s for k in keys)]
    rows = []
    for s in names:
        for m in METHODS:
            if not any((s, m) in arms[a] for a in ARMS):
                continue
            r = {"scenario": s, "method": m}
            ref = arms["refined"].get((s, m))
            r["dist_mean"], r["dist_std"] = (ref["dist_mean"], ref["dist_std"]) if ref else (None, None)
            r["l2rel_mean"], r["l2rel_std"] = (ref["l2rel_mean"], ref["l2rel_std"]) if ref else (None, None)
            for a in ARMS:
                v = arms[a].get((s, m))
                r[f"{a}_mean"], r[f"{a}_std"] = (v["tv_mean"], v["tv_std"]) if v else (None, None)
            rows.append(r)

    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["| Setting | Method | " + " | ".join(HEADER) + " |", "|---|---|" + "---|" * len(HEADER)]
    for r in rows:
        dx, dy = (int(v[1:]) for v in r["scenario"].split("_"))
        cells = [pm(r, "dist"), pm(r, "l2rel")] + [pm(r, a) for a in ["refined", "norefine", "annealed", "oracle"]]
        md.append(f"| ({dx}, {dy}) | {r['method']} | " + " | ".join(cells) + " |")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}.md and {out}.csv")


if __name__ == "__main__":
    main()
