"""Parity table on GB-FESO's benchmark: dF error (mean, median, share <= 0.3 k_BT) over the multi-basin targets of
a held-out split (targets 0-9, used to choose the loss per N), the test split (10-39) and all 40, from evaluate.py
per_run JSON files (several may be given; rows are keyed by method name, later files override earlier ones).

    python experiments/peptide/pipeline/parity_table.py --n 5 --per-run a.json b.json ... [--held-out 0-9]
GB-FESO rows: "GB-FESO, 1 start" pools its starts (each start one run of their method); "GB-FESO, best of 4"
takes per target the start with the lowest surrogate KL (their own choice rule)."""
import argparse
import collections
import json

import numpy as np


def stats(errs):
    e = np.asarray(errs, dtype=float)
    return (f"{e.mean():.3f}", f"{np.median(e):.3f}", f"{100 * np.mean(e <= 0.3):.0f}%", str(len(e))) if len(e) else ("-",) * 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--per-run", nargs="+", required=True)
    p.add_argument("--held-out", default="0-9")
    p.add_argument("--starts", nargs="*", default=None,
                   help="GB-FESO starts to use (e.g. low-low low-high high-low high-high; default: all of the file)")
    p.add_argument("--methods", nargs="*", default=None, help="method names to show, in order (default: all)")
    a = p.parse_args()
    lo, hi = map(int, a.held_out.split("-"))
    held = set(range(lo, hi + 1))
    M = collections.defaultdict(dict)
    for f in a.per_run:
        for r in json.load(open(f)):
            M[r["method"]][r["target_id"]] = r
    starts = sorted(m for m in M if m.startswith("GB-FESO [") and (a.starts is None or m[9:-1] in a.starts))
    rows = collections.OrderedDict()
    if starts:
        rows["GB-FESO, 1 start (pooled over starts)"] = [M[s][t] for s in starts for t in M[s]]
        best = []
        for t in M[starts[0]]:
            c = [M[s][t] for s in starts if t in M[s]]
            best.append({**min(c, key=lambda r: r["surrogate_loss"]), "time_s": sum(r["time_s"] for r in c)})
        rows[f"GB-FESO, best of {len(starts)} starts"] = best
    for m in (a.methods or [m for m in M if not m.startswith("GB-FESO")]):
        if m in M:
            rows[m] = list(M[m].values())
    print(f"| N={a.n} method | held-out mean | median | <=0.3 | n | test mean | median | <=0.3 | n | all-40 mean | median | <=0.3 | n | median time (s) |")
    print("|---|" + "---|" * 13)
    for name, rs in rows.items():
        mb = [r for r in rs if r["multi_basin"]]
        h = stats([r["df_median_abs_error"] for r in mb if r["target_id"] in held])
        te = stats([r["df_median_abs_error"] for r in mb if r["target_id"] not in held])
        al = stats([r["df_median_abs_error"] for r in mb])
        tm = np.median([r.get("time_s", np.nan) for r in rs])
        print(f"| {name} | " + " | ".join(h + te + al) + f" | {tm:.0f} |")


if __name__ == "__main__":
    main()
