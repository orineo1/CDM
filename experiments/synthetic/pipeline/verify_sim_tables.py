"""Compare the paper's tab:sim_full rows and Table 1 (MLGD, MLGD-F columns) with the run files.

Run from the CDM repository root (the one holding runs/table1v2), after
    make_table1.py --collect ... --tag <tag> --out $OUT/<tag>
for the tags _adam_refine, _adam, _adam_oracle (see the shell lines that come with this script).
"""
import json, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tmp/verify")
CELLS = [(1, 1), (1, 5), (1, 10), (5, 1), (5, 5), (5, 10), (10, 1), (10, 5), (10, 10)]

# the paper's tab:sim_full rows: (dist, l2rel, plain TV, oracle TV) as (mean, std); None = "--"
PAPER = {
    (1, 1):   {"MLGD": [(0.004, .003), (0.003, .003), (0.01, .01), (0.02, .01)], "MLGD-F": [(0.019, .003), (0.039, .011), (0.08, .05), None]},
    (1, 5):   {"MLGD": [(0.018, .011), (0.002, .002), (0.02, .01), (0.02, .01)], "MLGD-F": [(0.078, .022), (0.040, .020), (0.08, .05), None]},
    (1, 10):  {"MLGD": [(0.015, .014), (0.054, .087), (0.12, .08), (0.07, .04)], "MLGD-F": [(0.081, .011), (0.684, .138), (0.42, .08), None]},
    (5, 1):   {"MLGD": [(0.938, .249), (0.028, .007), (0.10, .06), (0.17, .17)], "MLGD-F": [(1.336, .636), (0.297, .222), (0.22, .11), None]},
    (5, 5):   {"MLGD": [(0.038, .023), (0.014, .007), (0.28, .09), (0.24, .15)], "MLGD-F": [(0.479, .016), (0.465, .153), (0.49, .03), None]},
    (5, 10):  {"MLGD": [(0.080, .012), (0.043, .007), (0.22, .06), (0.18, .08)], "MLGD-F": [(0.252, .017), (0.746, .289), (0.42, .09), None]},
    (10, 1):  {"MLGD": [(2.234, .947), (0.952, .331), (0.59, .24), (0.89, .10)], "MLGD-F": [(1.802, .445), (1.237, .187), (0.61, .08), None]},
    (10, 5):  {"MLGD": [(0.587, .275), (0.233, .122), (0.31, .07), (0.54, .15)], "MLGD-F": [(1.102, .255), (0.876, .049), (0.43, .10), None]},
    (10, 10): {"MLGD": [(0.227, .041), (0.168, .100), (0.66, .03), (0.54, .20)], "MLGD-F": [(1.321, .008), (2.920, .079), (0.83, .11), None]},
}
# Table 1 (tab:sim_main), TV columns MLGD and MLGD-F, as printed (2 decimals)
TABLE1 = {
    (1, 1): [(.02, .01), (.08, .01)], (1, 5): [(.02, .01), (.08, .02)], (1, 10): [(.07, .07), (.38, .05)],
    (5, 1): [(.08, .01), (.25, .09)], (5, 5): [(.06, .02), (.50, .05)], (5, 10): [(.14, .03), (.41, .04)],
    (10, 1): [(.49, .08), (.56, .03)], (10, 5): [(.13, .04), (.30, .01)], (10, 10): [(.20, .05), (.92, .02)],
}


def load(tag):
    p = os.path.join(OUT, tag + ".json")
    if not os.path.exists(p):
        print(f"!! missing {p}: run make_table1.py --collect for {tag} first")
        return {}
    return {(r["dim_x"], r["dim_y"], r["method"]): r for r in json.load(open(p))["rows"]}


def same(paper, mean, std, nd):
    """paper value (mean, std) printed with nd decimals vs the run's unrounded mean, std."""
    pm, ps = paper
    return round(mean, nd) == round(pm, nd) and round(std, nd) == round(ps, nd)


refine, plain, oracle = load("_adam_refine"), load("_adam"), load("_adam_oracle")
bad = n_checked = 0
print(f"{'cell':9s} {'method':7s} {'column':12s} {'paper':>16s} {'run files':>18s}")
for c in CELLS:
    for m in ("MLGD", "MLGD-F"):
        checks = []
        r = refine.get((*c, m))
        p = PAPER[c][m]
        if r:
            checks += [("dist x*", p[0], r["dist_mean"], r["dist_std"], 3),
                       ("rel L2", p[1], r["l2rel_mean"], r["l2rel_std"], 3)]
            checks += [("Table1 TV", TABLE1[c][0 if m == "MLGD" else 1], r["tv_mean"], r["tv_std"], 2)]
        q = plain.get((*c, m))
        if q:
            checks.append(("plain TV", p[2], q["tv_mean"], q["tv_std"], 2))
        if m == "MLGD":
            o = oracle.get((*c, "MLGD-X"))
            if o:
                checks.append(("oracle TV", p[3], o["tv_mean"], o["tv_std"], 2))
        for name, pv, mean, std, nd in checks:
            ok = same(pv, mean, std, nd)
            bad += not ok
            n_checked += 1
            print(f"{str(c):9s} {m:7s} {name:12s} {pv[0]:>7.{nd}f} ± {pv[1]:.{nd}f} {mean:>9.{nd}f} ± {std:.{nd}f}  "
                  f"{'ok' if ok else '<-- DIFFERENT'}")
print("\ntimes per run (caption says 496-526 s diffusion = MLGD, 46-55 s consistency = MLGD-F):")
for m in ("MLGD", "MLGD-F"):
    ts = [refine[(*c, m)]["time_mean"] for c in CELLS if (*c, m) in refine]
    if ts:
        print(f"  {m:7s} {min(ts):.0f}-{max(ts):.0f} s")
if n_checked == 0:
    print("\nNOTHING CHECKED: no run values were loaded (see the missing files above)")
    sys.exit(1)
print(f"\n{bad} of {n_checked} value(s) differ from the run files" if bad else f"\nall {n_checked} values match the run files")
expected = 9 * 2 * 4 + 9   # per cell and method: dist, L2, Table 1 TV, plain TV; plus 9 oracle values
if n_checked < expected:
    print(f"warning: only {n_checked} of {expected} values could be checked (some runs are missing)")
