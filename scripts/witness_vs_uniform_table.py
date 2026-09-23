"""
Witness vs. uniform gradient subsampling, in the Table 1 format: for every (dim X, dim Y) scenario
and every (n_cond, k/n) grid point, run the x*-finding algorithms with the gradient kept on only
k = k/n * n_cond of the n_cond conditional samples, chosen either uniformly or by the MMD witness
function, and collect the total variation between P(Y | X = x_hat) and G for both rules.

    python scripts/witness_vs_uniform_table.py                                  # nine scenarios, defaults below
    python scripts/witness_vs_uniform_table.py --settings 5x5 10x10 --k-fracs 0.1 0.5 --attempts 3
    python scripts/witness_vs_uniform_table.py --settings 1x1 --dry-run         # print the commands only
    python scripts/witness_vs_uniform_table.py --collect                        # only rebuild the table

Scenarios, data, networks and the model check are exactly Table 1's (scripts/make_table1.py, whose
steps this script calls): with the default --runs-dir the networks trained for Table 1 are reused,
and missing ones are trained the same way.

Per grid point (scripts/find_xstar.py --select <rule> --n-sel k --rescale <rescale>):
  uniform   k of the n_cond samples uniformly at random keep their gradient, the rest are detached
  witness   k samples drawn proportionally to the MMD witness |w(y_i)|, with the loss's kernel and
            the loss's target batch, mixed with --witness-floor of the uniform distribution (0: pure witness)
  --rescale ht (default): each kept sample's gradient is divided by its inclusion probability
            (Horvitz-Thompson), so under both rules the subsampled gradient is unbiased for the full one
The loss value always uses all n_cond samples. See utils/witness.py.

Per scenario, under <runs-dir>/<scenario>/: <METHOD>_<rule>_n<n_cond>_k<k/n>.json (find_xstar output).

Output: <out>.json (everything), <out>.md (one row per scenario, method, n_cond, k/n and rule: TV,
||x_hat - x*||, relative and raw L2, time; plus witness - uniform TV at matching points) and <out>.tex
(one row per scenario, method, n_cond and k/n: uniform TV, witness TV and their difference).
"""
import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_table1 as T                      # noqa: E402

RULES = ["uniform", "witness"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--settings", nargs="+", default=T.DEFAULT_SETTINGS, help="'<dimX>x<dimY>' or 'file:<params.txt>'")
    p.add_argument("--methods", nargs="+", default=T.METHODS, choices=T.ALL_METHODS)
    p.add_argument("--tag", default="", help="suffix for the method JSONs, e.g. _refine")
    p.add_argument("--runs-dir", default="runs/table1v2", help="Table 1's runs: data and networks are shared")
    p.add_argument("--out", default="results/witness_vs_uniform")
    g = p.add_argument_group("gradient subsampling")
    g.add_argument("--n-cond-list", type=int, nargs="+", default=[250], help="conditional samples per step (Table 1: 250)")
    g.add_argument("--k-fracs", type=float, nargs="+", default=[0.5], help="k / n_cond: fraction of samples that keep their gradient")
    g.add_argument("--rules", nargs="+", choices=RULES, default=RULES)
    g.add_argument("--witness-floor", type=float, default=0.0, help="witness: uniform mixture weight in the selection")
    g.add_argument("--rescale", choices=["ht", "none"], default="ht", help="ht: Horvitz-Thompson reweighting of the kept gradients")
    g = p.add_argument_group("data and training (as scripts/make_table1.py)")
    g.add_argument("--n-components", type=int, default=4)
    g.add_argument("--n-samples", type=int, default=20000)
    g.add_argument("--epochs", type=int, default=10000, help="training batches (fresh draws) or epochs (fixed set) per network")
    g.add_argument("--fixed-samples", action="store_true", help="train on the stored sample set instead of fresh draws")
    g.add_argument("--no-standardize", action="store_true", help="keep the raw mixture scale (default: whiten X and Y blocks)")
    g.add_argument("--batch-size", type=int, default=1024)
    g.add_argument("--nunits", type=int, default=128)
    g.add_argument("--nblocks", type=int, default=3)
    g = p.add_argument_group("x* search (as scripts/make_table1.py)")
    g.add_argument("--budget", type=int, default=10, help="trajectories per attempt: restarts (MLGD) or particles (PMLGD)")
    g.add_argument("--attempts", type=int, default=5, help="independent runs of each algorithm; mean/std over them")
    g.add_argument("--resample-beta", type=float, default=5.0)
    g.add_argument("--resample-schedule", choices=["constant", "linear"], default="constant")
    g.add_argument("--refine-steps", type=int, default=0)
    g.add_argument("--refine-lr", type=float, default=0.02)
    g.add_argument("--refine-n-cond", type=int, default=1000)
    g.add_argument("--guidance-optim", choices=["sgd", "adam"], default="sgd")
    g.add_argument("--step-size", default=None, help="find_xstar --step-size (default: preset)")
    g.add_argument("--extra", default="", help="extra flags passed verbatim to find_xstar.py")
    g = p.add_argument_group("control")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default=None)
    g.add_argument("--force", action="store_true", help="redo every step, training included")
    g.add_argument("--force-search", action="store_true", help="redo only the search runs; keep data, networks and the check")
    g.add_argument("--skip-check", action="store_true")
    g.add_argument("--collect", action="store_true", help="only rebuild the table from existing JSONs")
    g.add_argument("--no-collect", action="store_true", help="run the scenarios but do not write the table (array jobs)")
    g.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def grid(args):
    """(n_cond, k/n, rule, k) for every grid point."""
    return [(n, kf, rule, max(1, round(kf * n))) for n in args.n_cond_list for kf in args.k_fracs for rule in args.rules]


def point_args(args, n, kf, rule, k):
    """make_table1's args for one grid point: its n_cond, the selection flags for find_xstar and a
    JSON suffix, so make_table1.search/collect run and read this point's files."""
    a = copy.copy(args)
    a.n_cond = n
    a.tag = f"_{rule}_n{n}_k{kf:g}{args.tag}"
    sel = f"--select {rule} --n-sel {k} --rescale {args.rescale}"
    if rule == "witness":
        sel += f" --witness-floor {args.witness_floor}"
    a.extra = (sel + " " + args.extra).strip()
    return a


def collect(settings, args):
    rows, everything = [], {st["name"]: {"setting": st} for st in settings}
    for n, kf, rule, k in grid(args):
        r, e = T.collect(settings, point_args(args, n, kf, rule, k))
        for q in r:
            q.update({"n_cond": n, "k_frac": kf, "k": k, "rule": rule})
        rows += r
        for name, entry in e.items():
            everything[name].setdefault("check", entry.get("check"))
            for m in args.methods:
                if m in entry:
                    everything[name].setdefault(m, {})[f"{rule}_n{n}_k{kf:g}"] = entry[m]
    return rows, everything


def paired(rows):
    """(scenario, method, n_cond, k/n) -> {rule: row}, in row order."""
    out = {}
    for r in rows:
        out.setdefault((r["setting"], r["method"], r["n_cond"], r["k_frac"]), {})[r["rule"]] = r
    return out


def write_tables(rows, everything, args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pairs = paired(rows)
    with open(args.out + ".json", "w") as f:
        json.dump({"rows": rows, "scenarios": everything, "args": vars(args)}, f, indent=2)
    # markdown: every run, then witness - uniform at matching points
    md = [f"Gradient on k of n_cond samples, rescale={args.rescale}, witness floor={args.witness_floor}", "",
          "| dim X | dim Y | method | n_cond | k/n | rule | TV(P(Y|x_hat), G) mean +- std | ||x_hat-x*|| (sd units) | relative L2 | raw L2 GMM | time / attempt (s) | trajectories with TV<0.5 | in x* basin |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['dim_x']} | {r['dim_y']} | {r['method']} | {r['n_cond']} | {r['k_frac']:g} | {r['rule']} | "
                  f"{r['tv_mean']:.3f} +- {r['tv_std']:.3f} (n={r['n']}) | {r['dist_mean']:.3f} +- {r['dist_std']:.3f} | "
                  f"{r['l2rel_mean']:.3f} +- {r['l2rel_std']:.3f} | {r['l2_mean']:.4f} +- {r['l2_std']:.4f} | "
                  f"{r['time_mean']:.0f} | {r['success'] or ''} | {r['hits'] or ''} |")
    md += ["", "| dim X | dim Y | method | n_cond | k/n | TV uniform | TV witness | witness - uniform |", "|---|---|---|---|---|---|---|---|"]
    for (name, m, n, kf), by in pairs.items():
        if "uniform" in by and "witness" in by:
            u, w = by["uniform"], by["witness"]
            md.append(f"| {u['dim_x']} | {u['dim_y']} | {m} | {n} | {kf:g} | {u['tv_mean']:.3f} +- {u['tv_std']:.3f} | "
                      f"{w['tv_mean']:.3f} +- {w['tv_std']:.3f} | {w['tv_mean'] - u['tv_mean']:+.3f} |")
    open(args.out + ".md", "w").write("\n".join(md) + "\n")
    # latex rows (booktabs body): per scenario, one line per method, n_cond and k/n
    tex, last = [], None
    names = {"MLGD": "\\mlgd{}", "MLGD-F": "\\mlgdf{}", "PMLGD": "P\\mlgd{}", "PMLGD-F": "P\\mlgdf{}",
             "MLGD-X": "\\mlgd{} (oracle $\\fphi$)", "PMLGD-X": "P\\mlgd{} (oracle $\\fphi$)"}

    def cell(by, rule):
        r = by.get(rule)
        return "--" if r is None else f"${r['tv_mean']:.3f} \\pm {r['tv_std']:.3f}$"
    for (name, m, n, kf), by in pairs.items():
        r0 = next(iter(by.values()))
        if name != last:
            if last is not None:
                tex.append("\\midrule")
            count = sum(1 for key in pairs if key[0] == name)
            label = f"$\\dim X={r0['dim_x']}$, $\\dim Y={r0['dim_y']}$" if r0["dim_x"] is not None else name
            tex.append(f"\\multirow{{{count}}}{{*}}{{{label}}}")
            last = name
        diff = f"${by['witness']['tv_mean'] - by['uniform']['tv_mean']:+.3f}$" if "uniform" in by and "witness" in by else "--"
        tex.append(f" & {names[m]} & ${n}$ & ${kf:g}$ & {cell(by, 'uniform')} & {cell(by, 'witness')} & {diff} \\\\")
    open(args.out + ".tex", "w").write("\n".join(tex) + "\n")
    print("\n".join(md))
    print(f"[table] wrote {args.out}.json / .md / .tex")


def main():
    args = parse_args()
    settings = [T.parse_setting(t) for t in args.settings]
    if not args.collect:
        for st in settings:
            print(f"=== scenario {st['name']} ===", flush=True)
            paths, log = T.prepare(st, args)
            for n, kf, rule, k in grid(args):
                for m in args.methods:
                    T.search(st, paths, log, m, point_args(args, n, kf, rule, k))
    if args.dry_run or args.no_collect:
        return
    rows, everything = collect(settings, args)
    write_tables(rows, everything, args)


if __name__ == "__main__":
    main()
