"""Per scenario and method: mean and SD over seeds of the loss change from the source sketch.

Reads <outputs>/full_<Scenario>_seed<S>[_zeta<Z>]_<jobid>/eval/eval_results.json. For each seed,
delta = loss(source sketch) - loss(method's final sketch) on the N = 2,000 evaluation (positive = better
than the start), and delta % = 100 * delta / loss(source). If a seed has several runs, the latest job is used.

    python experiments/stable_diffusion/seed_deltas.py --outputs experiments/stable_diffusion/four_targets/outputs
"""
import argparse, glob, json, os, re
import statistics as st

p = argparse.ArgumentParser()
p.add_argument("--outputs", nargs="+", required=True, help="one or more four_targets/outputs directories")
p.add_argument("--zeta", default="7.5", help="base_zeta in the run name; 'any' keeps every run")
p.add_argument("--methods", nargs="+", default=["mlgd_f", "avg_prompt", "sdedit_best"])
p.add_argument("--seeds", nargs="*", type=int, help="keep only these seeds (default: all found)")
a = p.parse_args()

pat = re.compile(r"^full_(?P<sc>[A-Za-z]+)_seed(?P<seed>\d+)(?:_zeta(?P<z>[\d.]+))?_(?P<job>\d+)$")
runs = {}   # (scenario, seed) -> (job, path)
for d in a.outputs:
    for f in glob.glob(os.path.join(d, "full_*", "eval", "eval_results.json")):
        m = pat.match(os.path.basename(os.path.dirname(os.path.dirname(f))))
        if not m or (a.zeta != "any" and m["z"] != a.zeta):
            continue
        seed = int(m["seed"])
        if a.seeds and seed not in a.seeds:
            continue
        key = (m["sc"], seed)
        if key not in runs or int(m["job"]) > runs[key][0]:
            runs[key] = (int(m["job"]), f)

order = ["BalancedTarget", "SkewedTarget", "GenderInterpolation", "AgeInterpolation"]
scen = sorted({s for s, _ in runs}, key=lambda s: order.index(s) if s in order else 99)
for loss in ("mmd", "swd"):
    print(f"\n## {loss.upper()}: delta = {loss}(source) - {loss}(method), mean ± SD over seeds (positive = improvement)\n")
    print("| scenario | method | n seeds | Δ | Δ % | source | method |")
    print("|---|---|---|---|---|---|---|")
    for s in scen:
        seeds = sorted(k for sc, k in runs if sc == s)
        res = {k: json.load(open(runs[(s, k)][1]))["results"] for k in seeds}
        for meth in a.methods:
            rows = [(r["source"][loss], r[meth][loss]) for r in res.values() if meth in r and "source" in r]
            if not rows:
                continue
            src, met = [r[0] for r in rows], [r[1] for r in rows]
            d = [x - y for x, y in rows]
            dp = [100 * (x - y) / x for x, y in rows]
            sd = lambda v: st.stdev(v) if len(v) > 1 else float("nan")   # sample SD (n - 1)
            print(f"| {s} | {meth} | {len(rows)} | {st.mean(d):.4f} ± {sd(d):.4f} | {st.mean(dp):.1f} ± {sd(dp):.1f} "
                  f"| {st.mean(src):.4f} | {st.mean(met):.4f} |")
print("\nruns used:")
for (s, k), (job, f) in sorted(runs.items()):
    print(f"  {s} seed {k}: {os.path.dirname(os.path.dirname(f))}")
