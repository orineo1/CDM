"""Per scenario and method: mean and SD over seeds of the loss change from the source sketch.

Reads <outputs>/full_<Scenario>_seed<S>[_zeta<Z>]_<jobid>/eval/eval_results.json. For each seed,
delta = loss(source sketch) - loss(method's final sketch) on the N = 2,000 evaluation (positive = better
than the start), and delta % = 100 * delta / loss(source). Only runs whose optimization was deterministic are used (metrics.json "deterministic": true; runs from
before that field existed, or with --no_deterministic, are skipped unless --allow-nondeterministic). If a
seed has several such runs, the latest job is used.

    python experiments/stable_diffusion/seed_deltas.py --outputs experiments/stable_diffusion/four_targets/outputs

--mlgdf-report writes only the MLGD-F comparison over seeds (mean ± std, and the seed selected by MMD
alone), once over every seed and once more without the seeds given in --without (e.g. AgeInterpolation:6).
"""
import argparse, glob, json, os, re
import statistics as st

p = argparse.ArgumentParser()
p.add_argument("--outputs", nargs="+", required=True, help="one or more four_targets/outputs directories")
p.add_argument("--zeta", default="7.5", help="base_zeta in the run name; 'any' keeps every run")
p.add_argument("--methods", nargs="+", default=["mlgd_f", "avg_prompt", "sdedit_best"])
p.add_argument("--seeds", nargs="*", type=int, help="keep only these seeds (default: all found)")
p.add_argument("--mlgdf-report", action="store_true",
               help="only the MLGD-F table over seeds: all seeds, then without the --without seeds")
p.add_argument("--without", nargs="*", default=[], metavar="SCENARIO:SEED",
               help="seeds left out of the second MLGD-F table, e.g. AgeInterpolation:6 AgeInterpolation:9")
p.add_argument("--allow-nondeterministic", action="store_true",
               help="also use runs whose optimization was not deterministic (or did not record it)")
a = p.parse_args()

pat = re.compile(r"^full_(?P<sc>[A-Za-z]+)_seed(?P<seed>\d+)(?:_zeta(?P<z>[\d.]+))?_(?P<job>\d+)$")
def deterministic(run_dir):
    try:
        return json.load(open(os.path.join(run_dir, "metrics.json"))).get("deterministic") is True
    except (OSError, ValueError):
        return False

runs, skipped = {}, []   # (scenario, seed) -> (job, path)
for d in a.outputs:
    for f in glob.glob(os.path.join(d, "full_*", "eval", "eval_results.json")):
        m = pat.match(os.path.basename(os.path.dirname(os.path.dirname(f))))
        if not m or (a.zeta != "any" and m["z"] != a.zeta):
            continue
        seed = int(m["seed"])
        if a.seeds and seed not in a.seeds:
            continue
        run_dir = os.path.dirname(os.path.dirname(f))
        if not a.allow_nondeterministic and not deterministic(run_dir):
            skipped.append(run_dir)
            continue
        key = (m["sc"], seed)
        if key not in runs or int(m["job"]) > runs[key][0]:
            runs[key] = (int(m["job"]), f)

order = ["BalancedTarget", "SkewedTarget", "GenderInterpolation", "AgeInterpolation"]
scen = sorted({s for s, _ in runs}, key=lambda s: order.index(s) if s in order else 99)
for loss in (() if a.mlgdf_report else ("mmd", "swd")):
    print(f"\n## {loss.upper()}: delta = {loss}(source) - {loss}(method), mean ± SD over seeds (positive = improvement)\n")
    print(f"| scenario | method | n seeds | {loss.upper()} (method) | Δ | Δ % | {loss.upper()} (source) |")
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
            print(f"| {s} | {meth} | {len(rows)} | {st.mean(met):.4f} ± {sd(met):.4f} | {st.mean(d):.4f} ± {sd(d):.4f} "
                  f"| {st.mean(dp):.1f} ± {sd(dp):.1f} | {st.mean(src):.4f} ± {sd(src):.4f} |")
def mlgdf_table(title, drop=()):
  print(f"\n## {title}\n")
  print("| scenario | n seeds | MMD | Δ MMD % | SWD | Δ SWD % | p(male) | selected seed | its MMD | its Δ MMD % | its SWD | its p(male) |")
  print("|---|---|---|---|---|---|---|---|---|---|---|---|")
  for s in scen:
    res = {k: json.load(open(runs[(s, k)][1]))["results"] for sc, k in runs if sc == s and f"{sc}:{k}" not in drop}
    res = {k: r for k, r in res.items() if "mlgd_f" in r and "source" in r}
    if not res:
        continue
    def col(f, fmt=".4f"):
        v = [f(r) for r in res.values()]
        v = [x for x in v if x is not None]
        if not v:
            return "-"
        return f"{st.mean(v):{fmt}} ± {st.stdev(v):{fmt}}" if len(v) > 1 else f"{v[0]:{fmt}}"
    def pct(loss):
        return lambda r: 100 * (r["source"][loss] - r["mlgd_f"][loss]) / r["source"][loss]
    pm = lambda r: r["mlgd_f"].get("p_male")
    k = min(res, key=lambda k: res[k]["mlgd_f"]["mmd"])
    r = res[k]
    sel_pm = f"{pm(r):.3f}" if pm(r) is not None else "-"
    print(f"| {s} | {len(res)} | {col(lambda r: r['mlgd_f']['mmd'])} | {col(pct('mmd'), '.1f')} | {col(lambda r: r['mlgd_f']['swd'])} "
          f"| {col(pct('swd'), '.1f')} | {col(pm, '.3f')} | {k} | {r['mlgd_f']['mmd']:.4f} | {pct('mmd')(r):+.1f} "
          f"| {r['mlgd_f']['swd']:.4f} | {sel_pm} |")

if a.mlgdf_report:
    print("# MLGD-F over seeds (Stable Diffusion, base_zeta " + a.zeta + ")\n")
    print("Each seed is a different source sketch. Per scenario: MLGD-F's final MMD and SWD to the target (CLIP")
    print("embeddings, N = 2,000 evaluation), their improvement over the source sketch (Δ %, positive = better),")
    print("and p(male), as mean ± std over the seeds; and the same for the seed selected by MMD alone (lowest")
    print("MLGD-F MMD). Only deterministic runs are used. Written by `experiments/stable_diffusion/seed_deltas.py --mlgdf-report`.")
mlgdf_table("MLGD-F: mean ± std over all seeds, and the seed selected by MMD alone (lowest MLGD-F MMD)")
if a.without:
    mlgdf_table("The same without " + ", ".join(w.replace(":", " seed ") for w in a.without), drop=set(a.without))

if skipped:
    print("\nskipped (optimization not deterministic, or not recorded):")
    for d in sorted(skipped):
        print(f"  {d}")
print("\nruns used:")
for (s, k), (job, f) in sorted(runs.items()):
    print(f"  {s} seed {k}: {os.path.basename(os.path.dirname(os.path.dirname(f)))}")
