#!/usr/bin/env python
"""Check which Stable Diffusion evaluations finished, and build the final table.

For every run directory in four_targets/outputs/<tier>_<Scenario>_seed<S>_<id>/ it looks at
eval/ (run.sh eval, SDXL Turbo, 2 steps) and eval_base/ (run.sh eval_base, multi-step SDXL
Base) and reports each one as

    done       eval_results.json written (the last thing eval_all_experiments does)
    unfinished run_info.txt but no eval_results.json: still running, or the job died
               (check `squeue -u $USER` / `sacct -j <id>` and the log)
    missing    never started

Then, from the finished ones, one table per run with both samplers side by side:
MMD, Delta MMD % (vs the source scribble), SWD and, in the gender scenarios, p(male) with
its 95% CI. Every number is read from eval_results.json; nothing is recomputed.

    python experiments/stable_diffusion/final_table.py                       # status + table
    python experiments/stable_diffusion/final_table.py --out final_table.md  # also save it
    python experiments/stable_diffusion/final_table.py --run-dir <one run directory>
No GPU, no model downloads; standard library only.
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "four_targets", "outputs")
ORDER = ["BalancedTarget", "SkewedTarget", "GenderInterpolation", "AgeInterpolation"]
SAMPLERS = [("eval", "Turbo (2 steps)"), ("eval_base", "Base (multi-step)")]
METHODS = ["source", "avg_prompt", "sdedit_best", "mlgd_f"]


def status(sub_dir):
    if os.path.exists(os.path.join(sub_dir, "eval_results.json")):
        return "done"
    if os.path.exists(os.path.join(sub_dir, "run_info.txt")):
        return "unfinished"
    return "missing"


def load(sub_dir):
    with open(os.path.join(sub_dir, "eval_results.json")) as fh:
        return json.load(fh)


def describe(rec):
    s = rec.get("eval_sampler_settings") or {}
    if rec.get("eval_sampler") == "base":
        return (f"N={rec['n_eval']}, {s.get('base_steps')} steps {s.get('base_scheduler')}, "
                f"CFG {s.get('base_guidance_scale')}, {s.get('base_resolution')}px")
    return f"N={rec['n_eval']}, 2 steps"


def run_table(name, recs):
    """recs: {sub: eval_results.json} for the finished samplers of one run."""
    gender = any(r.get("mode") == "gender" for r in recs.values())
    subs = [(s, label) for s, label in SAMPLERS if s in recs]
    head = ["Method"]
    for _, label in subs:
        head += [f"MMD {label}", "ΔMMD", "SWD"] + (["p(male) [95% CI]"] if gender else [])
    methods = [m for m in METHODS if any(m in r["results"] for r in recs.values())]
    methods += sorted({m for r in recs.values() for m in r["results"]} - set(methods))
    lines = [f"## {name}", ""]
    lines += [f"- {label}: {describe(recs[s])}" for s, label in subs]
    target = next((r.get("target_groups") for r in recs.values() if r.get("target_groups")), None)
    if gender and target:
        tot = sum(n for _, n in target)
        lines.append("- target: " + ", ".join(f"{g} {n / tot:.2f}" for g, n in target))
    lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    best = {s: min(r["results"].values(), key=lambda x: x["mmd"])["mmd"] for s, r in recs.items()}
    for m in methods:
        row = [m]
        for s, _ in subs:
            r = recs[s]["results"].get(m)
            if r is None:
                row += ["–"] * (4 if gender else 3)
                continue
            mmd = f"{r['mmd']:.3f}"
            row += [f"**{mmd}**" if r["mmd"] == best[s] else mmd,
                    f"{r.get('mmd_improvement_pct', 0):+.1f}%", f"{r['swd']:.4f}"]
            if gender:
                row.append(f"{r['p_male']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def scenario_key(name):
    for i, sc in enumerate(ORDER):
        if f"_{sc}" in name:
            return (i, name)
    return (99, name)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--outputs", default=OUTPUTS, help="where run.sh writes the runs")
    p.add_argument("--tier", default="full", help="runs named <tier>_* (default full)")
    p.add_argument("--run-dir", nargs="*", default=None, help="these run directories instead")
    p.add_argument("--out", default=None, help="also write the table (markdown) here")
    args = p.parse_args()

    runs = args.run_dir or [d for d in glob.glob(os.path.join(args.outputs, f"{args.tier}_*")) if os.path.isdir(d)]
    runs = sorted((os.path.abspath(d) for d in runs), key=lambda d: scenario_key(os.path.basename(d)))
    if not runs:
        raise SystemExit(f"no run directories {args.tier}_* in {args.outputs}")

    print("Status (done = eval_results.json written):")
    tables, pending = [], 0
    for d in runs:
        name = os.path.basename(d)
        st = {s: status(os.path.join(d, s)) for s, _ in SAMPLERS}
        opt = "done" if os.path.exists(os.path.join(d, "metrics.json")) else "unfinished"
        print(f"  {name:<50} optimisation {opt:<10} " + "  ".join(f"{s} {v:<10}" for s, v in st.items()))
        pending += sum(v == "unfinished" for v in st.values())
        recs = {s: load(os.path.join(d, s)) for s, v in st.items() if v == "done"}
        if recs:
            tables.append(run_table(name, recs))
    if pending:
        print(f"\n{pending} evaluation(s) started but not finished: still running, or the job died. "
              "Check `squeue -u $USER`, `sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode` "
              "and the end of logs/sd-evalbase-<jobid>.out.")
    if not tables:
        raise SystemExit("\nno finished evaluation yet")

    text = ("# Final evaluation: Turbo vs multi-step Base sampler\n\n"
            "Each sampler generates its own target and its own method images, so compare methods within "
            "a column, not across samplers. ΔMMD = improvement over the source scribble; bold = lowest MMD "
            "for that sampler.\n\n" + "\n\n".join(tables) + "\n")
    print("\n" + text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
