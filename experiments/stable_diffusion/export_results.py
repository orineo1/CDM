#!/usr/bin/env python
"""Copy finished Stable Diffusion runs into the committed results/ folder.

Reads the run directories that `four_targets/run.sh full` writes
(four_targets/outputs/full_<Scenario>/) and writes files that are safe and small to
commit, so the results can be read without re-running:

    four_targets/results/mlgdf_runs.md          one row per scenario: final MMD, paper MMD,
                                                time, settings, commit, wandb link
    four_targets/results/<Scenario>/
        metrics.json                            the run's full record: args, every step,
                                                final MMD, time, wandb run id / url
        steps.csv                               one row per guided step: MMD, gradient
                                                norm, zeta, correction norm, elapsed time
        run_info.txt                            date, commit, host, GPU, exact command
        curves.png                              the per-step curves drawn from steps.csv
        scribble_source.png, source_portrait.png, final_scribble_mlgd_f.png
        final_photos_mlgd_f.png, target_clip_pca.png, target_samples_*.png
        (with --run_unguided also final_scribble_regular.png, scribble_heatmap.png,
         final_photos_regular.png)

    four_targets/results/eval_table.md          Table 3 / 15 of this code: every scenario's
                                                N = 2,000 evaluation (from outputs/eval_*)
    four_targets/results/<Scenario>/eval/
        eval_results.json, eval_table.md, run_info.txt, the 5-image previews of every
        method and target group, the MMD / SWD bars and the p(male) CI plot
    interpolation/results/<Scenario>/
        sorted_grid.png, pca_scatter.png, ranked_all.png, picks.json (from the notebook's
        outputs/<Scenario>)

Everything here is also in the run's wandb log. The per-step visualisations are logged
to wandb only and not saved to disk; the large arrays (npy/, the 250 evaluation
photos) stay in outputs/.

    python experiments/stable_diffusion/export_results.py
    python experiments/stable_diffusion/export_results.py --tier smoke --dest /tmp/sd_smoke_export
"""

import argparse
import csv
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "four_targets")
INTERP = os.path.join(HERE, "interpolation")
EVAL_FILES = ["eval_results.json", "eval_table.md", "run_info.txt", "mmd_bar.png", "swd_bar.png", "p_male_ci.png"]
INTERP_FILES = ["sorted_grid.png", "pca_scatter.png", "ranked_all.png", "picks.json"]
ORDER = ["BalancedTarget", "SkewedTarget", "GenderInterpolation", "AgeInterpolation"]
# MLGD-F MMD from paper Table 3 (N = 2,000 evaluation images; see results/table3_sd_results.md)
PAPER_MMD = {"BalancedTarget": 0.386, "SkewedTarget": 0.373,
             "GenderInterpolation": 0.328, "AgeInterpolation": 0.381}
SINGLE_FILES = ["run_info.txt", "source_portrait.png", "final_scribble_mlgd_f.png",
                "final_photos_mlgd_f.png", "target_clip_pca.png",
                "final_scribble_regular.png", "scribble_heatmap.png", "final_photos_regular.png"]
MAX_GALLERIES = 8   # the age target has one gallery per age; beyond this only the ends are kept
CURVES = [("mmd_loss", "MMD (per-step estimate)"), ("gradient_norm", "gradient norm"),
          ("zeta_i", "zeta"), ("correction_norm", "correction norm")]


def read_steps(run_dir, metrics):
    path = os.path.join(run_dir, "steps.csv")
    if os.path.exists(path):
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
    return [{k: ("" if v is None else v) for k, v in s.items()} for s in metrics.get("steps", [])]


def write_steps_csv(steps, path):
    fields = list(dict.fromkeys(k for s in steps for k in s))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(steps)


def plot_curves(steps, title, path):
    x = [int(s["step"]) for s in steps]
    fig, axes = plt.subplots(1, len(CURVES), figsize=(4.2 * len(CURVES), 3.4))
    for ax, (key, label) in zip(axes, CURVES):
        ax.plot(x, [float(s[key]) for s in steps], lw=1.4)
        if key == "mmd_loss" and any(s.get("unguided_cond_mmd") not in (None, "") for s in steps):
            pts = [(int(s["step"]), float(s["unguided_cond_mmd"])) for s in steps
                   if s.get("unguided_cond_mmd") not in (None, "")]
            ax.plot(*zip(*pts), "o--", ms=4, lw=1, label="unguided")
            ax.legend(fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("guided step")
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def export_run(run_dir, dest):
    with open(os.path.join(run_dir, "metrics.json")) as fh:
        metrics = json.load(fh)
    os.makedirs(dest, exist_ok=True)

    shutil.copy2(os.path.join(run_dir, "metrics.json"), dest)
    steps = read_steps(run_dir, metrics)
    if steps:
        write_steps_csv(steps, os.path.join(dest, "steps.csv"))
        plot_curves(steps, os.path.basename(dest), os.path.join(dest, "curves.png"))
    for name in SINGLE_FILES:
        src = os.path.join(run_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dest)
    if os.path.exists(os.path.join(run_dir, "scribble.png")):   # the name eval_baselines expects
        shutil.copy2(os.path.join(run_dir, "scribble.png"), os.path.join(dest, "scribble_source.png"))
    galleries = sorted(glob.glob(os.path.join(run_dir, "target_samples_*.png")))
    if len(galleries) > MAX_GALLERIES:
        galleries = [galleries[0], galleries[-1]]
    for g in galleries:
        shutil.copy2(g, dest)
    return metrics, steps


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def export_evals(run_dir, dest):
    """outputs/eval_<Scenario>/ -> results/<Scenario>/eval/ (no photos, only previews);
    returns the per-scenario tables for the combined results/eval_table.md."""
    tables = []
    for d in sorted(glob.glob(os.path.join(run_dir, "eval_*"))):
        if not os.path.exists(os.path.join(d, "eval_results.json")):
            print(f"skip {d}: no eval_results.json (evaluation not finished?)")
            continue
        name = os.path.basename(d)[len("eval_"):]
        out = os.path.join(dest, name, "eval")
        os.makedirs(out, exist_ok=True)
        for f in EVAL_FILES + [os.path.basename(p) for p in glob.glob(os.path.join(d, "*preview_*.png"))]:
            if os.path.exists(os.path.join(d, f)):
                shutil.copy2(os.path.join(d, f), out)
        with open(os.path.join(d, "eval_table.md")) as fh:
            tables.append((name, fh.read().strip()))
        print(f"exported {d} -> {out}")
    if tables:
        with open(os.path.join(dest, "eval_table.md"), "w") as fh:
            fh.write("# Final evaluation (N = 2,000), every scenario\n\n"
                     "Written by `experiments/stable_diffusion/export_results.py` from `outputs/eval_*`; the\n"
                     "protocol of the paper's Table 3 / Table 15 (`cdm/models/sd/eval_all_experiments.py`).\n"
                     "Delta% is the improvement over the source scribble.\n\n")
            fh.write("\n\n".join(t.replace("# ", "## ", 1) for _, t in tables) + "\n")
        print(f"wrote {os.path.join(dest, 'eval_table.md')}")
    return tables


def export_interpolation(run_dir, dest):
    """interpolation/outputs/<Scenario>/ -> interpolation/results/<Scenario>/."""
    for d in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if not os.path.exists(os.path.join(d, "picks.json")):
            continue
        out = os.path.join(dest, os.path.basename(d))
        os.makedirs(out, exist_ok=True)
        for f in INTERP_FILES:
            if os.path.exists(os.path.join(d, f)):
                shutil.copy2(os.path.join(d, f), out)
        print(f"exported {d} -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tier", default="full", help="export outputs/<tier>_* (default full)")
    p.add_argument("--run-dir", default=os.path.join(EXP, "outputs"),
                   help="where run.sh wrote the runs (default four_targets/outputs)")
    p.add_argument("--dest", default=os.path.join(EXP, "results"),
                   help="where to write (default four_targets/results, which is committed)")
    p.add_argument("--interp-dir", default=os.path.join(INTERP, "outputs"),
                   help="the interpolation notebook's outputs (default interpolation/outputs)")
    p.add_argument("--interp-dest", default=os.path.join(INTERP, "results"))
    args = p.parse_args()

    export_evals(args.run_dir, args.dest)
    export_interpolation(args.interp_dir, args.interp_dest)

    runs = {}
    for d in sorted(glob.glob(os.path.join(args.run_dir, f"{args.tier}_*"))):
        if os.path.exists(os.path.join(d, "metrics.json")):
            runs[os.path.basename(d)[len(args.tier) + 1:]] = d
        elif os.path.isdir(d):
            print(f"skip {d}: no metrics.json (run not finished?)")
    if not runs:
        print(f"no finished runs ({args.tier}_*/metrics.json) in {args.run_dir}; "
              f"run four_targets/run.sh {args.tier} <scenario> first")
        return
    names = [n for n in ORDER if n in runs] + sorted(set(runs) - set(ORDER))

    rows = []
    for name in names:
        dest = os.path.join(args.dest, name)
        metrics, steps = export_run(runs[name], dest)
        a, wb = metrics.get("args", {}), metrics.get("wandb") or {}
        commit = "unknown"
        info = os.path.join(runs[name], "run_info.txt")
        if os.path.exists(info):
            for line in open(info):
                if line.startswith("git commit:"):
                    commit = line.split(":", 1)[1].strip()
        rows.append({
            "scenario": name,
            "mmd": metrics.get("final_mlgd_f_mmd"),
            "paper": PAPER_MMD.get(name),
            "unguided": metrics.get("final_regular_mmd"),
            "n": metrics.get("final_mmd_n", 250),
            "steps": len(steps),
            "vars": a.get("num_variations"), "k": a.get("backsel_k"),
            "zeta": a.get("base_zeta"), "seed": a.get("seed"),
            "min": (metrics.get("optimization_time_sec") or 0) / 60,
            "commit": commit,
            "wandb": f"[{wb.get('name')}]({wb['url']})" if wb.get("url") else (wb.get("name") or "--"),
        })
        print(f"exported {runs[name]} -> {dest}")

    lines = [
        f"# MLGD-F runs ({args.tier}), exported from `outputs/{args.tier}_*`",
        "",
        "Written by `experiments/stable_diffusion/export_results.py`. MMD is the final",
        "(MMD²)^(1/2) of the run on n evaluation images; the paper's MMD (Table 3) uses",
        "N = 2,000, so the two are comparable in scale but not identical protocols.",
        "",
        "| Scenario | MMD (n) ↓ | paper MMD | unguided MMD | guided steps | variations / k | ζ | seed | time (min) | commit | wandb |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['scenario']} | {fmt(r['mmd'])} ({r['n']}) | {fmt(r['paper'])} | "
                     f"{fmt(r['unguided'])} | {r['steps']} | {r['vars']} / {r['k']} | {r['zeta']} | "
                     f"{r['seed']} | {r['min']:.1f} | {r['commit']} | {r['wandb']} |")
    os.makedirs(args.dest, exist_ok=True)
    summary = os.path.join(args.dest, "mlgdf_runs.md")
    with open(summary, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
