#!/usr/bin/env python
"""Copy finished Stable Diffusion runs into the committed results/ folder.

Reads the run directories that `four_targets/run.sh full` writes, one per run
(four_targets/outputs/full_<Scenario>_seed<S>_<jobid>/, with the baselines/ and eval/ that
`run.sh baselines` / `run.sh eval` add inside), and writes only what the results are read
from, each file once:

    four_targets/results/eval_table.md          every run's N = 2,000 table (MMD, SWD,
                                                p(male) with 95% CI; Table 3 / 15)
    four_targets/results/<run>/                 <run> = the run directory's name
        metrics.json                            the run's full record: args, every step,
                                                final MMD, time, model revisions, wandb link
        run_info.txt                            date, commit, host, GPU, exact command
        curves.png                              the per-step MMD, gradient norm, zeta, correction
        scribble_source.png, source_portrait.png, final_scribble_mlgd_f.png
        baselines/                              scribble_avg_prompt.png, scribble_sdedit_best.png,
                                                avg_prompt_portrait.png, sdedit_search.png,
                                                baselines_meta.json, run_info.txt
        eval/                                   eval_results.json (every number), run_info.txt,
                                                preview_<method>.png and target_preview_<group>.png
                                                (5 portraits each)
        saliency/                               saliency_results.json, scribble_change.png (Fig. 7),
                                                saliency_correlation.png (Fig. 8), run_info.txt
    step_cost/results/<GPU>_<id>/               step_cost.md (the two tables), step_cost.json,
                                                run_info.txt
    interpolation/results/<Scenario>/
        sorted_grid.png, pca_scatter.png, ranked_all.png, picks.json (from the notebook's
        outputs/<Scenario>)

<run> is the directory name as given, so to publish a run without its job id, point
--run-dir at a folder of symlinks named e.g. full_BalancedTarget_seed4. Left in outputs/:
steps.csv (also inside metrics.json), the photo grids, the target galleries, the plots of
numbers that are in eval_table.md, npy/ and the embeddings.

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
STEP_COST = os.path.join(HERE, "step_cost")
BASELINE_FILES = ["scribble_avg_prompt.png", "scribble_sdedit_best.png", "avg_prompt_portrait.png",
                  "sdedit_search.png", "baselines_meta.json", "run_info.txt"]
EVAL_FILES = ["eval_results.json", "run_info.txt"]
SALIENCY_FILES = ["saliency_results.json", "scribble_change.png", "saliency_correlation.png", "run_info.txt"]
STEP_COST_FILES = ["step_cost.md", "step_cost.json", "run_info.txt"]
INTERP_FILES = ["sorted_grid.png", "pca_scatter.png", "ranked_all.png", "picks.json"]
ORDER = ["BalancedTarget", "SkewedTarget", "GenderInterpolation", "AgeInterpolation"]
SINGLE_FILES = ["metrics.json", "run_info.txt", "source_portrait.png", "final_scribble_mlgd_f.png",
                "final_scribble_regular.png"]   # the last only with --run_unguided
CURVES = [("mmd_loss", "MMD (per-step estimate)"), ("gradient_norm", "gradient norm"),
          ("zeta_i", "zeta"), ("correction_norm", "correction norm")]


def read_steps(run_dir, metrics):
    path = os.path.join(run_dir, "steps.csv")
    if os.path.exists(path):
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
    return [{k: ("" if v is None else v) for k, v in s.items()} for s in metrics.get("steps", [])]


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

    steps = read_steps(run_dir, metrics)
    if steps:
        plot_curves(steps, os.path.basename(dest), os.path.join(dest, "curves.png"))
    for name in SINGLE_FILES:
        src = os.path.join(run_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dest)
    if os.path.exists(os.path.join(run_dir, "scribble.png")):   # the name eval_baselines expects
        shutil.copy2(os.path.join(run_dir, "scribble.png"), os.path.join(dest, "scribble_source.png"))
    copy_sub(os.path.join(run_dir, "baselines"), os.path.join(dest, "baselines"), BASELINE_FILES, ())
    copy_sub(os.path.join(run_dir, "eval"), os.path.join(dest, "eval"), EVAL_FILES, ("*preview_*.png",))
    copy_sub(os.path.join(run_dir, "saliency"), os.path.join(dest, "saliency"), SALIENCY_FILES, ())


def copy_sub(src, dest, names, patterns):
    """Copy the listed files and glob patterns of src (if it exists) into dest."""
    if not os.path.isdir(src):
        return
    os.makedirs(dest, exist_ok=True)
    files = [os.path.join(src, n) for n in names] + [f for pat in patterns for f in glob.glob(os.path.join(src, pat))]
    for f in files:
        if os.path.isfile(f):
            shutil.copy2(f, dest)


def split_run_name(name, tier):
    """full_<Scenario>_seed<S>_<id> -> (Scenario, id); the older full_<Scenario> -> (Scenario, None)."""
    rest = name[len(tier) + 1:]
    if "_seed" in rest:
        scen, tail = rest.split("_seed", 1)
        return scen, tail.partition("_")[2] or None
    return rest, None


def read_eval_table(run_dir):
    """The run's N = 2,000 table (markdown) from <run>/eval/, or None."""
    path = os.path.join(run_dir, "eval", "eval_table.md")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return fh.read().strip()


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


def export_step_cost(cost_dir, dest):
    """step_cost/outputs/<GPU>_<id>/ -> step_cost/results/<GPU>_<id>/."""
    for d in sorted(glob.glob(os.path.join(cost_dir, "*"))):
        if os.path.exists(os.path.join(d, "step_cost.json")):
            copy_sub(d, os.path.join(dest, os.path.basename(d)), STEP_COST_FILES, ())
            print(f"exported {d} -> {os.path.join(dest, os.path.basename(d))}")


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
    p.add_argument("--cost-dir", default=os.path.join(STEP_COST, "outputs"),
                   help="step_cost/run.sh's outputs (default step_cost/outputs)")
    p.add_argument("--cost-dest", default=os.path.join(STEP_COST, "results"))
    args = p.parse_args()

    export_evals(args.run_dir, args.dest)
    export_interpolation(args.interp_dir, args.interp_dest)
    export_step_cost(args.cost_dir, args.cost_dest)

    runs = {}
    for d in sorted(glob.glob(os.path.join(args.run_dir, f"{args.tier}_*"))):
        if os.path.exists(os.path.join(d, "metrics.json")):
            runs[os.path.basename(d)] = d
        elif os.path.isdir(d):
            print(f"skip {d}: no metrics.json (run not finished?)")
    if not runs:
        print(f"no finished runs ({args.tier}_*/metrics.json) in {args.run_dir}; "
              f"run four_targets/run.sh {args.tier} <scenario> first")
        return
    rank = {sc: i for i, sc in enumerate(ORDER)}
    names = sorted(runs, key=lambda n: (rank.get(split_run_name(n, args.tier)[0], 99), n))

    eval_tables = []
    for name in names:
        dest = os.path.join(args.dest, name)
        export_run(runs[name], dest)
        table = read_eval_table(runs[name])
        if table:   # "# <eval dir>: N = 2000" -> "## <run>: N = 2000"
            eval_tables.append(f"## {name}: " + table.split(": ", 1)[-1] if table.startswith("# ") else table)
        print(f"exported {runs[name]} -> {dest}")

    if eval_tables:
        os.makedirs(args.dest, exist_ok=True)
        with open(os.path.join(args.dest, "eval_table.md"), "w") as fh:
            fh.write("# Final evaluation (N = 2,000), every run\n\n"
                     "Written by `experiments/stable_diffusion/export_results.py` from each run's `eval/`; the\n"
                     "protocol of the paper's Table 3 / Table 15 (`cdm/models/sd/eval_all_experiments.py`).\n"
                     "Delta% is the improvement over the source scribble. Every number is also in\n"
                     "`<run>/eval/eval_results.json`.\n\n")
            fh.write("\n\n".join(eval_tables) + "\n")
        print(f"wrote {os.path.join(args.dest, 'eval_table.md')}")


if __name__ == "__main__":
    main()
