#!/usr/bin/env python
"""Copy the finished synthetic runs into the committed results/ folders.

For each experiment that produces a table or a figure of the paper, copies the small
files a reader needs (tables as .md/.csv/.json, figures as .png, the per-restart data)
from <experiment>/outputs/ to <experiment>/results/, and writes results/run_info.txt
(date, commit, GPU, torch). Networks, data.pt and per-trajectory traces stay in outputs/.

    python experiments/synthetic/export_results.py                 # every experiment that has outputs
    python experiments/synthetic/export_results.py main_grid toy_example   # some of them

Then:  python experiments/synthetic/pipeline/compare_to_paper.py --results
"""

import argparse
import datetime
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# experiment -> glob patterns under outputs/ (relative), copied flat into results/
FILES = {
    "main_grid": ["table1.json", "table1.md", "table1.csv", "table1.tex", "table1_compact.tex", "table1_models.tex",
                  "table1_appendix.md", "cells.csv", "runs/*/check.png"],
    "model_quality": ["table5.md", "table5.csv", "checks__*.json", "checks__*.png"],
    "ablations": ["table8.md", "table8.csv", "table8_*.json", "table8_*.md"],
    "infeasible_target": ["table9.md", "table9.csv", "table9_restarts.csv", "table9_fig4.png", "arm_*.json"],
    "training_times": ["timings.csv", "timings.txt"],
    "toy_example": ["fig1_*.png", "fig1_*_sim.json", "fig1_*_sweep.npz"],
    "smc_correction": ["fig_smc_hard.png", "fig_smc_hard.json", "fig_smc_hard.npz"],
}


def run_info():
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"
    gpu = sh([sys.executable, "-c", "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"])
    torch_v = sh([sys.executable, "-c", "import torch; print(torch.__version__, 'cuda', torch.version.cuda)"])
    return (f"date:        {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"git commit:  {sh(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'])}\n"
            f"gpu:         {gpu}\ntorch:       {torch_v}\n")


def cells_csv(main_grid_out, dest):
    """One row per scenario x method x attempt from main_grid's table1.json."""
    import csv
    import json
    path = os.path.join(main_grid_out, "table1.json")
    if not os.path.exists(path):
        return
    t = json.load(open(path))
    rows = []
    for name, e in t["scenarios"].items():
        for m, r in e.items():
            if not isinstance(r, dict) or "attempts" not in r:
                continue
            for i, a in enumerate(r["attempts"]):
                rows.append({"scenario": name, "method": m, "attempt": i,
                             "tv": a.get("tv"), "l2_rel": a.get("l2_rel"), "l2_to_xstar_sd": a.get("l2_to_xstar"),
                             "l2_gmm": a.get("l2_gmm"), "final_loss": a.get("final_loss"),
                             "time_s": a.get("time_s"), "n_resamples": a.get("n_resamples")})
    if rows:
        with open(os.path.join(dest, "cells.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(dest, "table1.csv"), "w", newline="") as fh:     # the table itself, one row per cell
        w = csv.DictWriter(fh, fieldnames=list(t["rows"][0]))
        w.writeheader()
        w.writerows(t["rows"])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("experiments", nargs="*", default=list(FILES), help=f"default: {' '.join(FILES)}")
    args = p.parse_args()

    info = run_info()
    for exp in args.experiments:
        if exp not in FILES:
            raise SystemExit(f"unknown experiment {exp}; one of {', '.join(FILES)}")
        out, res = os.path.join(HERE, exp, "outputs"), os.path.join(HERE, exp, "results")
        os.makedirs(res, exist_ok=True)
        if exp == "main_grid":
            cells_csv(out, out)
        copied = []
        for pat in FILES[exp]:
            for src in sorted(glob.glob(os.path.join(out, pat))):
                name = os.path.basename(src)
                if pat.startswith("runs/"):               # runs/<scenario>/check.png -> check_<scenario>.png
                    name = f"check_{os.path.basename(os.path.dirname(src))}.png"
                shutil.copy(src, os.path.join(res, name))
                copied.append(name)
        if copied:
            open(os.path.join(res, "run_info.txt"), "w").write(info)
            print(f"{exp:20s} {len(copied)} files -> results/: {', '.join(copied[:6])}{', ...' if len(copied) > 6 else ''}")
        else:
            print(f"{exp:20s} nothing in outputs/ (run its run.sh full first)")
    print("\ncommit with:  git add experiments/synthetic/*/results")
    print("compare with: python experiments/synthetic/pipeline/compare_to_paper.py --results")


if __name__ == "__main__":
    main()
