"""
The search x gradient estimator table (2 x 2): TV of MLGD* and GB-FESO (MMD), each with FiniteDiff and with
Backprop, on the retrained (d256) samplers, as LaTeX rows (lowest mean per row in bold). One result file per run:

  MLGD*, FiniteDiff     <gbfeso-dir>/<cell>/MLGDstar_s*.json         (the MLGD* runs of make_table1_d256.py)
  MLGD*, Backprop       <x-est-dir>/<cell>/mlgd_pw_s*.json           (slurm_search_x_estimator.sh)
  GB-FESO, FiniteDiff   <x-est-dir>/<cell>/gbfeso_zo_pw_s*.json      (slurm_search_x_estimator.sh)
  GB-FESO, Backprop     <gbfeso-dir>/<cell>/GBFESO-MMD_s*.json       (the GB-FESO (MMD) runs of make_table1_d256.py)

TV as make_table1_d256.py (make_table1.scale_free_distances, 20,000 Monte Carlo draws, seed 0); mean and population
std over the runs. A cell with other than 5 runs is marked (n=...). Run from the repository root:

    python experiments/synthetic/pipeline/make_table_2x2.py --data-root runs/table1v2 > results/table_2x2.tex
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_table1 import _load_mixture, scale_free_distances   # noqa: E402

CELLS = ["x1_y1", "x1_y5", "x1_y10", "x5_y1", "x5_y5", "x5_y10", "x10_y1", "x10_y5", "x10_y10"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="runs/table1v2", help="directory holding <cell>/data.pt")
    p.add_argument("--gbfeso-dir", default="runs/gbfeso_d256")
    p.add_argument("--x-est-dir", default="runs/search_x_estimator")
    a = p.parse_args()
    cols = [("MLGD$^\\star$, FiniteDiff", os.path.join(a.gbfeso_dir, "{c}", "MLGDstar_s*.json")),
            ("MLGD$^\\star$, Backprop", os.path.join(a.x_est_dir, "{c}", "mlgd_pw_s*.json")),
            ("GB-FESO, FiniteDiff", os.path.join(a.x_est_dir, "{c}", "gbfeso_zo_pw_s*.json")),
            ("GB-FESO, Backprop", os.path.join(a.gbfeso_dir, "{c}", "GBFESO-MMD_s*.json"))]
    print("Cell & " + " & ".join(n for n, _ in cols) + " \\\\\n\\hline")
    for c in CELLS:
        data = _load_mixture(os.path.join(a.data_root, c, "data.pt"))
        stats = []
        for _, pat in cols:
            tv = [scale_free_distances(data, json.load(open(f))["attempts"][0]["x_hat"])["tv"]
                  for f in sorted(glob.glob(pat.format(c=c)))]
            stats.append((np.mean(tv) if tv else np.nan, np.std(tv) if tv else np.nan, len(tv)))
        best = np.nanmin([m for m, _, _ in stats])
        dx, dy = c[1:].split("_y")
        cells = [(f"$\\mathbf{{{m:.3f} \\pm {s:.3f}}}$" if m == best else f"${m:.3f} \\pm {s:.3f}$")
                 + ("" if n == 5 else f" (n={n})") for m, s, n in stats]
        print(f"({dx}, {dy}) & " + " & ".join(cells) + " \\\\", flush=True)


if __name__ == "__main__":
    main()
