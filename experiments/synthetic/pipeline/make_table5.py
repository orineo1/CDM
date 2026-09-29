"""Table 5 (network quality) from the check_models.py JSONs.

Reads one check JSON per scenario and prints one row per scenario with the columns of
the paper's Table 5: the score network's relative Fisher divergence (the worst value
over t > 1), the SWD of prior samples to P(X), and for each conditional sampler (DDPM,
iCT) the distance at x* and the mean SWD over random x. Written to <out>.md and <out>.csv.

    python experiments/synthetic/pipeline/make_table5.py experiments/synthetic/model_quality/outputs
    python experiments/synthetic/pipeline/make_table5.py <runs-dir>      # main_grid: <runs-dir>/<scenario>/check.json
"""

import argparse
import csv
import glob
import json
import os

SCENARIOS = ["x1_y1", "x1_y5", "x1_y10", "x5_y1", "x5_y5", "x5_y10", "x10_y1", "x10_y5", "x10_y10"]
COLUMNS = ["fisher", "prior_swd", "ddpm_xstar", "ddpm_random", "ict_xstar", "ict_random"]
HEADER = ["score net Fisher (rel.)", "prior SWD", "cond DDPM at x*", "cond DDPM random x", "iCT at x*", "iCT random x"]


def find_checks(d):
    """{scenario: path}: model_quality writes checks__<scenario>.json, main_grid <scenario>/check.json."""
    out = {}
    for p in glob.glob(os.path.join(d, "checks__*.json")) + glob.glob(os.path.join(d, "check_*.json")):
        name = os.path.basename(p)[:-5].split("__")[-1].replace("check_", "")
        out[name] = p
    for p in glob.glob(os.path.join(d, "*", "check.json")):
        out[os.path.basename(os.path.dirname(p))] = p
    return out


def row(c):
    fisher = max((o["rel_fisher"] for o in c.get("score", []) if o["t"] > 1), default=float("nan"))
    prior = c.get("prior", {}).get("swd_norm", float("nan"))

    def at(kind, key):
        r = c.get(kind)
        if not r:
            return float("nan")
        if key == "xstar":
            v = r.get("points", {}).get("x*", {}).get("mean")
            return float("nan") if v is None else v
        return r["swd_norm_random_x_mean"]

    return [fisher, prior, at("cond_diffusion", "xstar"), at("cond_diffusion", "random"),
            at("cond_cm", "xstar"), at("cond_cm", "random")]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checks_dir", help="directory with checks__<scenario>.json, or a runs dir with <scenario>/check.json")
    p.add_argument("--out", default=None, help="output stem (default <checks_dir>/table5)")
    args = p.parse_args()
    out = args.out or os.path.join(args.checks_dir, "table5")

    checks = find_checks(args.checks_dir)
    if not checks:
        raise SystemExit(f"no check JSONs under {args.checks_dir}")
    names = [s for s in SCENARIOS if s in checks] + sorted(set(checks) - set(SCENARIOS))

    rows = []
    for name in names:
        c = json.load(open(checks[name]))
        dx, dy = (int(v[1:]) for v in name.split("_")) if name.startswith("x") and "_y" in name else (None, None)
        rows.append({"scenario": name, "dim_x": dx, "dim_y": dy, **dict(zip(COLUMNS, row(c)))})

    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["| Setting | " + " | ".join(HEADER) + " |", "|---|" + "---|" * len(HEADER)]
    for r in rows:
        label = f"dim X = {r['dim_x']}, dim Y = {r['dim_y']}" if r["dim_x"] is not None else r["scenario"]
        md.append(f"| {label} | " + " | ".join(f"{r[k]:.3f}" for k in COLUMNS) + " |")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}.md and {out}.csv ({len(rows)} scenarios)")


if __name__ == "__main__":
    main()
