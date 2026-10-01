"""
Ground-truth evaluation: every returned eps_hat is re-simulated (sim_ensembles.py, the same protocol as
the targets) and compared with the target ensemble using GB-FESO's own metrics (their utils.py basins
and Delta F, their loss.py KDE-KL with bandwidth 0.5, kl_lambda 1, raw 5D coordinates).

  1. python experiments/peptide/pipeline/evaluate.py collect --results runs/peptide/results/*.json \
         --targets-ens runs/peptide/targets_ens.pt --out runs/peptide/eval_eps.npy
     (adds an 'oracle' row eps_hat = eps*: an independent re-simulation at the true eps, the noise floor)
  2. python experiments/peptide/pipeline/sim_ensembles.py --eps-file runs/peptide/eval_eps.npy --keep 20000 \
         --seed 3 --out runs/peptide/eval_ens.pt
  3. python experiments/peptide/pipeline/evaluate.py metrics --eval-eps runs/peptide/eval_eps.npy \
         --eval-ens runs/peptide/eval_ens.pt --targets-ens runs/peptide/targets_ens.pt \
         --out experiments/peptide/results/

Success = every target basin with mass > 5% has mass >= 0.5% at eps_hat (their compare_basin_metrics:
`missing_active_basin` is False). Delta F error = their per-run median |dF_hat - dF| over the target's
non-reference active basins (k_BT, capped at 6), multi-basin targets only. eps_hat is clamped to the box
[0.21, 1.55]^2 before simulation (GB-FESO clamps during the search; ours is clamped only here).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common   # noqa: E402,F401
from simulate import EPS_BOX   # noqa: E402
from utils import compare_basin_metrics   # noqa: E402  (GB-FESO)
from loss import kde_kl_target_to_generated   # noqa: E402  (GB-FESO)


def load_runs(paths):
    runs = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            d = json.load(open(f))
            if "runs" in d:
                for r in d["runs"]:
                    method = d["method"] + (f" [{r['start']}]" if "start" in r else "")
                    runs.append({"method": method, "family": d["method"], **{k: r[k] for k in r if k not in ("loss_trace",)}})
            else:    # find_xstar output: one target per file, named <method>_t<id>.json
                tid = int(os.path.basename(f).split("_t")[-1].split(".")[0])
                b = d["best"]
                runs.append({"method": d["method"], "family": d["method"], "target_id": tid,
                             "x_hat_std": b["x_hat"], "eps_hat": (np.array(b["x_hat"]) * common.X_STD + common.X_MEAN).tolist(),
                             "surrogate_loss": b.get("select_loss", b["final_loss"]),
                             "time_s": d["summary"]["time_s"]["mean"]})
    return runs


def cmd_collect(a):
    runs = load_runs(a.results)
    te = torch.load(a.targets_ens, weights_only=False)
    for tid in range(te["eps"].shape[0]):
        runs.append({"method": "oracle (eps_hat = eps*)", "family": "oracle", "target_id": tid,
                     "eps_hat": te["eps"][tid].tolist(), "time_s": 0.0})
    for r in runs:
        r["eps_star"] = te["eps"][r["target_id"]].tolist()
        r["eps_eval"] = np.clip(r["eps_hat"], *EPS_BOX).tolist()
    np.save(a.out, np.array([r["eps_eval"] for r in runs], dtype=np.float32))
    json.dump(runs, open(os.path.splitext(a.out)[0] + "_runs.json", "w"))
    print(f"[evaluate] {len(runs)} runs -> {a.out}")


def cmd_metrics(a):
    runs = json.load(open(os.path.splitext(a.eval_eps)[0] + "_runs.json"))
    ev = torch.load(a.eval_ens, weights_only=False)
    te = torch.load(a.targets_ens, weights_only=False)
    assert ev["Y"].shape[0] == len(runs)
    g = torch.Generator().manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for i, r in enumerate(runs):
        T, G = te["Y"][r["target_id"]], ev["Y"][i]
        m = compare_basin_metrics(T, G)
        it = torch.randperm(T.shape[0], generator=g)[:a.kl_n]
        ig = torch.randperm(G.shape[0], generator=g)[:a.kl_n]
        kl = float(kde_kl_target_to_generated(T[it].to(dev), G[ig].to(dev), bandwidth=0.5, kl_lambda=1.0))
        r.update({"success": not m["missing_active_basin"], "multi_basin": m["target_type"] == "multi_basin",
                  "df_median_abs_error": m["df_median_abs_error"], "df_mae": m["df_mae"], "tv_basins": m["total_variation"],
                  "target_probs": m["target_probs"], "generated_probs": m["generated_probs"], "kl": kl,
                  "eps_err": float(np.linalg.norm(np.array(r["eps_eval"]) - np.array(r["eps_star"])))})
    os.makedirs(a.out, exist_ok=True)
    json.dump(runs, open(os.path.join(a.out, "per_run.json"), "w"), indent=0)
    write_table(runs, os.path.join(a.out, "results_table.md"))


def summarize(rs):
    df = [r["df_median_abs_error"] for r in rs if r["multi_basin"] and r["df_median_abs_error"] is not None]
    return {"n": len(rs), "success": np.mean([r["success"] for r in rs]),
            "n_fail": int(sum(not r["success"] for r in rs)),
            "df_med": np.median(df) if df else float("nan"), "n_multi": len(df),
            "kl_med": np.median([r["kl"] for r in rs]), "kl_mean": np.mean([r["kl"] for r in rs]),
            "tv_med": np.median([r["tv_basins"] for r in rs]),
            "eps_err_med": np.median([r["eps_err"] for r in rs]),
            "time_med": np.median([r["time_s"] for r in rs]), "time_mean": np.mean([r["time_s"] for r in rs])}


def write_table(runs, path):
    order = []
    for r in runs:
        if r["method"] not in order:
            order.append(r["method"])
    fams = []
    for r in runs:
        if r["family"] not in fams:
            fams.append(r["family"])
    lines = ["| method | runs | success | failures | median dF err (k_BT) | median KDE-KL | mean KDE-KL | median basin TV | median abs(eps_hat - eps*) | time / run (s), median (mean) |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, rs):
        s = summarize(rs)
        lines.append(f"| {name} | {s['n']} | {100 * s['success']:.1f}% | {s['n_fail']} | {s['df_med']:.3f} (n={s['n_multi']}) | "
                     f"{s['kl_med']:.3f} | {s['kl_mean']:.3f} | {s['tv_med']:.3f} | {s['eps_err_med']:.3f} | {s['time_med']:.1f} ({s['time_mean']:.1f}) |")
    for f in fams:
        rs = [r for r in runs if r["family"] == f]
        row(f"**{f}** (all)", rs)
        subs = [m for m in order if m != f and any(r["method"] == m and r["family"] == f for r in runs)]
        for m in subs:
            row(f"&nbsp;&nbsp;{m}", [r for r in runs if r["method"] == m])
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--results", nargs="+", required=True)
    c.add_argument("--targets-ens", required=True)
    c.add_argument("--out", required=True)
    m = sub.add_parser("metrics")
    m.add_argument("--eval-eps", required=True)
    m.add_argument("--eval-ens", required=True)
    m.add_argument("--targets-ens", required=True)
    m.add_argument("--kl-n", type=int, default=4096)
    m.add_argument("--out", required=True)
    a = p.parse_args()
    {"collect": cmd_collect, "metrics": cmd_metrics}[a.cmd](a)


if __name__ == "__main__":
    main()
