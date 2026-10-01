"""Per-trajectory diagnostics of a search_mog result file (a scenario with the mixture known).

For every trajectory (restart) of every attempt: exact TV of P(Y | X = x_hat) to the target after and
(when saved) before refinement, the mixture responsibilities pi_k(x_hat), and the final loss estimate
used for selection. Summaries: TV quantiles, share of trajectories below 0.2 / 0.5, Spearman
correlation between the final loss estimate and TV, the TV of the loss-selected trajectory versus the
best TV available in the same attempt, and best-of-k subsampling (k = 10) from larger budgets.

  python experiments/synthetic/pipeline/diag_trajectories.py --runs-dir runs/table1v2/x10_y1 \
      --files MLGD_r100.json MLGD-X_r100.json --out results/diag_x10_y1.md
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from make_table1 import _load_mixture, scale_free_distances          # noqa: E402
from cdm.utils.dist import compute_alpha                               # noqa: E402


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--k", type=int, default=10, help="best-of-k subsampling from larger budgets")
    p.add_argument("--n-mc", type=int, default=5000, help="Monte Carlo draws per TV")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    data = _load_mixture(os.path.join(a.runs_dir, "data.pt"))
    raw = torch.load(os.path.join(a.runs_dir, "data.pt"), map_location="cpu", weights_only=False)
    mu, Sig, al = raw["mu_list"], raw["Sigma_list"], raw["alpha"]
    xs = raw.get("x_star")
    pi_star = compute_alpha(mu, Sig, al, torch.as_tensor(xs).float().view(-1)).view(-1).tolist() if xs is not None else None
    lines = [f"# Trajectory diagnostics: {a.runs_dir}", "",
             f"responsibilities at x*: {np.round(pi_star, 3).tolist() if pi_star else 'n/a'}", ""]
    out_json = {}
    rng = np.random.default_rng(0)
    for f in a.files:
        r = json.load(open(os.path.join(a.runs_dir, f)))
        rows = []
        for att in r["attempts"]:
            for q in att.get("restarts", [att]):
                x = q["x_hat"]
                tv = scale_free_distances(data, x, n_mc=a.n_mc, seed=len(rows))["tv"]
                xb = (q.get("refine") or {}).get("x_before")
                tvb = scale_free_distances(data, np.array(xb).reshape(-1).tolist(), n_mc=a.n_mc, seed=len(rows))["tv"] if xb is not None else float("nan")
                pi = compute_alpha(mu, Sig, al, torch.as_tensor(x).float().view(-1)).view(-1).tolist()
                rows.append({"attempt": att["attempt"], "tv": tv, "tv_before": tvb, "loss": q["final_loss"], "pi": pi,
                             "k": int(np.argmax(pi)), "pmax": float(max(pi))})
        tv = np.array([q["tv"] for q in rows]); tvb = np.array([q["tv_before"] for q in rows]); loss = np.array([q["loss"] for q in rows])
        att_ids = sorted(set(q["attempt"] for q in rows))
        sel, best = [], []
        for t in att_ids:
            idx = [i for i, q in enumerate(rows) if q["attempt"] == t]
            sel.append(tv[idx][np.argmin(loss[idx])]); best.append(tv[idx].min())
        bok = []
        for _ in range(2000):
            idx = rng.choice(len(rows), size=min(a.k, len(rows)), replace=False)
            bok.append(tv[idx][np.argmin(loss[idx])])
        comp = {}
        for q in rows:
            c = comp.setdefault(q["k"], {"n": 0, "tv": []}); c["n"] += 1; c["tv"].append(q["tv"])
        s = {"n_traj": len(rows), "tv_q": np.round(np.quantile(tv, [0, .1, .25, .5, .75]), 3).tolist(),
             "frac_tv_lt_0.2": float((tv < 0.2).mean()), "frac_tv_lt_0.5": float((tv < 0.5).mean()),
             "tv_before_median": float(np.nanmedian(tvb)), "refine_improves_frac": float(np.nanmean(tv < tvb)) if np.isfinite(tvb).any() else None,
             "spearman_loss_tv": spearman(loss, tv),
             "selected_tv_per_attempt": np.round(sel, 3).tolist(), "best_tv_per_attempt": np.round(best, 3).tolist(),
             f"best_of_{a.k}_selected_tv_mean": float(np.mean(bok)), f"best_of_{a.k}_selected_tv_q": np.round(np.quantile(bok, [.1, .5, .9]), 3).tolist(),
             "by_component": {int(k): {"n": v["n"], "tv_median": round(float(np.median(v["tv"])), 3), "tv_min": round(float(np.min(v["tv"])), 3)}
                              for k, v in sorted(comp.items())}}
        out_json[f] = {"summary": s, "rows": rows}
        lines += [f"## {f} ({r.get('method')}, budget {r.get('budget')})", ""] + [f"- {k}: {v}" for k, v in s.items()] + [""]
        print(f, json.dumps(s))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(lines))
    json.dump(out_json, open(os.path.splitext(a.out)[0] + ".json", "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
