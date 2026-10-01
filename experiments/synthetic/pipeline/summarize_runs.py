"""Summarize search_mog result files by variant: per-seed selected TV, share of good trajectories, time.

Files are grouped by name prefix: <variant>_s<seed>.json. For each variant it prints the exact TV of the
returned design per seed, its mean and maximum, how many seeds end below TV 0.1, the share of trajectories
below 0.1 and 0.2, minutes per run, the best TV available among each run's trajectories, and the Spearman
correlation between the final loss estimate and TV (how well selection can work). TVs are cached in
<runs-dir>/_tvcache.json.

  python experiments/synthetic/pipeline/summarize_runs.py --runs-dir runs/night/x10_y1 \
      --data runs/table1v2/x10_y1/data.pt [--variants D3_ref100_n3000 E2_zoref_n3000]
"""
import argparse, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_table1 import _load_mixture, scale_free_distances   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--data", required=True, help="data.pt of the scenario (mixture parameters and x*)")
    p.add_argument("--variants", nargs="*", help="name prefixes; default: every <prefix>_s<seed>.json in --runs-dir")
    a = p.parse_args()
    data = _load_mixture(a.data)
    d = a.runs_dir
    pats = a.variants or sorted({re.sub(r"_s\d+\.json$", "", os.path.basename(f)) for f in glob.glob(f"{d}/*_s*.json")})
    cache_f = f"{d}/_tvcache.json"
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}

    def tv(x, n=5000):
        k = json.dumps(np.round(np.array(x).reshape(-1), 6).tolist())
        if k not in cache:
            cache[k] = scale_free_distances(data, np.array(x).reshape(-1).tolist(), n_mc=n, seed=0)["tv"]
        return cache[k]

    print(f"{'variant':28s} {'n':>2s} {'selected TV per seed':40s} {'mean':>5s} {'max':>5s} {'<0.1':>5s} "
          f"{'traj<.1':>7s} {'traj<.2':>7s} {'min/run':>7s}  best available per seed; rank corr.")
    for v in pats:
        sel, t01, t02, tm, best, rho = [], [], [], [], [], []
        for f in sorted(glob.glob(f"{d}/{v}_s*.json")):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            for att in r["attempts"]:
                runs = att.get("restarts", [att])
                ts = np.array([tv(q["x_hat"]) for q in runs])
                sel.append(tv(att["x_hat"], 20000))
                t01.append(np.mean(ts < 0.1)); t02.append(np.mean(ts < 0.2))
                tm.append(att.get("time_s", float("nan")) / 60)
                best.append(ts.min())
                L = [q["final_loss"] for q in runs]
                if len(ts) > 2:
                    rho.append(float(np.corrcoef(np.argsort(np.argsort(L)), np.argsort(np.argsort(ts)))[0, 1]))
        if not sel:
            continue
        print(f"{v:28s} {len(sel):2d} {str(np.round(sel, 3).tolist()):40s} {np.mean(sel):5.3f} {np.max(sel):5.3f} "
              f"{int(np.sum(np.array(sel) < 0.1)):2d}/{len(sel):<2d} {np.mean(t01):7.3f} {np.mean(t02):7.3f} "
              f"{np.nanmean(tm):7.1f}  {np.round(best, 3).tolist()}; {np.round(np.nanmean(rho), 2) if rho else 'n/a'}")
    json.dump(cache, open(cache_f, "w"))


if __name__ == "__main__":
    main()
