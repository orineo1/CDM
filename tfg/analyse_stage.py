"""Paired analysis of an EXP-021/022 stage: CIs, permutation tests, call counts."""
import argparse, json, random, statistics as st, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from tfg.selection import PENALTY   # noqa: E402


def _pen(r):
    v = r["L2"]
    if r["diverged"] or v != v or v == float("inf"):
        return PENALTY
    return min(v, PENALTY)


def paired(a, b, B=20000, P=20000, seed=0):
    """Positive difference => b better (lower penalised L2). Paired by restart."""
    ra = {r["restart"]: r for r in a["runs"]}
    rb = {r["restart"]: r for r in b["runs"]}
    common = sorted(set(ra) & set(rb))
    diff = [_pen(ra[i]) - _pen(rb[i]) for i in common]
    n = len(diff)
    m = st.mean(diff)
    se = st.stdev(diff) / n ** 0.5 if n > 1 else float("nan")
    random.seed(seed)
    boots = sorted(st.mean([diff[random.randrange(n)] for _ in range(n)]) for _ in range(B))
    random.seed(seed + 1)
    cnt = sum(1 for _ in range(P)
              if abs(st.mean([d if random.random() < .5 else -d for d in diff])) >= abs(m))
    return {"n_pairs": n, "mean_diff": m, "se": se,
            "median_diff": st.median(diff),
            "ci95": [boots[int(.025 * B)], boots[int(.975 * B)]],
            "wins_for_b": sum(1 for v in diff if v > 0),
            "perm_p": (cnt + 1) / (P + 1)}


def load(path):
    d = json.load(open(path))
    return {(x["n_max"], x["spatial"], x["temporal"]): x for x in d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--label", default="stage")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    M = load(a.input)

    rows = []
    print(f"=== {a.label}: per-arm summary "
          f"(score = failure-penalised mean L2; calls averaged over completed runs) ===")
    print(f"{'n':>4} {'spatial':<10}{'temporal':<8}{'score':>8}{'L2 mean':>9}{'L2 med':>9}"
          f"{'succ':>6}{'div':>5}{'calls':>8}")
    for k in sorted(M, key=lambda k: (k[0], k[1], k[2])):
        x = M[k]; s = x["summary"]
        print(f"{k[0]:>4} {k[1]:<10}{k[2]:<8}{x['score']:>8.4f}{s['L2_mean']:>9.4f}"
              f"{s['L2_median']:>9.4f}{s['success_rate']*100:>5.0f}%{s['diverged']:>5}"
              f"{s['conditional_calls_per_run']:>8.0f}")
        rows.append({"key": list(k), "score": x["score"],
                     **{kk: s.get(kk) for kk in
                        ("L2_mean", "L2_median", "success_rate", "diverged",
                         "conditional_calls_per_run", "n_completed",
                         "restarts", "restart_offset")}})

    print(f"\n=== {a.label}: paired comparisons (positive => second arm better) ===")
    comps = []
    for nm in (4, 8, 16, 32):
        for sp in ("no_lgd", "lgd_high"):
            for t2 in ("adam", "lambda"):
                if (nm, sp, "none") in M and (nm, sp, t2) in M:
                    r = paired(M[(nm, sp, "none")], M[(nm, sp, t2)])
                    comps.append({"n_max": nm, "spatial": sp,
                                  "comparison": f"none vs {t2}", **r})
        # H3: no-LGD + Adam  vs  full-LGD without momentum (3x calls)
        if (nm, "lgd_high", "none") in M and (nm, "no_lgd", "adam") in M:
            r = paired(M[(nm, "lgd_high", "none")], M[(nm, "no_lgd", "adam")])
            comps.append({"n_max": nm, "spatial": "H3",
                          "comparison": "full-LGD/none vs no-LGD/Adam (1/3 cost)", **r})
    print(f"{'n':>4} {'spatial':<10}{'comparison':<44}{'mean':>9}{'median':>9}"
          f"{'95% CI':>22}{'wins':>9}{'p':>8}")
    for c in comps:
        ci = f"[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]"
        print(f"{c['n_max']:>4} {c['spatial']:<10}{c['comparison']:<44}"
              f"{c['mean_diff']:>+9.4f}{c['median_diff']:>+9.4f}{ci:>22}"
              f"{c['wins_for_b']:>5}/{c['n_pairs']:<3}{c['perm_p']:>8.4f}")

    if a.output:
        Path(a.output).write_text(json.dumps(
            {"label": a.label, "arms": rows, "comparisons": comps}, indent=2))
        print("\nwritten", a.output)


if __name__ == "__main__":
    main()
