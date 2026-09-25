"""EXP-010 -- audit of EXP-009. Reads raw artifacts only; changes nothing."""
import json, math, random, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "research_wiki" / "runs"
MAIN = W / "run_20260820_0240_curriculum_main" / "raw" / "curriculum_main.json"
ROB = W / "run_20260820_0245_curriculum_robustness" / "raw" / "curriculum_robustness.json"


def load(f, seed, method, lam=0.0):
    d = json.load(open(f))
    return next(r for r in d["results"] if r["summary"]["seed"] == seed
                and r["summary"]["method"] == method
                and r["summary"]["lambda"] == lam)


def paired(f, seed, a="adaptive_n", b="curriculum", lam=0.0, B=20000):
    ra, rb = load(f, seed, a, lam)["runs"], load(f, seed, b, lam)["runs"]
    assert [x["restart"] for x in ra] == [x["restart"] for x in rb]
    diff = [x["L2"] - y["L2"] for x, y in zip(ra, rb)]   # >0 => b (curriculum) better
    n = len(diff); m = st.mean(diff); sd = st.stdev(diff); se = sd / n ** 0.5
    random.seed(0)
    boots = sorted(st.mean([diff[random.randrange(n)] for _ in range(n)]) for _ in range(B))
    # paired permutation test on the mean difference
    random.seed(1)
    P = 20000
    cnt = sum(1 for _ in range(P)
              if abs(st.mean([d if random.random() < .5 else -d for d in diff])) >= abs(m))
    return {"seed": seed, "n": n, "mean_diff": m, "se": se, "t": m / se,
            "median_diff": st.median(diff),
            "ci95": [boots[int(.025 * B)], boots[int(.975 * B)]],
            "wins_for_curriculum": sum(1 for v in diff if v > 0),
            "perm_p_two_sided": (cnt + 1) / (P + 1),
            "mean_L2_adaptive_n": st.mean([x["L2"] for x in ra]),
            "mean_L2_curriculum": st.mean([x["L2"] for x in rb])}


def schedule_audit():
    r = load(MAIN, 20240401, "curriculum")
    traj = r["runs"][0]["trajectory"]
    ns = [s["n"] for s in traj]; Ks = [s["K"] for s in traj]
    same = sum(1 for a, b in zip(Ks, Ks[1:]) if a == b)
    closed = [1 + math.floor(249 * (100 - t) / 99) for t in range(99, 0, -1)]
    return {"steps_executed": len(traj), "t_first": traj[0]["t"], "t_last": traj[-1]["t"],
            "K_equals_n_everywhere": ns == Ks,
            "n_min": min(ns), "n_max": max(ns),
            "steps_with_K_eq_1": sum(1 for v in Ks if v == 1),
            "steps_with_K_le_3": sum(1 for v in Ks if v <= 3),
            "sum_n_t": sum(ns),
            "summary_conditional_calls_per_run": r["summary"]["conditional_calls_per_run"],
            "closed_form_matches": closed == ns,
            "adjacent_pairs_same_K": same,
            "adjacent_pairs_total": len(Ks) - 1,
            "fraction_adjacent_same_K": same / (len(Ks) - 1),
            "n_t_sequence": ns}


if __name__ == "__main__":
    out = {"paired": [paired(MAIN, 20240401), paired(ROB, 20240402), paired(ROB, 20240403)],
           "schedule": schedule_audit()}
    print(json.dumps(out["paired"], indent=2)[:1200])
    s = out["schedule"]
    print(f"\nsteps={s['steps_executed']} K==n={s['K_equals_n_everywhere']} "
          f"n in [{s['n_min']},{s['n_max']}] sum={s['sum_n_t']} "
          f"K=1 steps={s['steps_with_K_eq_1']} "
          f"adjacent same K={s['adjacent_pairs_same_K']}/{s['adjacent_pairs_total']}")
    p = ROOT / "tfg_results" / "exp010_audit.json"
    p.write_text(json.dumps(out, indent=2))
    print("written", p)
