"""EXP-021 stage runner: tuning (offset 0) or held-out confirmation (offset 1000)."""
import argparse, json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from tfg.factorial_experiment import run_arm            # noqa: E402
from tfg.selection import score                          # noqa: E402
from tfg import oracle                                   # noqa: E402
from tfg.run_estimator_v2 import make_fixed_target_set   # noqa: E402
from tfg.train_cm import SEEDS, load_one                 # noqa: E402
from tfg.train_uncond import load as load_uncond         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--restarts", type=int, required=True)
ap.add_argument("--offset", type=int, required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--adam-rho", type=float, default=0.4)
a = ap.parse_args()

p = oracle.load_params(); S = make_fixed_target_set(p)
mc, _ = load_one(SEEDS[0], "tfg_results/cm_checkpoints")
mu, _ = load_uncond(out_dir="tfg_results/cm_checkpoints")

out = []
for nm in (4, 8, 16, 32):
    for sp in ("no_lgd", "lgd_high"):
        for tm in ("none", "adam", "lambda"):
            arm = run_arm(mc, mu, S, p, sp, tm, a.restarts, restart_offset=a.offset,
                          n_protocol="constant", n_max=nm, adam_rho=a.adam_rho)
            m = arm["summary"]
            out.append({"n_max": nm, "spatial": sp, "temporal": tm,
                        "score": score(arm["runs"]), "summary": m,
                        "runs": [{k: r[k] for k in
                                  ("restart", "L2", "abs_err_vs_xopt", "diverged",
                                   "conditional_calls", "seconds")} for r in arm["runs"]]})
            print(f"  n={nm:<3} {sp:<10}{tm:<7} score={out[-1]['score']:.4f} "
                  f"L2mean={m['L2_mean']:.4f} succ={m['success_rate']:.0%} "
                  f"calls={m['conditional_calls_per_run']:.0f}", flush=True)
            Path(a.output).write_text(json.dumps(out, indent=2))
print("DONE", len(out), "arms ->", a.output)
