"""Checkpoint 1 entry point: emit the equivalence matrix as machine-readable data.

A separate entry point on purpose: ``oz/run_optimization.py`` has uncommitted
local modifications and must not be touched or staged.

Usage (from the inner repository root):

    python -m tfg.run_checkpoint1 --output tfg_results/checkpoint1.json

Emits, per cell: the resolved configuration, the maximum absolute discrepancy
over every traced intermediate tensor, the first diverging key if any, the
tape-key set difference, call counts, and runtime.  Run-level metadata (git
commit, dirty state, seed, device, dtype, torch version) is recorded once.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))

from tfg.config import NScheduleConfig, TFGConfig          # noqa: E402
from tfg.engine import GeneralizedTFG                       # noqa: E402
from tfg.noise_tape import NoiseTape, compare_access        # noqa: E402
from tfg.reference_tfg import run_reference_tfg             # noqa: E402
from tfg.schedule import DiffusionSchedule, constant_vector  # noqa: E402
from tfg.trace import Tracer, compare_traces                # noqa: E402

SHAPE = (1, 2)


def git_metadata():
    def run(*args):
        try:
            return subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": run("git", "status", "--porcelain"),
    }


def run_cell(N_recur, N_iter, gamma_bar, rho_scalar, mu_scalar, n_mc, T, seed,
             use_neutral_n_schedule=False):
    import types
    sys.modules.setdefault("ot", types.ModuleType("ot"))
    from conftest import ToyDenoiser, make_quadratic_log_f

    schedule = DiffusionSchedule(T=T)
    denoiser = ToyDenoiser(d=SHAPE[1], T=T, seed=0, schedule=schedule)
    eps_theta = lambda x, t: denoiser(x, t)          # noqa: E731
    base_log_f = make_quadratic_log_f([1.5, -0.75], scale=0.6)

    def log_f(x, n_t=None, eta_keys=None):
        return base_log_f(x)

    rho = constant_vector(rho_scalar, T)
    mu = constant_vector(mu_scalar, T)

    t0 = time.perf_counter()
    tape_ref, tr_ref = NoiseTape(seed=seed), Tracer()
    x_ref = run_reference_tfg(eps_theta, base_log_f, schedule, tape_ref, SHAPE,
                              N_recur=N_recur, N_iter=N_iter, rho=rho, mu=mu,
                              gamma_bar=gamma_bar, n_mc=n_mc, trace=tr_ref)
    t_ref = time.perf_counter() - t0

    cfg = TFGConfig(T=T, N_recur=N_recur, N_iter=N_iter, gamma_bar=gamma_bar,
                    rho_scalar=rho_scalar, mu_scalar=mu_scalar, n_mc=n_mc)
    if use_neutral_n_schedule:
        cfg.n_schedule = NScheduleConfig(enabled=True, type="constant", n_max=1)

    t0 = time.perf_counter()
    tape_eng, tr_eng = NoiseTape(seed=seed), Tracer()
    engine = GeneralizedTFG(eps_theta, log_f, schedule, tape_eng, cfg)
    x_eng = engine.run(SHAPE, trace=tr_eng)
    t_eng = time.perf_counter() - t0

    ok, report = compare_traces(tr_ref, tr_eng, atol=0.0)
    only_ref, only_eng = compare_access(tape_ref, tape_eng)
    final_err = (x_ref - x_eng).abs().max().item()

    return {
        "config": cfg.resolved(),
        "seed": seed,
        "max_abs_err_all_intermediates": report["max_abs_err"],
        "final_x0_abs_err": final_err,
        "first_divergence": (None if report["first_divergence"] is None
                             else str(report["first_divergence"])),
        "n_traced_keys_compared": report["n_keys_compared"],
        "tape_keys_only_reference": [str(k) for k in only_ref],
        "tape_keys_only_engine": [str(k) for k in only_eng],
        "worst_keys": [[str(k), v] for k, v in report["per_key"][:3]],
        "counts": engine.counter.as_dict(),
        "runtime_s": {"reference": t_ref, "engine": t_eng},
        "final_x0_magnitude": x_ref.abs().max().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tfg_results/checkpoint1.json")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--T", type=int, default=8)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)

    modes = {"rho_only": (0.9, 0.0), "mu_only": (0.0, 0.35), "both": (0.9, 0.35)}
    cells = []
    for N_recur in (1, 2):
        for N_iter in (0, 1, 4):
            for gamma_bar in (0.0, 0.35):
                for mode, (rho_s, mu_s) in modes.items():
                    if N_iter == 0 and mode in ("mu_only", "both"):
                        continue
                    cells.append(dict(N_recur=N_recur, N_iter=N_iter,
                                      gamma_bar=gamma_bar, mode=mode,
                                      rho_scalar=rho_s, mu_scalar=mu_s,
                                      n_mc=3 if gamma_bar else 1, T=args.T))
    # extra non-grid cells
    cells.append(dict(N_recur=3, N_iter=2, gamma_bar=0.4, mode="both",
                      rho_scalar=0.9, mu_scalar=0.35, n_mc=2, T=args.T))
    cells.append(dict(N_recur=2, N_iter=1, gamma_bar=0.3, mode="both",
                      rho_scalar=0.9, mu_scalar=0.35, n_mc=2, T=2))

    results = []
    worst = 0.0
    for c in cells:
        out = run_cell(c["N_recur"], c["N_iter"], c["gamma_bar"],
                       c["rho_scalar"], c["mu_scalar"], c["n_mc"], c["T"],
                       args.seed)
        out["cell"] = c
        results.append(out)
        worst = max(worst, out["max_abs_err_all_intermediates"])
        print(f"N_recur={c['N_recur']} N_iter={c['N_iter']} "
              f"gamma={c['gamma_bar']} mode={c['mode']:<8} T={c['T']:<3} "
              f"max_abs_err={out['max_abs_err_all_intermediates']:.3e} "
              f"keys={out['n_traced_keys_compared']}")

    # Test B cells, with the composite exp(-beta*MMD^2) predictor.
    from test_equivalence import CompositePredictor, _make_S_G, _setup
    import types
    sys.modules.setdefault("ot", types.ModuleType("ot"))
    testb = []
    for N_recur, N_iter, gamma_bar in [(1, 0, 0.0), (1, 1, 0.0), (2, 1, 0.0),
                                       (1, 2, 0.3), (2, 2, 0.3)]:
        schedule, eps_theta, _, seed = _setup(args.T, args.seed)
        predictor = CompositePredictor(_make_S_G(), beta=1.0)
        n_mc = 2 if gamma_bar else 1
        rho, mu = constant_vector(0.5, args.T), constant_vector(0.2, args.T)

        tr_ref = Tracer()
        run_reference_tfg(eps_theta, predictor, schedule, NoiseTape(seed=seed),
                          SHAPE, N_recur=N_recur, N_iter=N_iter, rho=rho, mu=mu,
                          gamma_bar=gamma_bar, n_mc=n_mc, trace=tr_ref)
        cfg = TFGConfig(T=args.T, N_recur=N_recur, N_iter=N_iter,
                        gamma_bar=gamma_bar, rho_scalar=0.5, mu_scalar=0.2,
                        n_mc=n_mc)
        cfg.n_schedule = NScheduleConfig(enabled=True, type="constant", n_max=1)
        tr_eng = Tracer()
        eng = GeneralizedTFG(eps_theta, predictor, schedule,
                             NoiseTape(seed=seed), cfg)
        eng.run(SHAPE, trace=tr_eng)
        ok, rep = compare_traces(tr_ref, tr_eng, atol=0.0)
        worst = max(worst, rep["max_abs_err"])
        testb.append({"cell": {"N_recur": N_recur, "N_iter": N_iter,
                               "gamma_bar": gamma_bar, "n_max": 1},
                      "max_abs_err_all_intermediates": rep["max_abs_err"],
                      "n_traced_keys_compared": rep["n_keys_compared"],
                      "counts": eng.counter.as_dict()})
        print(f"[B] N_recur={N_recur} N_iter={N_iter} gamma={gamma_bar} "
              f"max_abs_err={rep['max_abs_err']:.3e}")

    payload = {
        "checkpoint": 1,
        "metadata": {
            **git_metadata(),
            "seed": args.seed,
            "device": "cpu",
            "dtype": str(torch.get_default_dtype()),
            "torch_version": torch.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "provenance_caveat": (
            "No TFG implementation existed in this repository before this work "
            "(only the LGD subspace in Optimization.py:88). tfg/reference_tfg.py "
            "is NEW code transcribed from Algorithm 1, so these results compare "
            "two new implementations, not a new engine against established code."
        ),
        "test_A": results,
        "test_B": testb,
        "worst_max_abs_err": worst,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nworst max_abs_err across all cells: {worst:.3e}")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
