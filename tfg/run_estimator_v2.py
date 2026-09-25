"""Corrected n_t and temporal-lambda estimator study.

WHAT CHANGED FROM THE INVALID PILOT
-----------------------------------
The pilot varied the conditional sample count AND the target sample count
together (it redrew ``n_t`` target samples each call, copying
``Optimization.py:120``). That is not the proposed algorithm: ``n_t`` is meant
to control conditional-generator calls only. Here:

  * ONE fixed target set S_G of 250 samples (the original experiment's
    ``nsamples_in_optim_for_mmd``) is drawn once and reused for every x, every
    n_t, every replication, and every lambda comparison.
  * only the conditional sample count varies.
  * at n_t = 1 a single conditional sample is compared against the FULL fixed
    target set, so the adaptive bandwidth no longer degenerates.

All pilot adaptive-bandwidth numbers and all pilot lambda numbers are void.

TWO REPORTED QUANTITIES, KEPT SEPARATE
--------------------------------------
With an adaptive bandwidth the effective objective changes with n_t, so
"bias against a single reference" conflates two things. We report:

  1. stochastic spread around that n_t's OWN expected gradient E[g_n]
     (variance, MSE about E[g_n], robust spread);
  2. deviation of E[g_n] from a COMMON fixed-bandwidth high-sample reference,
     which is the objective-drift term.

A high-n adaptive-bandwidth gradient is NOT treated as the same objective as a
small-n one.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tfg._compat import ensure_ot_stub          # noqa: E402

ensure_ot_stub()

import dist_utils                                # noqa: E402
from LossFunctions import MMDLoss, RBF           # noqa: E402
from tfg import oracle                           # noqa: E402
from tfg.train_cm import PAPER_TS, SEEDS, load_one   # noqa: E402

N_GRID = [1, 2, 4, 8, 16, 32, 64]
TARGET_SET_SIZE = 250          # nsamples_in_optim_for_mmd, notebook cell 9
N_REF = 2048


def key_seed(*parts):
    return int.from_bytes(
        hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(),
        "big") % (2 ** 31 - 1)


def git_metadata():
    def run(*a):
        try:
            return subprocess.run(a, cwd=str(_ROOT), capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("git", "status", "--porcelain")}


def make_fixed_target_set(params, size=TARGET_SET_SIZE, seed=987654):
    """Draw S_G ONCE. Reused everywhere."""
    torch.manual_seed(seed)
    tc = torch.stack([v.float() for v in params["target_variances"]])
    return dist_utils.generate_mog_samples_not_differentiable(
        size, params["target_means"].float(), tc,
        params["target_weights"].float()).detach()


def pick_fixed_bandwidth(model, params, S_G, x_ref=-5.0, n=4096, seed=555):
    """Select the fixed bandwidth ONCE from a large reference sample.

    Uses the repository's own rule (mean off-diagonal squared distance over the
    pooled set), applied to a large draw so the value is stable, then frozen.
    """
    torch.manual_seed(seed)
    cond = torch.full((n, 1), float(x_ref), dtype=torch.float32)
    with torch.no_grad():
        y, _, _ = model.sample(nsamples=n, condition_x=cond, device="cpu",
                               ts=PAPER_TS)
    pooled = torch.vstack([y, S_G])
    d2 = torch.cdist(pooled, pooled, p=2) ** 2
    m = pooled.shape[0]
    return float(d2.sum() / (m ** 2 - m))


def one_gradient(model, x_value, n, S_G, bandwidth, seed):
    """One realisation of the MLGD-F gradient with a FIXED target set."""
    torch.manual_seed(seed)
    x = torch.tensor([float(x_value)], dtype=torch.float32, requires_grad=True)
    cond = x.reshape(1, -1).repeat(n, 1)
    y, _, _ = model.sample(nsamples=n, condition_x=cond, device="cpu",
                           ts=PAPER_TS)
    loss = MMDLoss(kernel=RBF(bandwidth=bandwidth, device="cpu"),
                   device="cpu")(y, S_G)
    g, = torch.autograd.grad(loss, x, allow_unused=True)
    return float(loss.detach()), (0.0 if g is None else float(g[0]))


def replicate(model, x_value, n, reps, S_G, bandwidth, role):
    grads = []
    for r in range(reps):
        _, g = one_gradient(model, x_value, n, S_G, bandwidth,
                            key_seed(role, x_value, n, r))
        grads.append(g)
    return torch.tensor(grads, dtype=torch.float64)


def describe(g, g_ref=None):
    """Classical statistics plus the robust ones, never instead of them."""
    reps = g.numel()
    mean = float(g.mean())
    se = float(g.std(unbiased=True) / reps ** 0.5)
    var = float(g.var(unbiased=True))
    srt = g.sort().values
    k = max(1, int(0.05 * reps))
    out = {
        "reps": reps,
        "mean": mean, "se_mean": se,
        "variance": var,
        "se_variance": float(var * (2.0 / (reps - 1)) ** 0.5),
        "mse_about_own_mean": float(((g - mean) ** 2).mean()),
        "median": float(g.median()),
        "iqr": float(g.quantile(0.75) - g.quantile(0.25)),
        "trimmed_mean_5pct": float(srt[k:reps - k].mean()),
        "q01": float(g.quantile(0.01)), "q99": float(g.quantile(0.99)),
        "max_abs": float(g.abs().max()),
        "kurtosis": float(((g - mean) ** 4).mean() / max(1e-300, g.var(unbiased=False) ** 2)),
    }
    if g_ref is not None:
        sq = (g - g_ref) ** 2
        out.update({
            "g_ref": g_ref,
            "bias_vs_ref": mean - g_ref,
            "se_bias": se,
            "mse_vs_ref": float(sq.mean()),
            "se_mse": float(sq.std(unbiased=True) / reps ** 0.5),
            "sign_agreement": float(((g * g_ref) > 0).double().mean()),
        })
    return out


def reference(model, x_value, S_G, bandwidth, reps, role):
    g = replicate(model, x_value, N_REF, reps, S_G, bandwidth, role)
    mean = float(g.mean())
    se = float(g.std(unbiased=True) / reps ** 0.5)
    return {"g_ref": mean, "se": se, "snr": abs(mean) / se if se > 0 else float("inf"),
            "reps": reps, "n_ref": N_REF}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tfg_results/estimator_v2.json")
    ap.add_argument("--ckpt-dir", default="tfg_results/cm_checkpoints")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--reps-max", type=int, default=10000)
    ap.add_argument("--ref-reps", type=int, default=300)
    ap.add_argument("--probes", type=float, nargs="*", default=[-6.0, -5.5, -3.0])
    ap.add_argument("--stationary-probe", type=float, default=-4.648)
    args = ap.parse_args()

    params = oracle.load_params()
    S_G = make_fixed_target_set(params)
    print(f"fixed target set: {tuple(S_G.shape)}  mean={float(S_G.mean()):+.4f} "
          f"std={float(S_G.std()):.4f}  (drawn once, reused everywhere)")

    model, rec = load_one(SEEDS[0], args.ckpt_dir)
    bw_fixed = pick_fixed_bandwidth(model, params, S_G)
    print(f"fixed bandwidth selected once from n=4096 reference sample: {bw_fixed:.6f}\n")

    payload = {"metadata": {**git_metadata(), "torch": torch.__version__,
                            "platform": platform.platform(),
                            "target_set_size": TARGET_SET_SIZE,
                            "fixed_bandwidth": bw_fixed,
                            "n_grid": N_GRID, "n_ref": N_REF,
                            "paper_ts": PAPER_TS,
                            "provenance": "RE-TRAINED to paper Table 3 config"},
               "probes": {}, "lambda": [], "robustness": {}}

    # ---- references, both settings, all probes ----
    refs = {}
    print("=== references (n=2048) and signal-to-noise ===")
    for setting, bw in (("adaptive", None), ("fixed", bw_fixed)):
        for xv in args.probes + [args.stationary_probe]:
            r = reference(model, xv, S_G, bw, args.ref_reps, f"ref-{setting}")
            refs[(setting, xv)] = r
            flag = "OK" if r["snr"] > 10 else "LOW-SNR (absolute errors only)"
            print(f"  {setting:<9} x={xv:>7.3f}  g_ref={r['g_ref']:+.6e} "
                  f"+-{r['se']:.2e}  |g|/SE={r['snr']:>7.1f}  {flag}")

    # ---- n sweep ----
    for setting, bw in (("adaptive", None), ("fixed", bw_fixed)):
        for xv in args.probes + [args.stationary_probe]:
            ref = refs[(setting, xv)]
            usable = ref["snr"] > 10
            key = f"{setting}@x={xv}"
            print(f"\n=== {key}   ref={ref['g_ref']:+.5e} "
                  f"({'relative stats OK' if usable else 'LOW SNR: absolute only'}) ===")
            print(f"  {'n':>3} {'E[g_n]':>12} {'SE':>9} {'var':>10} {'MSE/ref':>10} "
                  f"{'median':>10} {'IQR':>9} {'trim5%':>10} {'kurt':>8} {'sign':>5}")
            rows = []
            for n in N_GRID:
                g = replicate(model, xv, n, args.reps, S_G, bw, f"sweep-{setting}")
                st = describe(g, ref["g_ref"] if usable else None)
                st["n"] = n
                st["objective_drift_vs_common_fixed_ref"] = (
                    st["mean"] - refs[("fixed", xv)]["g_ref"])
                rows.append(st)
                print(f"  {n:>3} {st['mean']:>12.4e} {st['se_mean']:>9.2e} "
                      f"{st['variance']:>10.3e} "
                      f"{st.get('mse_vs_ref', float('nan')):>10.3e} "
                      f"{st['median']:>10.3e} {st['iqr']:>9.2e} "
                      f"{st['trimmed_mean_5pct']:>10.3e} {st['kurtosis']:>8.1f} "
                      f"{st.get('sign_agreement', float('nan')):>5.2f}")
            payload["probes"][key] = {"reference": ref, "usable": usable,
                                      "sweep": rows}

    # ---- replication stability ----
    print("\n=== replication stability (adaptive, x=-6.0, n=8) ===")
    stab = []
    for reps in (args.reps, args.reps_max):
        g = replicate(model, -6.0, 8, reps, S_G, None, "sweep-adaptive")
        st = describe(g, refs[("adaptive", -6.0)]["g_ref"])
        stab.append({"reps": reps, **st})
        print(f"  reps={reps:<6} mean={st['mean']:+.5e} var={st['variance']:.4e} "
              f"MSE={st['mse_vs_ref']:.4e} median={st['median']:+.5e}")
    payload["stability"] = stab

    # ---- temporal aggregation ----
    #
    # Prediction  lambda* = V_t / (V_t + V_{t+1} + D_t^2)  minimises
    #   MSE(l) = E|| (1-l) ghat_t + l ghat_{t+1} - g_t^ref ||^2
    #          = (1-l)^2 V_t + l^2 V_{t+1} + l^2 D_t^2
    # under two assumptions, both stated rather than assumed silently:
    #   (A1) ghat_t and ghat_{t+1} are INDEPENDENT (guaranteed here: the two
    #        timesteps use disjoint tape keys, matching the algorithm's
    #        independent conditional randomness);
    #   (A2) each ghat is UNBIASED for its own step's reference. With an
    #        adaptive bandwidth (A2) holds only approximately, so a gap between
    #        empirical and predicted lambda is expected and is itself a result.
    print("\n=== temporal aggregation (lambda) ===")
    x_t = -6.0
    partners = [("low", -5.98), ("medium", -5.75), ("high", -5.0)]
    for setting, bw in (("adaptive", None), ("fixed", bw_fixed)):
        ref_t = refs[(setting, x_t)]
        for label, x_n in partners:
            ref_n = reference(model, x_n, S_G, bw, args.ref_reps,
                              f"ref-partner-{setting}")
            D2 = (ref_t["g_ref"] - ref_n["g_ref"]) ** 2
            for n in (1, 4, 32):
                gt = replicate(model, x_t, n, args.reps, S_G, bw, f"lam-t-{setting}")
                gn = replicate(model, x_n, n, args.reps, S_G, bw,
                               f"lam-next-{setting}-{label}")
                V_t = float(gt.var(unbiased=True))
                V_n = float(gn.var(unbiased=True))
                lam_pred = V_t / (V_t + V_n + D2)

                fine = torch.linspace(0, 1, 201, dtype=torch.float64)
                mses = torch.tensor([float((((1 - l) * gt + l * gn)
                                            - ref_t["g_ref"]).pow(2).mean())
                                     for l in fine])
                lam_emp = float(fine[int(mses.argmin())])
                mse0 = float((gt - ref_t["g_ref"]).pow(2).mean())
                rec = {"setting": setting, "drift_label": label, "n": n,
                       "x_t": x_t, "x_next": x_n,
                       "g_ref_t": ref_t["g_ref"], "g_ref_next": ref_n["g_ref"],
                       "snr_t": ref_t["snr"], "snr_next": ref_n["snr"],
                       "D2": D2, "V_t": V_t, "V_next": V_n,
                       "lambda_empirical": lam_emp,
                       "lambda_predicted": lam_pred,
                       "mse_lambda0": mse0, "mse_best": float(mses.min()),
                       "gain": 1 - float(mses.min()) / mse0,
                       "curve": [{"lambda": float(l), "mse": float(m)}
                                 for l, m in zip(fine[::20], mses[::20])]}
                payload["lambda"].append(rec)
                print(f"  {setting:<9} drift={label:<7} n={n:<3} "
                      f"D2={D2:.3e} V_t={V_t:.3e} V_n={V_n:.3e}  "
                      f"lam_emp={lam_emp:.3f} lam_pred={lam_pred:.3f}  "
                      f"gain={rec['gain']:+.1%}")

    # ---- robustness cells on the other two models ----
    print("\n=== robustness cells (other seeds, adaptive, x=-6.0) ===")
    for seed in SEEDS[1:]:
        m2, _ = load_one(seed, args.ckpt_dir)
        r2 = reference(m2, -6.0, S_G, None, args.ref_reps, f"ref-rob-{seed}")
        cells = []
        for n in (4, 32):
            g = replicate(m2, -6.0, n, args.reps, S_G, None, f"rob-{seed}")
            st = describe(g, r2["g_ref"]); st["n"] = n
            cells.append(st)
            print(f"  seed={seed} n={n:<3} E[g]={st['mean']:+.4e} "
                  f"var={st['variance']:.3e} MSE={st['mse_vs_ref']:.3e} "
                  f"sign={st['sign_agreement']:.2f}  (ref={r2['g_ref']:+.4e}, "
                  f"snr={r2['snr']:.1f})")
        payload["robustness"][str(seed)] = {"reference": r2, "cells": cells}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {k: (v if k != "probes" else v) for k, v in payload.items()},
        indent=2, default=str))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
