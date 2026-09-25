"""n_t and temporal-lambda estimator study, using the trained consistency model.

The estimator under study is exactly the one MLGD-F uses (``Optimization.py:113-133``):

    x  ->  condition = x.repeat(n, 1)
       ->  y = model_cond.sample(n, condition)          [14 network applications]
       ->  L = MMDLoss(y, target_samples)               [repository RBF]
       ->  g = dL/dx

Two settings, per instruction:

  * ``adaptive``  -- the repository's MMD verbatim, including the
    sample-dependent bandwidth (``LossFunctions.py:15-17``) and target samples
    redrawn at the same count as the conditional samples, exactly as
    ``Optimization.py:120`` does. This is the main empirical setting.
  * ``fixed``     -- identical except the bandwidth is pinned, isolating the
    extra gradient noise the adaptive bandwidth introduces.

Reference gradients
-------------------
There is no closed form for the consistency model, so its population-MMD
gradient is approximated at high sample count and its stability is verified by
increasing BOTH the sample count and the number of independent replications.
Two references are built:

  ``g_ref_cm``       high-sample gradient through the consistency model.
                     Finite-sample error of small-n estimates is measured
                     against this.
  ``g_ref_analytic`` the same high-sample construction but drawing the
                     conditional samples from the EXACT analytic conditional.
                     The gap ``g_ref_cm - g_ref_analytic`` is model error.

Keeping both references in the same estimator form is deliberate: comparing a
sampled MMD gradient against a closed-form L2 gradient would measure the gap
between two different objectives, not estimator error.

Randomness
----------
Conditional samples are drawn independently at each timestep, matching the
proposed algorithm, but every draw is keyed deterministically by
``(role, x_index, n, replicate)`` through a blake2b-derived seed, so the whole
study reproduces exactly.
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

from tfg._compat import ensure_ot_stub    # noqa: E402

ensure_ot_stub()

import dist_utils                          # noqa: E402
from LossFunctions import MMDLoss, RBF     # noqa: E402
from tfg import oracle                     # noqa: E402
from tfg.train_cm import SEEDS, load_one   # noqa: E402

N_GRID = [1, 2, 4, 8, 16, 32, 64]


def key_seed(*parts):
    payload = repr(parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(),
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


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------

def target_samples(params, n):
    """Target set S_G, drawn exactly as Optimization.py:120 does."""
    tc = torch.stack([v.float() for v in params["target_variances"]])
    return dist_utils.generate_mog_samples_not_differentiable(
        n, params["target_means"].float(), tc, params["target_weights"].float())


def conditional_samples_cm(model, x, n, device="cpu"):
    cond = x.reshape(1, -1).repeat(n, 1)
    y, _, _ = model.sample(nsamples=n, condition_x=cond, device=device)
    return y


def conditional_samples_analytic(x, params, n):
    """Differentiable draw from the EXACT conditional, reparameterised.

    Component indices are sampled without gradient (as the mixture's discrete
    latent), and the pathwise dependence on x flows through the conditional
    means and Cholesky factors. This mirrors what the consistency model does
    functionally -- map x to samples -- so the two references are comparable.
    """
    m, c, w = oracle.conditional_params(
        x.double(), params["mu_list"], params["Sigma_list"], params["alpha"])
    chol = torch.linalg.cholesky(c)
    comp = torch.multinomial(w.detach(), n, replacement=True)
    eps = torch.randn(n, m.shape[1], dtype=torch.float64)
    return (m[comp] + torch.einsum("nij,nj->ni", chol[comp], eps)).float()


def one_gradient(x_value, n, params, model=None, bandwidth=None, seed=0,
                 source="cm", device="cpu"):
    """One realisation of the MLGD-F gradient estimate at ``x``."""
    torch.manual_seed(seed)          # keyed determinism for the model's RNG
    x = torch.tensor([float(x_value)], dtype=torch.float32, requires_grad=True)

    if source == "cm":
        y = conditional_samples_cm(model, x, n, device=device)
    else:
        y = conditional_samples_analytic(x, params, n)

    tgt = target_samples(params, n)
    loss = MMDLoss(kernel=RBF(bandwidth=bandwidth, device="cpu"),
                   device="cpu")(y, tgt)
    g, = torch.autograd.grad(loss, x, allow_unused=True)
    g = torch.zeros_like(x) if g is None else g
    return float(loss.detach()), float(g[0])


def replicate(x_value, n, reps, params, role, **kw):
    vals, grads = [], []
    for r in range(reps):
        s = key_seed(role, x_value, n, r)
        v, g = one_gradient(x_value, n, params, seed=s, **kw)
        vals.append(v)
        grads.append(g)
    t = torch.tensor(grads, dtype=torch.float64)
    return torch.tensor(vals, dtype=torch.float64), t


def reference_gradient(x_value, params, n_ref, reps, role, **kw):
    _, g = replicate(x_value, n_ref, reps, params, role, **kw)
    return float(g.mean()), float(g.std(unbiased=True) / reps ** 0.5)


def stats_vs_reference(grads, g_ref):
    reps = grads.numel()
    mean = float(grads.mean())
    se_mean = float(grads.std(unbiased=True) / reps ** 0.5)
    var = float(grads.var(unbiased=True))
    sq = (grads - g_ref) ** 2
    return {
        "mean": mean, "se_mean": se_mean,
        "bias": mean - g_ref, "se_bias": se_mean,
        "variance": var,
        "se_variance": float(var * (2.0 / (reps - 1)) ** 0.5),
        "mse": float(sq.mean()),
        "se_mse": float(sq.std(unbiased=True) / reps ** 0.5),
        "rel_rmse": float(sq.mean() ** 0.5 / max(1e-30, abs(g_ref))),
        "sign_agreement": float(((grads * g_ref) > 0).double().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tfg_results/cm_estimator_study.json")
    ap.add_argument("--ckpt-dir", default="tfg_results/cm_checkpoints")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--reps-stability", type=int, default=1200)
    ap.add_argument("--ref-reps", type=int, default=120)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = ap.parse_args()

    params = oracle.load_params()
    x_opt = -4.9984462966235057
    x_probe = round(x_opt + 0.35, 6)

    payload = {"metadata": {**git_metadata(), "torch": torch.__version__,
                            "python": sys.version.split()[0],
                            "platform": platform.platform(),
                            "n_grid": N_GRID, "x_opt": x_opt,
                            "x_probe": x_probe, "reps": args.reps,
                            "ref_reps": args.ref_reps,
                            "provenance": "RE-TRAINED consistency models; "
                                          "not the paper's weights"},
               "models": []}

    for i, seed in enumerate(args.seeds):
        role = "MAIN" if i == 0 else f"robustness-{i}"
        model, _ = load_one(seed, args.ckpt_dir)
        entry = {"seed": seed, "role": role, "settings": {}}
        print(f"\n################ {role}  seed={seed} ################")

        for setting, bw in (("adaptive", None), ("fixed", 1.0)):
            print(f"\n=== setting: {setting} bandwidth ===")
            t0 = time.perf_counter()

            # -- references, with a stability check on n_ref and on reps --
            refs = {}
            for n_ref in (1024, 2048):
                for rr in (args.ref_reps, 2 * args.ref_reps):
                    m_, se_ = reference_gradient(
                        x_probe, params, n_ref, rr, f"ref-cm-{setting}",
                        model=model, bandwidth=bw, source="cm")
                    refs[f"cm_n{n_ref}_reps{rr}"] = {"mean": m_, "se": se_}
                    print(f"   ref_cm  n={n_ref:<5} reps={rr:<5} "
                          f"g={m_:+.6e} +- {se_:.2e}")
            g_ref_cm = refs[f"cm_n2048_reps{2 * args.ref_reps}"]["mean"]

            m_a, se_a = reference_gradient(
                x_probe, params, 2048, 2 * args.ref_reps,
                f"ref-analytic-{setting}", bandwidth=bw, source="analytic")
            refs["analytic_n2048"] = {"mean": m_a, "se": se_a}
            print(f"   ref_analytic n=2048 g={m_a:+.6e} +- {se_a:.2e}")
            print(f"   MODEL ERROR (cm - analytic) = {g_ref_cm - m_a:+.6e}")

            # -- n sweep --
            rows = []
            print(f"   {'n':>3} {'mean':>13} {'bias':>13} {'variance':>12} "
                  f"{'MSE':>12} {'relRMSE':>9} {'sign%':>6}")
            for n in N_GRID:
                _, grads = replicate(x_probe, n, args.reps, params,
                                     f"sweep-{setting}", model=model,
                                     bandwidth=bw, source="cm")
                st = stats_vs_reference(grads, g_ref_cm)
                st["n"] = n
                st["model_error_vs_analytic"] = g_ref_cm - m_a
                st["total_error_vs_analytic"] = st["mean"] - m_a
                rows.append(st)
                print(f"   {n:>3} {st['mean']:>13.5e} {st['bias']:>13.5e} "
                      f"{st['variance']:>12.4e} {st['mse']:>12.4e} "
                      f"{st['rel_rmse']:>9.3f} {st['sign_agreement']:>6.2f}")

            # replication-count stability at one n
            _, g_hi = replicate(x_probe, 8, args.reps_stability, params,
                                f"sweep-{setting}", model=model,
                                bandwidth=bw, source="cm")
            st_hi = stats_vs_reference(g_hi, g_ref_cm)
            print(f"   [stability] n=8 reps={args.reps_stability}: "
                  f"MSE={st_hi['mse']:.4e} vs reps={args.reps} "
                  f"MSE={rows[3]['mse']:.4e}")

            entry["settings"][setting] = {
                "references": refs, "g_ref_cm": g_ref_cm,
                "g_ref_analytic": m_a,
                "model_error": g_ref_cm - m_a,
                "sweep": rows,
                "stability_n8": st_hi,
                "seconds": time.perf_counter() - t0,
            }

        # -- temporal lambda, main setting only --
        print("\n=== temporal lambda (adaptive bandwidth) ===")
        bw = None
        g_ref_t = entry["settings"]["adaptive"]["g_ref_cm"]
        lam_runs = []
        for label, dx in (("low_drift", 0.01), ("low_drift_finer", 0.003),
                          ("high_drift", 0.5), ("high_drift_extreme", 1.5)):
            x_next = round(x_probe + dx, 6)
            g_ref_n, _ = reference_gradient(x_next, params, 2048,
                                            2 * args.ref_reps,
                                            f"ref-next-{label}", model=model,
                                            bandwidth=bw, source="cm")
            D2 = (g_ref_t - g_ref_n) ** 2
            for n in (4, 32):
                # Independent conditional samples across timesteps, as the
                # algorithm specifies: different role strings -> different keys.
                _, gt = replicate(x_probe, n, args.reps, params,
                                  f"lam-t-{label}", model=model,
                                  bandwidth=bw, source="cm")
                _, gn = replicate(x_next, n, args.reps, params,
                                  f"lam-next-{label}", model=model,
                                  bandwidth=bw, source="cm")
                v = 0.5 * (float(gt.var(unbiased=True)) + float(gn.var(unbiased=True)))

                curve = []
                for lam in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
                    agg = (1 - lam) * gt + lam * gn
                    sq = (agg - g_ref_t) ** 2
                    curve.append({"lambda": lam, "mse": float(sq.mean()),
                                  "se_mse": float(sq.std(unbiased=True)
                                                  / args.reps ** 0.5)})
                fine = torch.linspace(0, 0.95, 192, dtype=torch.float64)
                mses = torch.tensor([float((((1 - l) * gt + l * gn) - g_ref_t)
                                           .pow(2).mean()) for l in fine])
                lam_emp = float(fine[int(mses.argmin())])
                lam_pred = v / (D2 + 2 * v) if (D2 + 2 * v) > 0 else 0.0
                mse0 = float((gt - g_ref_t).pow(2).mean())
                rec = {"label": label, "n": n, "x_t": x_probe, "x_next": x_next,
                       "D2": D2, "v": v, "regime": "D2<v" if D2 < v else "D2>=v",
                       "lambda_empirical": lam_emp,
                       "lambda_predicted": lam_pred,
                       "mse_lambda0": mse0, "mse_best": float(mses.min()),
                       "gain": 1 - float(mses.min()) / mse0, "curve": curve}
                lam_runs.append(rec)
                print(f"   {label:<19} n={n:<3} D2={D2:.3e} v={v:.3e} "
                      f"({rec['regime']})  lam_emp={lam_emp:.3f} "
                      f"lam_pred={lam_pred:.3f}  gain={rec['gain']:+.1%}")
        entry["lambda_study"] = lam_runs
        payload["models"].append(entry)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
