"""Is a trajectory's outcome decided by its start x_T or by the noise of the guidance gradient?

Runs guided MLGD trajectories on a grid of (x_T seed) x (guidance seed): x_T is drawn from its own
seed and passed in fixed; the guidance seed drives everything after it (conditional samples, MC
perturbations, refinement draws). The reverse process is deterministic DDIM (eta = 0), so these are
the only sources of randomness. Reports the exact TV of every trajectory, the share of its variance
explained by x_T (between-start variance / total), the agreement of the final mixture component
across guidance seeds of one start, and the step after which the dominant component of the Tweedie
estimate x0_hat stops changing.

All search_mog flags are accepted (the same settings as the run being diagnosed), plus:
  --n-xt N --n-guide G --xt-seed-base S --out results/xt_vs_noise.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import search_mog as fx                                                            # noqa: E402
from make_table1 import _load_mixture, scale_free_distances                        # noqa: E402
from cdm.guidance import optimize_guided                                           # noqa: E402
from cdm.losses.exact_mmd import ExactConditionalMMD, IdentitySampler              # noqa: E402
from cdm.utils.dist import compute_alpha                                           # noqa: E402


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--n-xt", type=int, default=40)
    extra.add_argument("--n-guide", type=int, default=4)
    extra.add_argument("--xt-seed-base", type=int, default=7000)
    extra.add_argument("--diag-out", required=True)
    e, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = fx.parse_args()
    dev = args.device
    prior, _ = fx.load_model(args.prior, dev)
    gen, info = fx.load_target(args.target, dev)
    if args.sampler == "exactloss":
        sampler, stype = IdentitySampler(), "exactloss"
        ex = ExactConditionalMMD(info["mu_list"], info["Sigma_list"], info["alpha"], info["target"])
        loss_fn = lambda pred, target: ex(pred[0]).float().to(pred.device)          # noqa: E731
    else:
        sampler, stype = fx.load_model(args.sampler, dev)
        loss_fn = fx.loss_for_target(args, gen)
    data = _load_mixture(args.target)
    raw = torch.load(args.target, map_location="cpu", weights_only=False)
    mu, Sig, al = raw["mu_list"], raw["Sigma_list"], raw["alpha"]
    d = int(prior.nfeatures)
    abar = prior.baralphas.detach().cpu()

    def comp_path(trace):
        ks = []
        for s in trace:
            x = torch.tensor(s["x_t"], device=dev).view(1, -1)
            t = int(s["t"])
            with torch.no_grad():
                eps = prior(x, torch.full([1, 1], t, device=dev), None)
            ab = float(abar[t])
            x0 = ((x - (1 - ab) ** 0.5 * eps) / ab ** 0.5).view(-1).float().cpu()
            ks.append(int(torch.argmax(compute_alpha(mu, Sig, al, x0)).item()))
        return ks

    rows = []
    t0 = time.time()
    for i in range(e.n_xt):
        g = torch.Generator(device="cpu").manual_seed(e.xt_seed_base + i)
        xT = torch.randn(d, generator=g).to(dev)
        for j in range(e.n_guide):
            cfg = fx.make_cfg(args, prior, stype, gen, seed=100000 + 1000 * i + j)
            cfg.init_fn = lambda nf, dv, v=xT: v.clone().detach().requires_grad_(True)
            cfg.record_x = True
            res = optimize_guided(prior, sampler, loss_fn, cfg)
            xh = res["x_final"].detach().float().view(-1).cpu()
            tv = scale_free_distances(data, xh.tolist(), n_mc=5000, seed=i * 100 + j)["tv"]
            xb = (res.get("refine") or {}).get("x_before")
            tvb = scale_free_distances(data, xb.view(-1).cpu().tolist(), n_mc=5000, seed=i * 100 + j)["tv"] if xb is not None else float("nan")
            ks = comp_path(res["trace"])
            last_change = max([n for n in range(1, len(ks)) if ks[n] != ks[n - 1]], default=0)
            k_final = int(torch.argmax(compute_alpha(mu, Sig, al, xh)).item())
            rows.append({"xt": i, "guide": j, "tv": tv, "tv_before": tvb, "final_loss": float(res["final_loss"]),
                         "k_final": k_final, "k_path": ks, "commit_step": last_change, "n_steps": len(ks)})
            print(f"xT {i:2d} guide {j}: TV {tv:.3f} (before refine {tvb:.3f}) comp {k_final} commit@{last_change}/{len(ks)}  [{time.time()-t0:.0f}s]", flush=True)
    tv = np.array([r["tv"] for r in rows]).reshape(e.n_xt, e.n_guide)
    good = tv < 0.1
    between = float(np.var(tv.mean(1)))
    total = float(np.var(tv))
    kf = np.array([r["k_final"] for r in rows]).reshape(e.n_xt, e.n_guide)
    same_comp = float(np.mean([len(set(row)) == 1 for row in kf]))
    commits = np.array([r["commit_step"] for r in rows])
    summ = {"n_xt": e.n_xt, "n_guide": e.n_guide, "sampler": args.sampler,
            "tv_mean": float(tv.mean()), "frac_good_tv_lt_0.1": float(good.mean()),
            "share_var_tv_between_xT": between / total if total > 0 else float("nan"),
            "starts_with_any_good": int(good.any(1).sum()), "starts_all_good": int(good.all(1).sum()),
            "P_good_given_start_has_a_good": float(good[good.any(1)].mean()) if good.any() else float("nan"),
            "starts_same_final_component_all_seeds": same_comp,
            "commit_step_quantiles_of_n_steps": np.round(np.quantile(commits, [.1, .5, .9]) / rows[0]["n_steps"], 3).tolist()}
    print(json.dumps(summ, indent=1))
    os.makedirs(os.path.dirname(e.diag_out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": [{k: v for k, v in r.items() if k != "k_path"} for r in rows],
               "k_paths": [r["k_path"] for r in rows]}, open(e.diag_out, "w"))
    print("wrote", e.diag_out)


if __name__ == "__main__":
    main()
