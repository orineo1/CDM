"""
Control experiment for the paper's Appendix G claim that gradient subsampling
(Uniform / Witness at k/n = 0.5) beats backpropagating through all samples.

Runs the library optimizer (guidance.optimize_guided) with the exact
configuration of the legacy optimize_LGD(CM=True) path, plus one knob:
which conditional samples keep their autograd graph. Setting: 10D_cond_1D
synthetic MoG with the consistency-model sampler, as in
conditional-matching-paper/simulations/notebooks/Exp_10D_cond_1D.ipynb.
Pretrained checkpoints and GMM parameters come from that repo (HuggingFace
download on first use), so pass its simulations/ directory with --sim-dir.

Configurations (each run --n-restarts times):

  all_250            n=250, k=250, zeta=1          the paper's "All" rule
  uniform_detach     n=250, k=125, zeta=1          loss on all n, n-k detached
                                                   (Appendix G description);
                                                   E[grad] ~ (k/n) x All
  uniform_detach_rs  n=250, k=125, zeta=n/k        same, gradient rescaled by n/k
  uniform_subset     n=250, k=125, zeta=1          loss and gradient on the k
                                                   kept samples only (Alg. 2 l.6)
  all_125            n=125, k=125, zeta=1          All at half the sample budget
  all_250_halfzeta   n=250, k=250, zeta=0.5        All with half the step size

Reading it: if all_125 or all_250_halfzeta match uniform_detach, the
subsampling advantage is a budget / step-size effect. If uniform_detach_rs is
worse than uniform_detach, the unrescaled gradient was gaining from shrinkage.

Usage (one GPU):

    python scripts/control_subsample.py --sim-dir /path/to/conditional-matching-paper/simulations \
        --n-restarts 25 --out results/control_subsample_10d.json

Quick pass: --n-restarts 10, or --configs all_250 uniform_detach all_125.
Every run is also appended to results/runs.jsonl (gcio.results) under
experiment "control_subsample_10d".
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from guidance import (GuidanceConfig, optimize_guided, init_randn, init_zeros,  # noqa: E402
                      perturb_beta_ratio, aggregate_logsumexp)
from losses.functions import MMDLoss, RBF                                          # noqa: E402
from gcio.results import save_run                                               # noqa: E402
from utils import dist as dist_utils  # noqa: E402

# name: (n_cond, k_grad, zeta, rescale_by_n_over_k, grad_loss)
CONFIGS = {
    "all_250":           (250, 250, 1.0, False, "full"),
    "uniform_detach":    (250, 125, 1.0, False, "full"),
    "uniform_detach_rs": (250, 125, 1.0, True,  "full"),
    "uniform_subset":    (250, 125, 1.0, False, "subset"),
    "all_125":           (125, 125, 1.0, False, "full"),
    "all_250_halfzeta":  (250, 250, 0.5, False, "full"),
}

EXPERIMENT = "10D_cond_1D"
SEED = 42            # the notebook's GLOBAL_SEED; checkpoints are keyed on it
CONDITION_ON = 9
NBLOCKS, NUNITS, DIFFUSION_STEPS = 8, 512, 100
N_MC = 5             # NUM_X_T_LGD_CM in the notebook


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-dir", required=True, help="conditional-matching-paper/simulations")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-restarts", type=int, default=25)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--init", choices=["zeros", "randn"], default="zeros",
                   help="x_T init: the paper-repo code uses zeros, the paper text says N(0,I)")
    p.add_argument("--out", default="results/control_subsample_10d.json")
    return p.parse_args()


def load_setting(sim_dir, device):
    """GMM parameters and the two pretrained models, from the paper repo."""
    src = os.path.join(os.path.abspath(sim_dir), "src")
    sys.path.insert(0, src)
    import experiment_utils                       # noqa: E402  (paper repo)
    import Diffusion as PDiffusion                # noqa: E402  (paper repo classes match the checkpoints)
    import ConsistencyModels as PCM               # noqa: E402

    params_dir = os.path.join(sim_dir, "params")
    results_dir = os.path.join(sim_dir, "results", EXPERIMENT)
    ckpt_dir = os.path.join(sim_dir, "checkpoints", EXPERIMENT)

    loaded = (experiment_utils.load_gmm_params(params_dir, EXPERIMENT)
              or experiment_utils.load_gmm_params(results_dir, EXPERIMENT))
    if loaded is None:                            # same fallback as the notebook
        experiment_utils.set_global_seed(SEED)
        mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = \
            dist_utils.get_param_mog_with_target(dim_data=10, num_components=4, device="cpu",
                                                 conditional_modes=2, distanceOrScale="Distance")
        mog_means, mog_variances, weights = dist_utils.filter_and_normalize(
            mog_means, mog_variances, weights, threshold=0.001)
    else:
        mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = loaded
    gmm = dict(mu_list=[m.float() for m in mu_list], Sigma_list=[c.float() for c in Sigma_list],
               alpha=alpha.float(), mog_means=mog_means, mog_variances=mog_variances,
               weights=weights, x_star=x_star)

    cm = PCM.ConsistencyModeliCT(nfeatures=10 - CONDITION_ON, condition_on=CONDITION_ON,
                                 nunits=NUNITS, depth=NBLOCKS)
    uncond = PDiffusion.DiffusionModel(nfeatures=CONDITION_ON, nblocks=NBLOCKS, nunits=NUNITS,
                                       condition=False, diffusion_steps=DIFFUSION_STEPS)
    for model, name in ((cm, "CM"), (uncond, "Diffusion_uncond")):
        if not experiment_utils.load_checkpoint_with_hf_fallback(model, name, ckpt_dir, EXPERIMENT, SEED, device):
            sys.exit(f"could not load checkpoint {name}")
        model.to(device).eval()
    return gmm, uncond, cm


def make_cfg(gmm, n_cond, k_grad, zeta, rescale, grad_loss, init, device, seed):
    """GuidanceConfig of legacy optimize_LGD(CM=True) plus the subsampling knob."""

    def keep_subset(y):
        # Called on the n_cond conditional samples before the loss. Under
        # no_grad (final evaluation) it is a no-op, so the final loss always
        # uses the full budget.
        if k_grad >= y.shape[0] or not torch.is_grad_enabled():
            return y
        perm = torch.randperm(y.shape[0], device=y.device)
        keep, drop = perm[:k_grad], perm[k_grad:]
        if grad_loss == "full":
            return torch.cat([y[keep], y[drop].detach()], dim=0)   # loss on all n
        return y[keep]                                            # loss on k only

    step = zeta * (n_cond / k_grad if rescale else 1.0)
    return GuidanceConfig(
        n_cond=n_cond, n_mc=N_MC,
        init_fn=init_zeros if init == "zeros" else init_randn, unsqueeze_init=True,
        perturb_fn=perturb_beta_ratio, perturb_square=False,
        aggregate_fn=aggregate_logsumexp,
        step_size_fn=lambda t, i: step,
        timestep_source="range", ddim_step_style="condition_x",
        repeat_condition=True, condition_on_slice=False,
        cond_output_transform=keep_subset,
        target_generator=lambda n, dev: dist_utils.generate_mog_samples_not_differentiable(
            n, gmm["mog_means"], gmm["mog_variances"], gmm["weights"]).to(dev),
        compute_final_loss=True, device=device, seed=seed,
        tag=f"control_subsample:{n_cond}/{k_grad}/zeta{zeta}/{'rs' if rescale else 'raw'}/{grad_loss}",
    )


def evaluate(x_pred, gmm):
    mu_p, Sigma_p = dist_utils.compute_conditionals(gmm["mu_list"], gmm["Sigma_list"], x_pred)
    w_p = dist_utils.compute_alpha(gmm["mu_list"], gmm["Sigma_list"], gmm["alpha"], x_pred)
    l2_gmm = dist_utils.gmm_l2_distance(mu_p, Sigma_p, w_p, gmm["mog_means"], gmm["mog_variances"], gmm["weights"])
    l2_x = (x_pred - gmm["x_star"].float().view(-1).cpu()).pow(2).sum().sqrt().item()
    return float(l2_gmm), l2_x


def summarise(name, runs):
    fl = np.array([r["final_loss"] for r in runs]); g = np.array([r["l2_gmm"] for r in runs])
    lx = np.array([r["l2_x"] for r in runs]); tm = np.array([r["time_s"] for r in runs])
    top = np.argsort(fl)[:10]
    return (f"{name:18s} all: L2GMM {g.mean():.3f}±{g.std():.3f} L2x {lx.mean():.2f}±{lx.std():.2f} | "
            f"top10: L2GMM {g[top].mean():.3f}±{g[top].std():.3f} L2x {lx[top].mean():.2f}±{lx[top].std():.2f} | "
            f"{tm.mean():.1f}s/run")


def main():
    args = parse_args()
    gmm, uncond, cm = load_setting(args.sim_dir, args.device)
    loss_fn = MMDLoss(kernel=RBF(device=args.device), device=args.device)

    out = {"experiment": EXPERIMENT, "seed": SEED, "init": args.init, "n_mc": N_MC,
           "n_restarts": args.n_restarts, "configs": {}}
    for name in args.configs:
        n_cond, k_grad, zeta, rescale, grad_loss = CONFIGS[name]
        runs = []
        for i in range(args.n_restarts):
            cfg = make_cfg(gmm, n_cond, k_grad, zeta, rescale, grad_loss, args.init, args.device, SEED + i)
            t0 = time.time()
            res = optimize_guided(uncond, cm, loss_fn, cfg)
            dt = time.time() - t0
            x_pred = res["x_final"].view(-1).cpu()
            l2_gmm, l2_x = evaluate(x_pred, gmm)
            metrics = {"final_loss": float(res["final_loss"]), "l2_gmm": l2_gmm, "l2_x": l2_x,
                       "time_s": dt, "x_final": x_pred.tolist()}
            save_run(experiment="control_subsample_10d", config=cfg, metrics=metrics,
                     seed=SEED + i, variant=name)
            runs.append(metrics)
            print(f"[{name} {i+1}/{args.n_restarts}] loss {metrics['final_loss']:.4f} "
                  f"L2GMM {l2_gmm:.4f} L2x {l2_x:.3f} {dt:.1f}s", flush=True)
        out["configs"][name] = dict(n_cond=n_cond, k_grad=k_grad, zeta=zeta, rescale=rescale,
                                    grad_loss=grad_loss, runs=runs)
        print(summarise(name, runs), flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)

    print("\n==== summary (mean ± std over restarts; top10 by final loss) ====")
    for name, cfg in out["configs"].items():
        print(summarise(name, cfg["runs"]))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
