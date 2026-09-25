#!/usr/bin/env python
"""MLGD-F on the MNIST rotated-digits task (paper Table 2).

One target, N seeds. The loop is `guidance.core.optimize_guided`; this script
only builds the models, the target and the config, and records what comes back.

    python run_mlgdf.py --target unimodal --steps 130 --n-cond 1500 \
        --n-mc 3 --step-size-mode double --n-seeds 15

Ported from `MNIST/run_mlgdf.py` of `orineo1/conditional-matching-paper`
(branch `main`, MNIST tree at 7c856e2). What the source file carried privately
and this one does not: the optimisation loop, a second copy of the classifier
and its training loop, a sliced-Wasserstein implementation, a circular MoG
density and sampler, and the W&B logging.
"""

import argparse
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.dirname(HERE)                       # experiments/image_generation/mnist
REPO_ROOT = os.path.abspath(os.path.join(MNIST_DIR, "..", "..", ".."))
for _p in (REPO_ROOT, MNIST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusers import DDIMScheduler                                  # noqa: E402
from gcio import save_run                                            # noqa: E402
from guidance.core import optimize_guided                            # noqa: E402

from mnist_cdm.checkpoints import (load_cond_model, load_uncond_model,  # noqa: E402
                                   make_noise_scheduler)
from mnist_cdm.classifier import classify_generated_images, load_or_train_classifier  # noqa: E402
from mnist_cdm.guidance import build_config, make_swd_loss           # noqa: E402
from mnist_cdm.targets import make_target                            # noqa: E402

GLOBAL_SEED = 42


def make_deterministic(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=["uniform", "bimodal", "unimodal"], required=True)
    p.add_argument("--steps", type=int, required=True,
                   help="DDIM steps over the reverse process (Table 13 'T')")
    p.add_argument("--n-cond", type=int, required=True,
                   help="angle samples drawn from f_phi per candidate (Table 13 'n_cond')")
    p.add_argument("--n-mc", type=int, required=True,
                   help="Tweedie perturbations per reverse step (Table 13 'n_MC')")
    p.add_argument("--step-size-mode", default="original",
                   help="zeta_t rule: original | double | half | dps | no_linear | ...")
    p.add_argument("--n-seeds", type=int, default=15)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--unimodal-var", type=float, default=515.0)
    p.add_argument("--bimodal-var", type=float, default=252.0)
    p.add_argument("--no-clamp", action="store_true",
                   help="disable the [-1, 1] clamp on the guided iterate")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt-dir", default=os.path.join(REPO_ROOT, "MnistModels"),
                   help="local checkpoint directory; HuggingFace is the fallback")
    p.add_argument("--classifier-path", default=None,
                   help="rotation-robust classifier .pth (trained if absent)")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="classifier confidence below which a seed is labelled None")
    p.add_argument("--no-classify", action="store_true")
    p.add_argument("--allow-untrained-prior", action="store_true",
                   help="SMOKE TIER ONLY: run with an untrained UNet if no prior checkpoint")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--runs-store", default=None,
                   help="gcio JSONL store for the per-seed records")
    return p.parse_args()


def main():
    args = parse_args()
    make_deterministic(GLOBAL_SEED)
    os.makedirs(args.out, exist_ok=True)
    device = args.device

    target = make_target(args.target, unimodal_var=args.unimodal_var,
                         bimodal_var=args.bimodal_var)

    cond_model = load_cond_model(device=device, local_dir=args.ckpt_dir)
    uncond_model = load_uncond_model(device=device, local_dir=args.ckpt_dir,
                                     allow_untrained=args.allow_untrained_prior)

    ddim = DDIMScheduler.from_config(make_noise_scheduler().config)
    ddim.set_timesteps(num_inference_steps=args.steps)

    var_str = ""
    if args.target == "unimodal":
        var_str = f"_var{int(args.unimodal_var)}"
    elif args.target == "bimodal":
        var_str = f"_var{int(args.bimodal_var)}"
    run_name = (f"{args.target}{var_str}_st{args.steps}_ss{args.step_size_mode}"
                f"_xt{args.n_mc}_ns{args.n_cond}"
                f"_cl{0 if args.no_clamp else 1}")

    print(f"target={args.target}  steps={args.steps}  n_cond={args.n_cond}  "
          f"n_mc={args.n_mc}  rule={args.step_size_mode}  device={device}")

    loss_function = make_swd_loss(device=device, n_projections=50)

    results, loss_log, seed_log, time_log = [], [], [], []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        cfg = build_config(target, ddim, n_cond=args.n_cond, n_mc=args.n_mc,
                           step_size_mode=args.step_size_mode, device=device,
                           seed=seed, clamp=not args.no_clamp,
                           tag=run_name)
        t0 = time.time()
        out = optimize_guided(uncond_model, cond_model, loss_function, cfg)
        elapsed = time.time() - t0
        loss = float(out["final_loss"])
        results.append(out["x_final"].squeeze().detach().cpu().numpy())
        loss_log.append(loss)
        seed_log.append(seed)
        time_log.append(elapsed)
        print(f"[seed {seed:2d}] swd={loss:.4f}  time={elapsed:.1f}s")
        save_run(experiment=f"mnist_{args.target}", config=cfg,
                 metrics={"final_loss": loss, "time_s": elapsed,
                          "n_steps": out["n_steps"], "diverged": out["diverged"]},
                 seed=seed, store=args.runs_store, run_name=run_name)

    x_range, target_pdf = target.density_curve()
    payload = {
        "experiment_name": run_name,
        "target": args.target,
        "nsamples": args.n_cond,
        "num_x_t": args.n_mc,
        "num_inference_steps": args.steps,
        "step_size_mode": args.step_size_mode,
        "clamp": not args.no_clamp,
        "results": results,
        "loss_log": loss_log,
        "seed_log": seed_log,
        "time_log": time_log,
        "x_range": x_range,
        "target_pdf": target_pdf,
        "use_uniform": target.is_uniform,
    }

    if not args.no_classify:
        clf_path = args.classifier_path or os.path.join(args.out, "robust_classifier.pth")
        classifier = load_or_train_classifier(clf_path, device=device)
        preds, confs = classify_generated_images(results, classifier, device,
                                                 threshold=args.threshold)
        payload["predicted_labels"] = preds
        payload["predicted_confidences"] = confs

    path = os.path.join(args.out, f"{run_name}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
