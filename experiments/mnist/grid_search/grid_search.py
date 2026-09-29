#!/usr/bin/env python
"""Hyperparameter grid search for MLGD-F on rotated MNIST (uniform / bimodal targets).

Every combination of the value lists below is one configuration; each configuration
runs N seeds exactly like `table_mlgdf/run_mlgdf.py` (same optimiser, same seeding,
same classifier and threshold), then:

  * labels every recovered image with the classifier (label + confidence),
  * ranks the seeds by final SWD and takes the best 5 ("top-5"),
  * scores the configuration by the classifier on those top-5:
        top5_clf_score = (# top-5 images the classifier labels) + (their mean confidence)
    so 5 labelled images beat 4, and confidence breaks ties (range 0 .. 6),
  * saves the run record, a metrics JSON and two figures: all seeds, and the top-5,
    each image titled 'Seed | Loss | label (confidence)',
  * logs all of it to Weights & Biases (one W&B run per configuration, grouped by
    target), so the W&B table can be sorted by `top5_clf_score`.

    # how many configurations the grid has
    python grid_search.py --target uniform --count  --n-cond 600 1000 --steps 290 300 ...
    # run configuration 7 of it (SLURM array task), or all of them when --index is omitted
    python grid_search.py --target uniform --index 7 --n-cond 600 1000 --steps 290 300 ...

`summarize.py` ranks the finished configurations of one target locally.
"""

import argparse
import itertools
import json
import os
import pickle
import random
import time

import numpy as np
# torch, diffusers and the models are imported in main(), after --count: counting the
# grid (done on the login node to size the SLURM array) must not load them.

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.dirname(HERE)
GLOBAL_SEED = 42
PAPER_THRESHOLD = 0.75          # the classifier threshold used for the paper's Table 2


def make_deterministic(seed=GLOBAL_SEED):
    import torch
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
    p.add_argument("--target", choices=["uniform", "bimodal"], required=True)
    g = p.add_argument_group("the grid: every combination of these lists is one configuration")
    g.add_argument("--n-cond", type=int, nargs="+", required=True)
    g.add_argument("--steps", type=int, nargs="+", required=True)
    g.add_argument("--n-mc", type=int, nargs="+", required=True)
    g.add_argument("--step-size-mode", nargs="+", required=True)
    g.add_argument("--bimodal-var", type=float, nargs="+", default=[252.0],
                   help="bimodal target variance in degrees^2 (ignored for uniform)")
    g.add_argument("--clamp", type=int, nargs="+", default=[0], choices=[0, 1],
                   help="per-step clamp to [-1, 1]; the paper's runs used 0")
    s = p.add_argument_group("what to run")
    s.add_argument("--count", action="store_true", help="print the number of configurations and exit")
    s.add_argument("--index", type=int, default=None,
                   help="run only this configuration (0-based; the SLURM array index)")
    s.add_argument("--n-seeds", type=int, default=15)
    s.add_argument("--skip-existing", action="store_true",
                   help="skip a configuration whose metrics.json already exists")
    c = p.add_argument_group("classifier")
    c.add_argument("--threshold", type=float, default=PAPER_THRESHOLD)
    c.add_argument("--classifier-path",
                   default=os.path.join(MNIST_DIR, "checkpoints", "robust_classifier.pth"))
    r = p.add_argument_group("runtime")
    r.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    r.add_argument("--ckpt-dir", default=os.path.join(MNIST_DIR, "checkpoints"))
    r.add_argument("--allow-untrained-prior", action="store_true",
                   help="SMOKE ONLY: use an untrained UNet if the prior checkpoint is missing")
    r.add_argument("--out", default=os.path.join(HERE, "outputs"))
    w = p.add_argument_group("Weights & Biases")
    w.add_argument("--wandb-project", default="cdm-mnist-grid")
    w.add_argument("--wandb-entity", default=None)
    w.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online",
                   help="offline writes to ./wandb for a later `wandb sync` (compute nodes "
                        "without internet); disabled skips W&B")
    return p.parse_args()


def grid(args):
    bvars = args.bimodal_var if args.target == "bimodal" else [None]
    return [dict(n_cond=nc, steps=st, n_mc=nm, step_size_mode=ss, bimodal_var=bv, clamp=cl)
            for nc, st, nm, ss, bv, cl in itertools.product(
                args.n_cond, args.steps, args.n_mc, args.step_size_mode, bvars, args.clamp)]


def run_name(target, cfg):
    var = f"_var{int(cfg['bimodal_var'])}" if cfg["bimodal_var"] is not None else ""
    return (f"{target}{var}_st{cfg['steps']}_ss{cfg['step_size_mode']}_xt{cfg['n_mc']}"
            f"_ns{cfg['n_cond']}_cl{cfg['clamp']}")


def top5_metrics(losses, preds, confs, times):
    losses = np.asarray(losses, dtype=float)
    top = np.argsort(losses)[:5]
    top_preds = [preds[i] for i in top]
    top_confs = [confs[i] for i in top]
    labelled = [p for p in preds if p is not None]
    n_top_lab = sum(p is not None for p in top_preds)
    return {
        "top5_clf_score": n_top_lab + float(np.mean(top_confs)),
        "top5_n_labelled": n_top_lab,
        "top5_conf_mean": float(np.mean(top_confs)),
        "top5_labels": [p if p is not None else "None" for p in top_preds],
        "top5_seeds": [int(i) for i in top],
        "swd_top5_mean": float(losses[top].mean()),
        "swd_top5_std": float(losses[top].std()),
        "swd_all_mean": float(losses.mean()),
        "swd_all_std": float(losses.std()),
        "n_labelled": len(labelled),
        "digit_counts": {str(d): sum(p == d for p in labelled) for d in range(10)},
        "time_mean_s": float(np.mean(times)),
    }


def log_wandb(args, name, cfg, metrics, record, fig_all, fig_top):
    import wandb
    from cdm.models.mnist.plotting import clamp_normalize
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
                     group=args.target, name=name, reinit=True,
                     config={"target": args.target, **cfg, "n_seeds": args.n_seeds,
                             "threshold": args.threshold, "global_seed": GLOBAL_SEED},
                     tags=["mnist", "grid", args.target])
    run.summary.update({k: v for k, v in metrics.items() if not isinstance(v, (list, dict))})
    run.summary["top5_labels"] = str(metrics["top5_labels"])
    shown = clamp_normalize(record)["results"]
    table = wandb.Table(columns=["seed", "swd", "label", "confidence", "in_top5", "image"])
    for k in range(len(record["results"])):
        lab = record["predicted_labels"][k]
        table.add_data(int(record["seed_log"][k]), float(record["loss_log"][k]),
                       "None" if lab is None else str(lab), float(record["predicted_confidences"][k]),
                       k in metrics["top5_seeds"], wandb.Image(np.asarray(shown[k])))
    run.log({"seeds": table,
             "all_seeds": wandb.Image(fig_all, caption="all seeds: Seed | Loss | label (confidence)"),
             "top5": wandb.Image(fig_top, caption="top-5 by SWD: Seed | Loss | label (confidence)")})
    run.finish()


def main():
    args = parse_args()
    configs = grid(args)
    if args.count:
        print(len(configs))
        return
    todo = range(len(configs)) if args.index is None else [args.index]
    if args.index is not None and not 0 <= args.index < len(configs):
        raise SystemExit(f"--index {args.index} out of range: the grid has {len(configs)} configurations")

    import torch
    from diffusers import DDIMScheduler
    from cdm.guidance.core import optimize_guided
    from cdm.models.mnist.checkpoints import load_cond_model, load_uncond_model, make_noise_scheduler
    from cdm.models.mnist.classifier import classify_generated_images, load_or_train_classifier
    from cdm.models.mnist.guidance import build_config, make_swd_loss
    from cdm.models.mnist.plotting import clamp_normalize, plot_all_seeds_grid
    from cdm.models.mnist.targets import make_target

    make_deterministic(GLOBAL_SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cond_model = load_cond_model(device=device, local_dir=args.ckpt_dir)
    uncond_model = load_uncond_model(device=device, local_dir=args.ckpt_dir,
                                     allow_untrained=args.allow_untrained_prior)
    classifier = load_or_train_classifier(args.classifier_path, device=device)
    loss_function = make_swd_loss(device=device, n_projections=50)
    warmed = False

    for idx in todo:
        cfg = configs[idx]
        name = run_name(args.target, cfg)
        out_dir = os.path.join(args.out, args.target, name)
        if args.skip_existing and os.path.exists(os.path.join(out_dir, "metrics.json")):
            print(f"[{idx}] {name}: exists, skipped")
            continue
        os.makedirs(out_dir, exist_ok=True)
        target = make_target(args.target, bimodal_var=cfg["bimodal_var"] or 252.0)
        ddim = DDIMScheduler.from_config(make_noise_scheduler().config)
        ddim.set_timesteps(num_inference_steps=cfg["steps"])
        print(f"\n[{idx}/{len(configs) - 1}] {name}")

        def config_for(seed, sched=ddim, tag=name):
            return build_config(target, sched, n_cond=cfg["n_cond"], n_mc=cfg["n_mc"],
                                step_size_mode=cfg["step_size_mode"], device=device,
                                seed=seed, clamp=bool(cfg["clamp"]), tag=tag)

        if str(device).startswith("cuda") and not warmed:
            # untimed warm-up; every seed re-seeds torch, so no result can change
            warm = DDIMScheduler.from_config(make_noise_scheduler().config)
            warm.set_timesteps(num_inference_steps=3)
            optimize_guided(uncond_model, cond_model, loss_function, config_for(GLOBAL_SEED, warm, "warmup"))
            torch.cuda.synchronize()
            warmed = True

        results, losses, seeds, times = [], [], [], []
        for seed in range(args.n_seeds):
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            out = optimize_guided(uncond_model, cond_model, loss_function, config_for(seed))
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            times.append(time.time() - t0)
            results.append(out["x_final"].squeeze().detach().cpu().numpy())
            losses.append(float(out["final_loss"]))
            seeds.append(seed)
            print(f"  [seed {seed:2d}] swd={losses[-1]:.4f}  time={times[-1]:.1f}s")

        preds, confs = classify_generated_images(results, classifier, device, threshold=args.threshold)
        metrics = top5_metrics(losses, preds, confs, times)
        record = {"experiment_name": name, "target": args.target, "config": cfg,
                  "results": results, "loss_log": losses, "seed_log": seeds, "time_log": times,
                  "predicted_labels": preds, "predicted_confidences": confs,
                  "threshold": args.threshold}
        with open(os.path.join(out_dir, "record.pkl"), "wb") as fh:
            pickle.dump(record, fh)
        with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
            json.dump({"run_name": name, "target": args.target, **cfg, **metrics,
                       "threshold": args.threshold}, fh, indent=1)

        shown = clamp_normalize(record)
        fig_all = plot_all_seeds_grid(shown, out_dir, show_conf=True, title=name)
        fig_top = plot_all_seeds_grid(shown, out_dir, indices=metrics["top5_seeds"], suffix="top5",
                                      show_conf=True, title=f"{name} - top-5 by SWD")
        print(f"  top5_clf_score={metrics['top5_clf_score']:.3f}  "
              f"labelled {metrics['top5_n_labelled']}/5  conf {metrics['top5_conf_mean']:.3f}  "
              f"labels {metrics['top5_labels']}  swd_top5={metrics['swd_top5_mean']:.4f}")

        if args.wandb_mode != "disabled":
            log_wandb(args, name, cfg, metrics, record, fig_all, fig_top)


if __name__ == "__main__":
    main()
