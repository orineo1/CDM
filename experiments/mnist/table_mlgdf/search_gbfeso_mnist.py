#!/usr/bin/env python
"""GB-FESO on rotated MNIST (paper Table 2 setting): the competitor, run on our MNIST networks.

GB-FESO (Zick & Mendels, arXiv:2607.06421) optimizes the design x directly: from a start x0 it
draws n_cond angles from the frozen conditional iCT sampler f_phi(x, .), scores them against target
angles with a distributional loss, and takes Adam steps on x through the differentiated sampler.
No prior over X: the unconditional DDPM is not loaded. The optimizer is the one of the synthetic port,
`experiments/synthetic/pipeline/search_gbfeso.py` (`run_phase` is imported from there):

  phase 0  --steps Adam steps at --lr; after each step Gaussian noise of std --noise-std * (1 - k/steps)
           is added to every pixel; x is clamped to [-1, 1]; the iterate with the lowest loss is kept.
  phase 1  from the best phase-0 point, --steps Adam steps at --lr-refine, no noise, best iterate kept
           (--phase1 always | threshold | never, as in the synthetic port).
  answer   the better of the two best iterates, each scored on final_n = 4 n_cond fresh draws.

Everything that defines the Table 2 problem is MLGD-F's (search_mnist.py): the conditional sampler
(iCT steps [150, 50, 20, 10, 5, 1], the (cos, sin) -> degrees -> (cos, sin) round trip), the targets
(cdm.models.mnist.targets.make_target), the loss (make_swd_loss, 50 projections, fresh target draws
at every step), n_cond per step (config/targets.tsv), and the reported SWD: `_final_loss` of
cdm.guidance.core with the GuidanceConfig of build_config (4 n_cond fresh draws at x), evaluated once
more at the answer, after the phase choice, so the reported number is not the minimum of the two
phase scores.

Design space: x is a 1x28x28 image in the prior's pixel space [-1, 1] (train_uncond.py: x * 2 - 1),
the space MLGD-F's x lives in and is fed to f_phi in; clamping keeps it in [-1, 1] (MLGD-F's Table 2
runs do not clamp; GB-FESO clamps to the data range).

Starts (--init data, default): a random MNIST training digit (torchvision, ToTensor), rotated by a
uniform angle with torchvision's rotate (fill 0) exactly as RotatedMNISTDataset builds the prior's
data, mapped to [-1, 1]. Index and angle are drawn from a generator seeded by the start's seed.
--init randn: x0 ~ N(0, I) (MLGD-F's x_T), clamped by the first step.

A run (seed) is --n-restarts starts (default 1, as MLGD-F's Table 2 run is one trajectory); the start
with the smallest final SWD is the seed's answer, as search() selects MLGD-F restarts.

Budget, per start: 2 (steps + 1) sampler draws of n_cond with backprop + 3 scoring draws of 4 n_cond
(no backprop). MLGD-F: T n_MC draws of n_cond with backprop through f_phi and the UNet + one final
draw of 4 n_cond. With the defaults (201 steps) and Table 2's T, n_MC: unimodal 404 vs 390,
bimodal 404 vs 375, uniform 404 vs 900. The record stores the counts and the wall time per seed.

Output: one .pkl in --out with the layout of search_mnist.py (make_table.py and
figures/make_figures.py read it), plus "method": "GB-FESO" and per-start diagnostics.

    python search_gbfeso_mnist.py --target unimodal --n-cond 1500 --unimodal-var 515 --n-seeds 15 \
        --out outputs/gbfeso
"""

import argparse
import dataclasses
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.dirname(HERE)                                         # experiments/mnist
REPO = os.path.dirname(os.path.dirname(MNIST_DIR))
sys.path.insert(0, os.path.join(REPO, "experiments", "synthetic", "pipeline"))

from diffusers import DDIMScheduler                                       # noqa: E402
from cdm.guidance.core import _EvalView, _draw_conditional, _final_loss   # noqa: E402
from cdm.models.mnist.checkpoints import load_cond_model, make_noise_scheduler   # noqa: E402
from cdm.models.mnist.classifier import classify_generated_images, load_or_train_classifier  # noqa: E402
from cdm.models.mnist.guidance import build_config, make_swd_loss         # noqa: E402
from cdm.models.mnist.targets import make_target                          # noqa: E402
from search_gbfeso import run_phase                                       # noqa: E402

GLOBAL_SEED = 42
LO, HI = -1.0, 1.0                    # the prior's pixel range (train_uncond.py: x * 2 - 1)


def make_deterministic(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("problem (as search_mnist.py)")
    g.add_argument("--target", choices=["uniform", "bimodal", "unimodal"], required=True)
    g.add_argument("--n-cond", type=int, required=True, help="angle samples per step (config/targets.tsv)")
    g.add_argument("--unimodal-var", type=float, default=515.0)
    g.add_argument("--bimodal-var", type=float, default=300.0)
    g.add_argument("--final-factor", type=int, default=4, help="final SWD on factor * n_cond draws (MLGD-F: 4)")
    g = p.add_argument_group("optimizer (GB-FESO CLI defaults, as search_gbfeso.py)")
    g.add_argument("--steps", type=int, default=201, help="Adam steps per phase")
    g.add_argument("--lr", type=float, default=5e-2)
    g.add_argument("--lr-refine", type=float, default=5e-2)
    g.add_argument("--noise-std", type=float, default=0.05, help="phase-0 per-pixel noise std, decayed linearly to 0")
    g.add_argument("--phase1", choices=["always", "threshold", "never"], default="always")
    g.add_argument("--phase1-threshold", type=float, default=0.2)
    g.add_argument("--no-clamp", action="store_true", help="do not clamp x to [-1, 1]")
    g.add_argument("--init", choices=["data", "randn"], default="data",
                   help="data: a uniformly rotated MNIST training digit; randn: x0 ~ N(0, I)")
    g = p.add_argument_group("run control")
    g.add_argument("--n-restarts", type=int, default=1, help="starts per seed; the smallest final SWD wins")
    g.add_argument("--n-seeds", type=int, default=15)
    g.add_argument("--seed-start", type=int, default=0)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--ckpt-dir", default=os.path.join(MNIST_DIR, "checkpoints"))
    g.add_argument("--mnist-data-root", default=os.path.join(MNIST_DIR, "train", "outputs", "data"),
                   help="MNIST (downloaded if absent): the data starts, and the classifier if it has to be trained")
    g.add_argument("--classifier-path", default=None)
    g.add_argument("--threshold", type=float, default=0.8)
    g.add_argument("--no-classify", action="store_true")
    g.add_argument("--out", required=True, help="output directory")
    return p.parse_args()


class DataStarts:
    """Uniformly rotated MNIST training digits in [-1, 1], as RotatedMNISTDataset rotates them."""

    def __init__(self, root):
        from torchvision import datasets, transforms
        self.data = datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())

    def draw(self, seed, device):
        from torchvision.transforms import functional as TF
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(len(self.data)))
        angle = float(rng.uniform(0.0, 360.0))
        img, label = self.data[idx]
        img = TF.rotate(img.float(), angle, fill=0)
        return (img * 2 - 1).unsqueeze(0).to(device), {"index": idx, "label": int(label), "angle": angle}


def main():
    args = parse_args()
    make_deterministic(GLOBAL_SEED)
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    cuda = str(dev).startswith("cuda")

    target = make_target(args.target, unimodal_var=args.unimodal_var, bimodal_var=args.bimodal_var)
    sampler = load_cond_model(device=dev, local_dir=args.ckpt_dir)
    for prm in sampler.parameters():
        prm.requires_grad_(False)
    loss_fn = make_swd_loss(device=dev, n_projections=50)

    # MLGD-F's GuidanceConfig supplies the sampler plumbing and the final-SWD evaluation; its DDIM
    # schedule is never stepped (no prior). The step count only has to be valid.
    ddim = DDIMScheduler.from_config(make_noise_scheduler().config)
    ddim.set_timesteps(num_inference_steps=10)
    cfg = build_config(target, ddim, n_cond=args.n_cond, n_mc=1, step_size_mode="original", device=dev,
                       clamp=False, final_n_cond_factor=args.final_factor, tag="gbfeso")
    final_n = cfg.final_n_cond
    gen = target.generator()
    target_draw = lambda n: gen(n, dev)                                     # noqa: E731  fresh every step
    lo, hi = ((None, None) if args.no_clamp else
              (torch.tensor(LO, device=dev), torch.tensor(HI, device=dev)))
    starts = DataStarts(args.mnist_data_root) if args.init == "data" else None

    var_str = {"unimodal": f"_var{int(args.unimodal_var)}", "bimodal": f"_var{int(args.bimodal_var)}"}.get(args.target, "")
    run_name = (f"gbfeso_{args.target}{var_str}_ns{args.n_cond}_st{args.steps}_lr{args.lr:g}"
                f"_init{args.init}_R{args.n_restarts}_cl{0 if args.no_clamp else 1}")
    calls_per_start = 2 * (args.steps + 1) if args.phase1 == "always" else None
    print(f"[gbfeso-mnist] target={args.target} n_cond={args.n_cond} final_n={final_n} | {args.steps} steps x 2 phases "
          f"(phase 1: {args.phase1}) lr={args.lr}/{args.lr_refine} noise={args.noise_std} | clamp="
          f"{'off' if lo is None else f'[{LO:g}, {HI:g}]'} | init={args.init} | R={args.n_restarts} | device={dev}")

    def score(x, n):
        with torch.no_grad():
            return float(loss_fn(_draw_conditional(sampler, x, _EvalView(cfg, n)), target_draw(n)))

    results, loss_log, seed_log, time_log, restart_log, records = [], [], [], [], [], []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        if cuda:
            torch.cuda.synchronize()
        t_seed = time.time()
        runs = []
        for r in range(args.n_restarts):
            s = seed * args.n_restarts + r                      # MLGD-F's trajectory seeds
            torch.manual_seed(s)
            t0 = time.time()
            if starts is not None:
                x0, start_info = starts.draw(s, dev)
            else:
                x0, start_info = torch.randn(1, 1, 28, 28, device=dev), {}
            b0, l0, tr0 = run_phase(sampler, cfg, loss_fn, target_draw, x0, args.steps, args.lr, args.noise_std, lo, hi)
            n_calls = args.steps + 1
            do_p1 = args.phase1 == "always" or (args.phase1 == "threshold" and l0 > args.phase1_threshold)
            b1 = l1 = tr1 = None
            if do_p1:
                b1, l1, tr1 = run_phase(sampler, cfg, loss_fn, target_draw, b0, args.steps, args.lr_refine, 0.0, lo, hi)
                n_calls += args.steps + 1
            s0 = score(b0, final_n)
            s1 = score(b1, final_n) if do_p1 else float("inf")
            x_hat, phase = (b1, "phase1") if s1 < s0 else (b0, "phase0")
            final = float(_final_loss(sampler, loss_fn, cfg, x_hat))      # the reported SWD, as MLGD-F's
            n_score = 3 if do_p1 else 2
            if cuda:
                torch.cuda.synchronize()
            runs.append({"seed": s, "x": x_hat.detach().cpu(), "x0": x0.detach().cpu().numpy().squeeze(),
                         "start": start_info, "final_loss": final, "phase": phase,
                         "phase0_best_loss": l0, "phase1_best_loss": l1, "phase0_score": s0,
                         "phase1_score": s1 if do_p1 else None,
                         "trace": {"phase0": tr0, "phase1": tr1},
                         "sampler_calls_grad": n_calls, "sampler_calls_eval": n_score,
                         "sampler_samples": n_calls * args.n_cond + n_score * final_n,
                         "time_s": time.time() - t0})
            print(f"    start {r + 1}/{args.n_restarts} (seed {s}, {start_info or 'randn'}): swd {final:.4f} "
                  f"({phase}; phase scores {s0:.4f} / {s1:.4f})  {runs[-1]['time_s']:.1f}s", flush=True)
        best = min(runs, key=lambda q: q["final_loss"])
        elapsed = time.time() - t_seed
        results.append(best["x"].squeeze().numpy())
        loss_log.append(best["final_loss"])
        seed_log.append(seed)
        time_log.append(elapsed)
        restart_log.append([q["final_loss"] for q in runs])
        records.append([{k: v for k, v in q.items() if k != "x"} for q in runs])
        print(f"[seed {seed:2d}] swd={best['final_loss']:.4f}  time={elapsed:.1f}s  "
              f"sampler calls {sum(q['sampler_calls_grad'] + q['sampler_calls_eval'] for q in runs)}", flush=True)

    x_range, target_pdf = target.density_curve()
    payload = {
        "method": "GB-FESO",
        "experiment_name": run_name,
        "target": args.target,
        "nsamples": args.n_cond,
        "num_x_t": None,
        "num_inference_steps": None,
        "step_size_mode": None,
        "clamp": not args.no_clamp,
        "results": results,
        "loss_log": loss_log,
        "seed_log": seed_log,
        "time_log": time_log,
        "restart_losses": restart_log,
        "search": {"n_restarts": args.n_restarts},
        "gbfeso": {**{k: v for k, v in vars(args).items()}, "final_n": final_n, "calls_per_start_grad": calls_per_start,
                   "box": None if args.no_clamp else [LO, HI]},
        "restarts": records,                  # per seed, per start: start, phase scores, traces, calls, time
        "x_range": x_range,
        "target_pdf": target_pdf,
        "use_uniform": target.is_uniform,
    }
    if not args.no_classify:
        clf_path = args.classifier_path or os.path.join(args.out, "robust_classifier.pth")
        classifier = load_or_train_classifier(clf_path, device=dev, data_root=args.mnist_data_root)
        preds, confs = classify_generated_images(results, classifier, dev, threshold=args.threshold)
        payload["predicted_labels"] = preds
        payload["predicted_confidences"] = confs

    path = os.path.join(args.out, f"{run_name}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"[gbfeso-mnist] SWD {np.mean(loss_log):.4f} +- {np.std(loss_log):.4f} over {len(loss_log)} seeds, "
          f"time {np.mean(time_log):.1f}s per seed\nwrote {path}")


if __name__ == "__main__":
    main()
