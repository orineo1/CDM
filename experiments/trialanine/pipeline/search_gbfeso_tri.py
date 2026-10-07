"""GB-FESO on the trialanine benchmark, on OUR conditional sampler (the same network MLGD uses).

Their optimizer loop as ported in experiments/synthetic/pipeline/search_gbfeso.py (run_phase: Adam on x through
the differentiated sampler, decaying perturbation noise in phase 0, clamping to the box, best iterate kept;
phase 1 from the phase-0 best; each phase's best scored on --final-n fresh samples). Their CLI defaults:
201 steps per phase, lr 0.05, 3,500 samples per step, KDE-KL bandwidth 0.5, kl_lambda 1, noise 0.05.
This driver only adds the trialanine inputs: a .npy target sample, the box [-B, B]^12 of the standardized
design, and the starts:
  --init zero      x0 = 0, i.e. the published force field ff14SB (one start)
  --init randn     x0 ~ N(0, I) truncated to the box: the design prior (the distribution of MLGD's x_T)
  --init json      the --init-top K restarts with the lowest (re-)scored loss in an MLGD result JSON
                   (the hybrid "MLGD + GB-FESO polish")
The run returns the start with the lowest score on --final-n fresh samples.
"""
import argparse, json, os, sys, time
from types import SimpleNamespace
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, "..", "..", "synthetic", "pipeline")
sys.path.insert(0, SYN)
from cdm.losses.functions import MMDLoss, RBF                        # noqa: E402
from cdm.losses.kde_kl import KDEKLLoss                              # noqa: E402
from cdm.guidance.core import _draw_conditional                     # noqa: E402
from run_optimization import load_model                              # noqa: E402
from search_gbfeso import run_phase                                  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sampler", required=True)
    p.add_argument("--target", required=True, help=".npy standardized target sample")
    p.add_argument("--box", type=float, default=2.0)
    p.add_argument("--loss", choices=["kl", "mmd"], default="kl")
    p.add_argument("--kde-bandwidth", type=float, default=0.5)
    p.add_argument("--kl-lambda", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=201)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--lr-refine", type=float, default=5e-2)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--phase1", choices=["always", "threshold", "never"], default="threshold",
                   help="their rule: phase 1 only when the best phase-0 KL > --phase1-threshold")
    p.add_argument("--phase1-threshold", type=float, default=0.2)
    p.add_argument("--num-samples", type=int, default=3500)
    p.add_argument("--final-n", type=int, default=3500)
    p.add_argument("--init", choices=["zero", "randn", "json"], default="zero")
    p.add_argument("--n-starts", type=int, default=1)
    p.add_argument("--init-json", default=None)
    p.add_argument("--init-top", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--method-name", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = a.device
    sampler, arch = load_model(a.sampler, dev)
    for prm in sampler.parameters():
        prm.requires_grad_(False)
    tgt = torch.as_tensor(np.load(a.target)).float().to(dev)
    target = lambda n: tgt[torch.randint(0, tgt.shape[0], (n,), device=dev)]      # noqa: E731
    c = 12
    lo, hi = -a.box * torch.ones(1, c, device=dev), a.box * torch.ones(1, c, device=dev)
    loss_fn = (KDEKLLoss(bandwidth=a.kde_bandwidth, kl_lambda=a.kl_lambda) if a.loss == "kl"
               else MMDLoss(kernel=RBF(n_kernels=5, mul_factor=2.0, device=dev)))
    cfg = lambda n: SimpleNamespace(n_cond=n, repeat_condition=True, device=dev, cond_sample_kwargs={},   # noqa: E731
                                    condition_on_slice=(arch == "diffusion"), cond_output_transform=None)
    torch.manual_seed(a.seed)
    if a.init == "zero":
        starts = [torch.zeros(1, c, device=dev)]
    elif a.init == "randn":
        starts = []
        for _ in range(a.n_starts):
            z = torch.randn(1, c, device=dev)
            while (z.abs() > a.box).any():
                z = torch.where(z.abs() > a.box, torch.randn_like(z), z)
            starts.append(z)
    else:
        J = json.load(open(a.init_json))
        rs = J["best"].get("restarts") or J["attempts"][0]["restarts"]
        key = lambda q: (0, q["select_loss"]) if q.get("select_loss") is not None else (1, q["final_loss"])  # noqa: E731
        rs = sorted(rs, key=key)[:a.init_top]
        starts = [torch.tensor(q["x_hat"], device=dev).view(1, c).float() for q in rs]
    runs, t_all = [], time.time()
    n_draw = 0
    for r, x0 in enumerate(starts):
        torch.manual_seed(a.seed * 1000 + r)
        t0 = time.time()
        b0, l0, tr0 = run_phase(sampler, cfg(a.num_samples), loss_fn, target, x0, a.steps, a.lr, a.noise_std, lo, hi)
        n_draw += (a.steps + 1) * a.num_samples
        do1 = a.phase1 == "always" or (a.phase1 == "threshold" and l0 > a.phase1_threshold)
        if do1:
            b1, l1, tr1 = run_phase(sampler, cfg(a.num_samples), loss_fn, target, b0, a.steps, a.lr_refine, 0.0, lo, hi)
            n_draw += (a.steps + 1) * a.num_samples
        with torch.no_grad():
            sc = lambda xx: float(loss_fn(_draw_conditional(sampler, xx, cfg(a.final_n)), target(a.final_n)))   # noqa: E731
            s0 = sc(b0); s1 = sc(b1) if do1 else float("inf")
            n_draw += a.final_n * (2 if do1 else 1)
        xh, score = (b1, s1) if s1 < s0 else (b0, s0)
        runs.append(dict(x0=x0.view(-1).tolist(), x_hat=xh.view(-1).tolist(), final_loss=score, phase1=bool(do1),
                         phase0_best=l0, phase1_best=l1 if do1 else None, time_s=time.time() - t0,
                         trace0=tr0[::10], trace1=(tr1[::10] if do1 else None)))
        print(f"  start {r + 1}/{len(starts)}: score {score:.5f} (phase1 {do1}) {time.time() - t0:.1f}s", flush=True)
    best = min(runs, key=lambda q: q["final_loss"])
    name = a.method_name or f"GBFESO-{a.loss.upper()}-{a.init}{len(starts)}"
    json.dump(dict(method=name, args=vars(a), best=dict(x_hat=best["x_hat"], final_loss=best["final_loss"]), restarts=runs,
                   time_s=time.time() - t_all, n_conditional_samples=n_draw,
                   device_name=(torch.cuda.get_device_name() if dev.startswith("cuda") else "cpu")),
              open(a.out, "w"), indent=1)
    print(f"[gbfeso_tri] {name}: best score {best['final_loss']:.5f}, {time.time() - t_all:.0f}s, {n_draw} samples -> {a.out}")


if __name__ == "__main__":
    main()
