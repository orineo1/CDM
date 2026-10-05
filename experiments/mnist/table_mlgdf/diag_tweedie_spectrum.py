#!/usr/bin/env python
"""Spectrum of the Tweedie Jacobian J = d x0_hat / d x_t of the MNIST prior along unguided trajectories.

Tangent zeroth-order guidance (`--zo-dirs-mode tangent --zo-dirs M`) is exact when M directions span the
range of J, and cheap when few do. This measures how many: at a few reverse steps of plain DDIM
trajectories it sketches J with a randomized range finder (Y = J Omega by central differences of the
Tweedie map, Q = orth(Y), B = Q^T J by M vector-Jacobian products), takes the singular values of B (the
top-M singular values of J) and compares their energy with ||J||_F^2 (Hutchinson estimate from Y). It
reports, per step, the singular values and the number of directions holding 90% and 99% of ||J||_F^2.

    python diag_tweedie_spectrum.py --steps 125 --at 0.9 0.6 0.3 0.1 0.03 --m 100 --n-traj 3 --out spectrum.json
"""
import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.dirname(HERE)

from diffusers import DDIMScheduler                                          # noqa: E402
from types import SimpleNamespace                                             # noqa: E402
from cdm.guidance.core import _tweedie_jvp, _orthonormal_rows                 # noqa: E402
from cdm.models.mnist.checkpoints import load_uncond_model, make_noise_scheduler   # noqa: E402
from cdm.models.mnist.guidance import make_ddim_step_fn                       # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=125, help="DDIM steps (bimodal target of Table 2: 125)")
    p.add_argument("--at", type=float, nargs="+", default=[0.9, 0.6, 0.3, 0.1, 0.03],
                   help="fractions of the trajectory (1 = start, pure noise) at which J is measured")
    p.add_argument("--m", type=int, default=100, help="sketch size")
    p.add_argument("--n-traj", type=int, default=3)
    p.add_argument("--jvp-delta", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt-dir", default=os.path.join(MNIST_DIR, "checkpoints"))
    p.add_argument("--allow-untrained-prior", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    model = load_uncond_model(device=dev, local_dir=a.ckpt_dir, allow_untrained=a.allow_untrained_prior)
    ddim = DDIMScheduler.from_config(make_noise_scheduler().config)
    ddim.set_timesteps(num_inference_steps=a.steps)
    ts = [int(t) for t in ddim.timesteps]
    step = make_ddim_step_fn(ddim, ts)
    cfg = SimpleNamespace(ddim_step_fn=step, zo_jvp_delta=a.jvp_delta)
    n = len(ts) - 1
    at_idx = sorted({min(n - 1, max(0, int(round((1 - f) * n)))) for f in a.at})
    rows = []
    for k in range(a.n_traj):
        x = torch.randn(1, 1, 28, 28, device=dev)
        for i, t in enumerate(ts[:-1]):
            if i in at_idx:
                d = x.numel()
                Om = torch.randn(a.m, d, device=dev)
                Y = _tweedie_jvp(model, x, t, i, cfg, Om)                       # [m, d] rows J omega
                fro2 = float((Y ** 2).sum() / a.m)                              # Hutchinson: E|J w|^2 = ||J||_F^2
                Q = _orthonormal_rows(Y.float())                                # [m, d] basis of range(J Omega)
                B = []
                for q in Q:                                                     # rows q^T J (vector-Jacobian products)
                    xi = x.detach().clone().requires_grad_(True)
                    x0 = step(model, xi, t, i)[1]
                    B.append(torch.autograd.grad((x0.view(-1) * q).sum(), xi)[0].view(-1))
                s = torch.linalg.svdvals(torch.stack(B)).cpu()
                cum = torch.cumsum(s ** 2, 0) / max(fro2, 1e-12)
                r90 = int((cum < 0.90).sum()) + 1 if cum[-1] >= 0.90 else None
                r99 = int((cum < 0.99).sum()) + 1 if cum[-1] >= 0.99 else None
                rows.append({"traj": k, "i": i, "t": t, "frac_of_trajectory": 1 - i / n,
                             "abar_t": float(ddim.alphas_cumprod[t]), "fro2": fro2,
                             "captured_by_m": float(cum[-1]), "rank90": r90, "rank99": r99,
                             "sv": s.tolist()})
                print(f"traj {k} step {i:3d} (t={t:4d}, abar={rows[-1]['abar_t']:.3f}): top-{a.m} hold "
                      f"{100 * float(cum[-1]):.1f}% of ||J||_F^2; 90% in {r90}, 99% in {r99} directions; "
                      f"sv[0,4,9,29,99] = {[round(float(s[j]), 3) for j in (0, 4, 9, 29, a.m - 1) if j < len(s)]}",
                      flush=True)
            with torch.no_grad():
                x = step(model, x, t, i)[0]
    json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
