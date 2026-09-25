"""Retrain the unconditional diffusion model over X (the DDIM backbone).

Needed only for end-to-end optimisation. Paper Table 3 (2D): Blocks 3,
Units 128, Epochs 20,000, Batch 1,024. (The notebook uses batch_size=512 for
the diffusion models; the paper says 1,024. Per decision D9 the paper wins.)

Trained on the X coordinate only, matching notebook cell 22's
``kernel_func=lambda X: X[:, :condition_on]``.
"""
import argparse, json, subprocess, sys, time
from functools import partial
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dist_utils                       # noqa: E402
from Diffusion import DiffusionModel    # noqa: E402
from tfg import oracle                  # noqa: E402

CONFIG = dict(nfeatures=1, nblocks=3, nunits=128, diffusion_steps=100,
              nepochs=20_000, batch_size=1_024)
SEED = 20240401


def train(seed=SEED, nepochs=None, device="cpu", out_dir=None):
    nepochs = CONFIG["nepochs"] if nepochs is None else nepochs
    torch.manual_seed(seed)
    p = oracle.load_params()
    gen = partial(dist_utils.generate_mog_samples_not_differentiable,
                  means=[m.float() for m in p["mu_list"]],
                  variances=[s.float() for s in p["Sigma_list"]],
                  weights=p["alpha"].float(),
                  kernel_func=lambda X: X[:, :1])
    m = DiffusionModel(nfeatures=CONFIG["nfeatures"], nblocks=CONFIG["nblocks"],
                       nunits=CONFIG["nunits"], condition=False,
                       diffusion_steps=CONFIG["diffusion_steps"],
                       device=torch.device(device))
    t0 = time.perf_counter()
    m.train_model(None, data_generator=gen, nepochs=nepochs,
                  batch_size=CONFIG["batch_size"], condition_on=1)
    secs = time.perf_counter() - t0
    m.eval()
    rec = {"seed": seed, "config": {**CONFIG, "nepochs": nepochs},
           "seconds": secs, "device": device,
           "provenance": "RE-TRAINED to paper Table 3; not the paper's weights",
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_ROOT),
                                        capture_output=True, text=True).stdout.strip()}
    if out_dir:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": m.state_dict(), "record": rec},
                   d / f"uncond_seed{seed}.pt")
        (d / "uncond_record.json").write_text(json.dumps(rec, indent=2))
        rec["checkpoint"] = str(d / f"uncond_seed{seed}.pt")
    return m, rec


def load(seed=SEED, out_dir="tfg_results/cm_checkpoints", device="cpu"):
    blob = torch.load(Path(out_dir) / f"uncond_seed{seed}.pt",
                      map_location=device, weights_only=False)
    m = DiffusionModel(nfeatures=CONFIG["nfeatures"], nblocks=CONFIG["nblocks"],
                       nunits=CONFIG["nunits"], condition=False,
                       diffusion_steps=CONFIG["diffusion_steps"],
                       device=torch.device(device))
    m.load_state_dict(blob["state_dict"]); m.eval()
    return m, blob["record"]


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--out-dir", default="tfg_results/cm_checkpoints")
    a.add_argument("--nepochs", type=int, default=CONFIG["nepochs"])
    args = a.parse_args()
    _, rec = train(nepochs=args.nepochs, out_dir=args.out_dir)
    print(f"done in {rec['seconds']:.1f}s -> {rec.get('checkpoint')}")
