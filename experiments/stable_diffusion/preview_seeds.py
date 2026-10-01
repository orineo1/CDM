"""5-portrait previews of every scribble of some runs, for several seeds.

Same generation as the eval's preview_<method>.png (neutral prompt, ControlNet 0.5, the
eval's gen_images), so --seeds 42 reproduces those. Writes
<run>/previews/seed<S>/preview_<method>.png.

    python experiments/stable_diffusion/preview_seeds.py --seeds 7 123 2026 --run_dirs <run> [<run> ...]
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image

from cdm.models.sd.eval_all_experiments import N_PREVIEW, NEUTRAL_PROMPT, find_scribbles, gen_images
from cdm.models.sd.models import load_models, make_deterministic
from cdm.models.sd.visualization import plot_row

MODELS_ENV = Path(__file__).parent / "four_targets" / "config" / "models.env"


def pinned_revisions():
    out = {}
    for line in MODELS_ENV.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip().lower()] = v.split("#")[0].strip()
    return out


p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--run_dirs", nargs="+", required=True)
p.add_argument("--seeds", type=int, nargs="+", default=[7, 123, 2026])
p.add_argument("--n", type=int, default=N_PREVIEW)
p.add_argument("--controlnet_scale", type=float, default=0.5)
args = p.parse_args()

make_deterministic(0)
rev = pinned_revisions()
architect, sprinter = load_models("cuda", architect_revision=rev.get("architect_revision"),
                                  sprinter_revision=rev.get("sprinter_revision"),
                                  controlnet_revision=rev.get("controlnet_revision"))
del architect
torch.cuda.empty_cache()

for run in args.run_dirs:
    scribbles = find_scribbles(run, [])
    print(f"{os.path.basename(os.path.normpath(run))}: {', '.join(scribbles)}", flush=True)
    for seed in args.seeds:
        out = os.path.join(run, "previews", f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        for name, path in scribbles.items():
            imgs = gen_images(sprinter, Image.open(path).convert("RGB"), NEUTRAL_PROMPT, args.n,
                              args.controlnet_scale, seed)
            plot_row(imgs, f"{name}  (seed {seed})", count=args.n,
                     save_path=os.path.join(out, f"preview_{name}.png"))
        print(f"  seed {seed} -> {out}", flush=True)
