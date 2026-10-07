"""Accuracy of the trained conditional sampler against the exact (reweighted) ensembles: at every target x*
and at --n-random prior designs, the label-free count distribution decoded from --n sampler draws
(phi = atan2 of the unstandardized sin / cos features) vs the exact one. Writes <out>/sampler_check.json."""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "synthetic", "pipeline"))
import system as S
from rw import Frames
from run_optimization import load_model
from cdm.guidance.core import _draw_conditional
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument("--root", default="runs/trialanine/cdm")
p.add_argument("--frames", default="runs/trialanine/prod_x0_w*.npz")
p.add_argument("--n", type=int, default=5000)
p.add_argument("--n-random", type=int, default=20)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--out", default="experiments/trialanine/results/cdm")
a = p.parse_args()
t0 = time.time()
meta = json.load(open(os.path.join(a.root, "meta.json")))
F = Frames(glob.glob(a.frames), thin=4)
NPLUS = np.eye(4)[np.array([bin(k).count("1") for k in range(8)])]
sampler, arch = load_model(os.path.join(a.root, "models", "diff", "model_conditional.pt"), a.device)
cfg = SimpleNamespace(n_cond=a.n, repeat_condition=True, device=a.device, cond_sample_kwargs={},
                      condition_on_slice=(arch == "diffusion"), cond_output_transform=None)
ym, ys = np.array(meta["y_mean"]), np.array(meta["y_std"])


def sampler_counts(x_std):
    with torch.no_grad():
        y = _draw_conditional(sampler, torch.tensor(x_std, dtype=torch.float32, device=a.device).view(1, -1), cfg)
    y = y.cpu().numpy().reshape(-1, 12) * ys + ym
    phi = np.arctan2(y[:, 1::4], y[:, 0::4])
    return np.bincount(S.count_plus(phi), minlength=4) / len(phi)


rows = []
rng = np.random.default_rng(7)
pts = [("target", t["id"], np.array(t["x_star_std"])) for t in meta["targets"] if t["id"] is not None]
pts += [("prior", k, np.clip(rng.normal(size=12), -meta["box_std"], meta["box_std"])) for k in range(a.n_random)]
for kind, i, xs in pts:
    ex = F.pops(xs * meta["x_scale_kJ"]) @ NPLUS
    sm = sampler_counts(xs)
    rows.append(dict(kind=kind, id=i, exact=ex.round(4).tolist(), sampler=sm.round(4).tolist(), tv=float(0.5 * np.abs(ex - sm).sum())))
    print(kind, i, "TV %.3f" % rows[-1]["tv"], np.round(ex, 3), np.round(sm, 3), flush=True)
tv = {k: [r["tv"] for r in rows if r["kind"] == k] for k in ("target", "prior")}
res = dict(rows=rows, median_tv={k: float(np.median(v)) for k, v in tv.items()}, max_tv={k: float(np.max(v)) for k, v in tv.items()},
           n=a.n, wall_s=time.time() - t0, device=a.device)
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(os.path.join(a.out, "sampler_check.json"), "w"), indent=1)
print("median TV", res["median_tv"], "max", res["max_tv"])
