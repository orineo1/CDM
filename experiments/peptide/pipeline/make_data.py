"""
Convert simulated ensembles into the inputs of our pipeline (experiments/synthetic/pipeline/train_models.py,
find_xstar.py):

  <out>/data.pt          joint rows [x(2), y(7)] for train_models.py --model conditional (condition_on = 2)
  <out>/prior_data.pt    fresh uniform draws of eps in the box, for train_models.py --model unconditional
  <out>/targets/target_XXX.npy   standardized 7D target samples for find_xstar.py --target
  <out>/meta.json        standardization constants and the target eps*

x = (eps - 0.88) / 0.387 (the uniform box standardized); y = standardized
(r12, r23, r34, cos th123, sin th123, cos th234, sin th234).

    python experiments/peptide/pipeline/make_data.py --train runs/peptide/train_ens.pt \
        --targets runs/peptide/targets_ens.pt --frames-per-eps 4000 --out runs/peptide
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import raw_to_feat, x_to_std, X_MEAN, X_STD, YScaler   # noqa: E402
from simulate import EPS_BOX   # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--frames-per-eps", type=int, default=4000)
    p.add_argument("--n-prior", type=int, default=200_000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    g = torch.Generator().manual_seed(a.seed)
    tr = torch.load(a.train, weights_only=False)
    Y = tr["Y"][:, :a.frames_per_eps]                     # frames are already a random subset
    E, K, _ = Y.shape
    feat = raw_to_feat(Y.reshape(-1, 5))
    sc = YScaler(feat.mean(0), feat.std(0))
    x = x_to_std(tr["eps"]).repeat_interleave(K, dim=0)
    samples = torch.cat([x, sc.fwd(Y.reshape(-1, 5))], 1)
    os.makedirs(os.path.join(a.out, "targets"), exist_ok=True)
    torch.save({"samples": samples, "condition_on": 2, "y_mean": sc.mean, "y_std": sc.std,
                "x_mean": X_MEAN, "x_std": X_STD, "eps": tr["eps"]}, os.path.join(a.out, "data.pt"))
    lo, hi = EPS_BOX
    prior = x_to_std(lo + (hi - lo) * torch.rand(a.n_prior, 2, generator=g))
    torch.save({"samples": prior, "condition_on": 2}, os.path.join(a.out, "prior_data.pt"))
    te = torch.load(a.targets, weights_only=False)
    for i in range(te["eps"].shape[0]):
        np.save(os.path.join(a.out, "targets", f"target_{i:03d}.npy"), sc.fwd(te["Y"][i]).numpy())
    json.dump({"x_mean": X_MEAN, "x_std": X_STD, "y_mean": sc.mean.tolist(), "y_std": sc.std.tolist(),
               "n_train_eps": E, "frames_per_eps": K, "eps_star": te["eps"].tolist(),
               "train_protocol": tr["protocol"], "target_protocol": te["protocol"]},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"[make_data] data.pt {tuple(samples.shape)}, prior_data.pt {tuple(prior.shape)}, "
          f"{te['eps'].shape[0]} targets -> {a.out}")


if __name__ == "__main__":
    main()
