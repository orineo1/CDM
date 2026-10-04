"""
GB-FESO's trained surrogate (their AngleDiffusion from their train.py, `gbfeso.py train`) exposed as a conditional
sampler for OUR search, so that MLGD can be run on their surrogate ("MLGD, their surrogate"): the comparison of
search methods on one and the same surrogate.

    sampler = GBFESOSampler("runs/peptide/models/gbfeso.pt", "runs/peptide/meta.json", "cuda")
    out = sampler.sample(nsamples, condition_x, device)     # [n, K + 3N - 5] = [x, y]

condition_x: [n, K] (or [K]) standardized eps, our search variable; y = our standardized features
(common.YScaler of meta.json) of their DDIM draw at eps = x * X_STD + X_MEAN, drawn exactly as GB-FESO draws
during its optimization (DDIM, 40 steps, eta 0, no clamping, their de-normalization). The output is the joint
[x, y] with `condition_on` = K, like our conditional diffusion checkpoints (search_mog slices it). The draw is
differentiable in condition_x (pathwise gradients); with --guide-grad zo only forward passes are used.
search_peptide.py loads it when --sampler points at a GB-FESO checkpoint (meta.json: <ckpt dir>/../meta.json,
or $PEPTIDE_META).
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common   # noqa: E402
from simulate import Design   # noqa: E402


def is_gbfeso_checkpoint(path):
    if not str(path).endswith(".pt") or not os.path.isfile(path):
        return False
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return isinstance(ck, dict) and "model_state_dict" in ck and "eps_min" in ck


class GBFESOSampler(torch.nn.Module):
    def __init__(self, ckpt, meta=None, device="cuda", sample_steps=40):
        super().__init__()
        import gbfeso
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        design = Design.from_dict(ck.get("peptide_design"))
        dummy = [{"target": torch.zeros(1, design.dim_y), "eps": torch.full((design.dim_x,), 0.5)}]
        self.model, self.frame_ds, _, _ = gbfeso.load_model(ckpt, dummy, device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        meta = meta or os.environ.get("PEPTIDE_META") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))), "meta.json")
        mj = json.load(open(meta))
        self.scaler = common.YScaler(torch.tensor(mj["y_mean"], device=device), torch.tensor(mj["y_std"], device=device))
        self.condition_on, self.steps, self.dev, self.design = design.dim_x, sample_steps, device, design
        print(f"  Loaded GB-FESO surrogate from {ckpt} as a conditional sampler ({design}, {sample_steps} DDIM steps)")

    def sample(self, nsamples, condition_x, device=None, **kw):
        x = torch.as_tensor(condition_x, dtype=torch.float32).to(self.dev)
        x = x.view(1, -1).expand(nsamples, -1) if x.dim() == 1 or x.shape[0] == 1 else x
        assert x.shape == (nsamples, self.condition_on), (tuple(x.shape), nsamples)
        eps = x * common.X_STD + common.X_MEAN
        z = self.model.ddim_sample(eps, shape=(nsamples, self.frame_ds.target_dim), sample_steps=self.steps,
                                   eta=0.0, clamp_x0=False)
        y = self.scaler.fwd(self.frame_ds.denormalize_target(z, clamp=False))
        return torch.cat([x, y], 1)
