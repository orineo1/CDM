"""Label-free COUNT statistic used as the loss feature of the trialanine CDM benchmark.

Samples y (standardized residue-sorted features, make_data_tri.py) are mapped to the soft number of residues in
the phi ~ +60 basin, c(y) = sum_i sigmoid(kappa * sin(phi_i)), phi_i = atan2(sin, cos) of the unstandardized
features (sin phi ~ +0.87 in the '+' basin, -0.94 / -0.5 in the alpha / beta parts of the '-' basin; kappa = 8
gives c within 0.01 of the integer count for basin frames). The distributional loss (KDE-KL or MMD) is then
computed between the 1-D samples c(pred) and c(target): it depends only on the count distribution, so designs
that put the '+' population on different residues are equally good (the 3-mode case).
"""
import json
import torch


class CountFeature:
    def __init__(self, y_mean, y_std, kappa=8.0):
        self.m = torch.as_tensor(y_mean, dtype=torch.float32)
        self.s = torch.as_tensor(y_std, dtype=torch.float32)
        self.kappa = kappa

    @classmethod
    def from_meta(cls, path, kappa=8.0):
        meta = json.load(open(path))
        return cls(meta["y_mean"], meta["y_std"], kappa)

    def __call__(self, y):
        y = y * self.s.to(y.device, y.dtype) + self.m.to(y.device, y.dtype)
        c, s = y[..., 0::4], y[..., 1::4]                      # cos phi, sin phi of the three slots
        sinphi = s / torch.sqrt(c ** 2 + s ** 2 + 1e-8)
        return torch.sigmoid(self.kappa * sinphi).sum(-1, keepdim=True)


class CountLoss:
    """loss(c(pred), c(target)) for any base loss(pred, target) of the package."""

    def __init__(self, base, feature):
        self.base, self.f = base, feature

    def __call__(self, pred, target):
        return self.base(self.f(pred), self.f(target))
