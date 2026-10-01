"""
Shared helpers of the peptide experiment.

* Imports Zick & Mendels' GB-FESO code (github.com/.../GB-FESO, MIT), cloned next to this repository
  (`<parent of CDM>/GB-FESO`, or $GBFESO_DIR). Their modules import two names that their dataset.py no
  longer defines (`LammpsTrajectoryAngleDataset`, `AngleFrameDataset`, renamed to *Geometry*); we alias them
  so that train.py, run_feso.py and plotters.py import unchanged.
* The Y representation of OUR networks: 7 features (r12, r23, r34, cos th123, sin th123, cos th234,
  sin th234), standardized, so that Euclidean MMD treats the angles periodically. X = (eps1, eps2),
  standardized. GB-FESO keeps its own representation (raw 5D, angles / pi).
"""
import math
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GBFESO_DIR = os.environ.get("GBFESO_DIR", os.path.join(os.path.dirname(REPO), "GB-FESO", "GB-FESO"))
if GBFESO_DIR not in sys.path:
    sys.path.insert(0, GBFESO_DIR)
import dataset as _gb_dataset   # noqa: E402
_gb_dataset.AngleFrameDataset = _gb_dataset.GeometryFrameDataset
_gb_dataset.LammpsTrajectoryAngleDataset = _gb_dataset.LammpsTrajectoryGeometryDataset

# X standardization: uniform box [0.21, 1.55]
X_MEAN, X_STD = 0.88, 1.34 / math.sqrt(12.0)


def raw_to_feat(Y):
    """[..., 5] raw -> [..., 7] unstandardized features."""
    th1, th2 = Y[..., 3], Y[..., 4]
    return torch.cat([Y[..., :3], torch.stack([th1.cos(), th1.sin(), th2.cos(), th2.sin()], -1)], -1)


def feat_to_raw(F):
    """[..., 7] unstandardized features -> [..., 5] raw (angles by atan2)."""
    return torch.cat([F[..., :3], torch.stack([torch.atan2(F[..., 4], F[..., 3]),
                                               torch.atan2(F[..., 6], F[..., 5])], -1)], -1)


def x_to_std(eps):
    return (torch.as_tensor(eps, dtype=torch.float32) - X_MEAN) / X_STD


def x_from_std(x):
    return torch.as_tensor(x, dtype=torch.float32) * X_STD + X_MEAN


class YScaler:
    def __init__(self, mean, std):
        self.mean, self.std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()

    def fwd(self, Y_raw):          # raw 5D -> standardized 7D
        return (raw_to_feat(Y_raw) - self.mean) / self.std

    def inv(self, Z):              # standardized 7D -> raw 5D
        return feat_to_raw(Z * self.std.to(Z.device) + self.mean.to(Z.device))
