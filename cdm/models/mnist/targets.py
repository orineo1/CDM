"""The three angle targets G of the MNIST rotated-digits experiment.

Uniform on [0, 360), bimodal (equal Gaussian modes at 0 and 180 degrees) and
unimodal (a Gaussian at 0 degrees) -- the three rows of the paper's Table 2,
with the variances of Table 13.

Everything distributional here is the shared library:
  * mixture sampling  -> `utils.dist.generate_mog_samples_not_differentiable`
  * circular density  -> `cdm.losses.circular.mog_pdf_circular`
  * angle <-> (cos,sin) -> `cdm.losses.circular.angles_to_circular`
The source repo's `MNIST/run_mlgdf.py` carried its own copies of all three
(`generate_mog_samples`, `mog_pdf`, `angles_to_circular`); those copies are
not ported.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch

from cdm.losses.circular import angles_to_circular, mog_pdf_circular
from cdm.utils.dist import generate_mog_samples_not_differentiable

# Digit classes the source repo highlighted per target (console summary only).
UNIMODAL_DIGITS_OF_INTEREST = [2, 3, 5, 7]
BIMODAL_DIGITS_OF_INTEREST = [1, 6, 8, 9]

# Table 13 variances, in degrees^2.
UNIMODAL_VAR = 515.0
BIMODAL_VAR = 252.0


@dataclass
class AngleTarget:
    """A target distribution over the rotation angle, in degrees."""

    name: str
    mog_means: Optional[List[torch.Tensor]] = None
    mog_variances: Optional[List[torch.Tensor]] = None
    weights: Optional[torch.Tensor] = None
    digits_of_interest: List[int] = field(default_factory=lambda: list(range(10)))
    # The uniform target is the only one the source repo guides on a truncated
    # schedule: it zeroes the guidance push for t below this timestep.
    guidance_cutoff_t: Optional[int] = None

    @property
    def is_uniform(self) -> bool:
        return self.mog_means is None

    def sample_angles(self, n: int, device: str = "cpu") -> torch.Tensor:
        """n draws from G, in degrees.

        `device` applies to the uniform target only. The mixture draw is taken
        on the CPU generator because `MNIST/run_mlgdf.py` calls
        `generate_mog_samples(...)` without a device argument, i.e. on the CPU
        default; the loss moves both sample sets to the compute device anyway,
        so the only thing this preserves is which RNG stream is consumed.
        """
        if self.is_uniform:
            return torch.rand(n, device=device) * 360.0
        return generate_mog_samples_not_differentiable(
            n, self.mog_means, self.mog_variances, self.weights, device="cpu"
        ).squeeze()

    def generator(self) -> Callable[[int, str], torch.Tensor]:
        """`GuidanceConfig.target_generator`: (n, device) -> (n, 2) circular samples."""
        def target_generator(n, device):
            # float32 BEFORE cos/sin, as the source's angles_to_circular does
            # (`angles_deg.float()`); the mixture targets are sampled in float64.
            return angles_to_circular(self.sample_angles(n, device).float())
        return target_generator

    def density_curve(self, n_points: int = 200):
        """(angles_deg, density) for plotting, as stored in the run payload."""
        x_range = np.linspace(0.0, 360.0, n_points)
        if self.is_uniform:
            return x_range, np.ones(n_points) / 360.0
        pdf = mog_pdf_circular(
            torch.linspace(0, 360, n_points),
            [m.item() for m in self.mog_means],
            [v.squeeze().item() for v in self.mog_variances],
            self.weights,
        )
        return x_range, pdf.numpy()


def make_target(name: str, unimodal_var: float = UNIMODAL_VAR,
                bimodal_var: float = BIMODAL_VAR) -> AngleTarget:
    """Build one of "uniform", "bimodal", "unimodal"."""
    if name == "uniform":
        return AngleTarget(name="uniform", guidance_cutoff_t=250)
    if name == "bimodal":
        return AngleTarget(
            name="bimodal",
            mog_means=[torch.tensor([0], dtype=torch.float64),
                       torch.tensor([180], dtype=torch.float64)],
            mog_variances=[torch.tensor([[bimodal_var]], dtype=torch.float64),
                           torch.tensor([[bimodal_var]], dtype=torch.float64)],
            weights=torch.tensor([0.5, 0.5], dtype=torch.float64),
            digits_of_interest=BIMODAL_DIGITS_OF_INTEREST,
        )
    if name == "unimodal":
        return AngleTarget(
            name="unimodal",
            mog_means=[torch.tensor([360], dtype=torch.float64)],
            mog_variances=[torch.tensor([[unimodal_var]], dtype=torch.float64)],
            weights=torch.tensor([1.0], dtype=torch.float64),
            digits_of_interest=UNIMODAL_DIGITS_OF_INTEREST,
        )
    raise ValueError(f"unknown target: {name}")
