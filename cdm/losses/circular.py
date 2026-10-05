"""Circular (angle) representation helpers.

Promoted verbatim from mnistExpThesisFinal.ipynb cell 7. The MNIST rotation
task represents an angle theta as (cos theta, sin theta) so that the
distributional loss is computed in a space where 0 and 360 degrees coincide.
"""

import torch


def circular_to_angles(circular):
    """Convert circular representation back to angles in degrees"""
    angles_rad = torch.atan2(circular[..., 1], circular[..., 0])
    return torch.rad2deg(angles_rad) % 360


def angles_to_circular(angles_deg):
    """Convert angles in degrees to circular representation (cos, sin)"""
    angles_rad = torch.deg2rad(angles_deg)
    return torch.stack([torch.cos(angles_rad), torch.sin(angles_rad)], dim=-1)


def mog_pdf_circular(x, means, variances, weights=None):
    """MoG density on the circle, in degrees.

    IMPORTANT: this is NOT the same function as `utils.dist.mog_pdf`, which
    computes the ordinary (linear) Gaussian mixture density. This version
    wraps the difference to the shortest signed angle,
        diff = (x - mean + 180) % 360 - 180,
    so that 359 deg and 1 deg are close. Both are kept, under distinct names.
    """
    components = len(means)
    if weights is None:
        weights = torch.ones(components) / components

    pdf = torch.zeros_like(x)
    for mean, var, weight in zip(means, variances, weights):
        var_tensor = torch.tensor(var, dtype=torch.float32)
        mean_tensor = torch.tensor(mean, dtype=torch.float32)

        # Handle circular distance (shortest angle between x and mean)
        diff = (x - mean_tensor + 180) % 360 - 180

        pdf += weight * torch.exp(-0.5 * (diff**2) / var_tensor) / (torch.sqrt(2 * torch.pi * var_tensor))
    return pdf


def create_mog_pdf_evaluator(mog_means, mog_variances, weights):
    """Closure evaluating the circular MoG PDF at given angles (degrees).

    Promoted from mnistExpThesisFinal.ipynb cell 25.

    NOTE: with the support penalty disabled in SupportAwareSlicedWasserstein-
    Loss, the returned values are computed and then discarded -- the evaluator
    affects runtime but not the loss. Kept for faithful reproduction.
    """
    def evaluate_pdf(angles):
        return mog_pdf_circular(
            angles,
            [mean.item() for mean in mog_means],
            [var.squeeze().item() for var in mog_variances],
            weights,
        )
    return evaluate_pdf
