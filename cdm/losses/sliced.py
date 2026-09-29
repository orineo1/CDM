"""Sliced-Wasserstein loss used by the MNIST rotation experiment.

Promoted verbatim from mnistExpThesisFinal.ipynb cell 23.

NAMING WARNING -- read before using. Despite the class name and the live
`support_weight` constructor argument, this computes **plain SWD**: the
support/coverage penalty is commented out in `forward` (`total_loss = swd
#+ ...`). `support_weight` is inert. The paper describes the MNIST objective
as SWD, so the *computation* matches the paper; only the name misleads.
Kept verbatim so the MNIST results stay reproducible -- do not rename or
re-enable the penalty without re-running that experiment.

Note also that `sliced_wasserstein_distance` uses mean |sorted difference|,
i.e. sliced W_1, and requires X and Y to have equal sample counts.
"""

import torch
from torch import nn


class SupportAwareSlicedWassersteinLoss(nn.Module):
    def __init__(self, device='cpu', support_weight=10.0, pdf_threshold=1e-6,
                 n_projections=50):
        super().__init__()
        self.device = device
        self.support_weight = support_weight
        self.pdf_threshold = pdf_threshold
        self.n_projections = n_projections
        self.to(device)

    def sliced_wasserstein_distance(self, X, Y, n_projections=50):
        """
        Compute Sliced Wasserstein Distance between X and Y
        X, Y: [batch_size, features]
        """
        # Ensure X and Y are on the correct device and dtype
        X = X.to(self.device).float()
        Y = Y.to(self.device).float()

        dim = X.shape[1]

        # Generate random projections on the correct device with matching dtype
        projections = torch.randn(n_projections, dim, device=self.device, dtype=torch.float32)
        projections = projections / torch.norm(projections, dim=1, keepdim=True)

        # Project samples onto random directions
        X_projections = X @ projections.T  # [batch_size, n_projections]
        Y_projections = Y @ projections.T  # [batch_size, n_projections]

        # Sort projections
        X_sorted = torch.sort(X_projections, dim=0)[0]
        Y_sorted = torch.sort(Y_projections, dim=0)[0]

        # Compute Wasserstein distance for each projection (L2 distance between sorted samples)
        wasserstein_distances = torch.mean(torch.abs(X_sorted - Y_sorted), dim=0)

        # Average over all projections
        return torch.mean(wasserstein_distances)

    def forward(self, X, Y, X_angles, target_pdf_func):
        """
        X: target circular samples [batch_size, 2] (cos, sin)
        Y: reference circular samples [batch_size, 2]
        X_angles: angles in [0, 360] corresponding to X (for PDF evaluation)
        target_pdf_func: function that computes PDF values for angles
        """
        # Ensure all inputs are on the correct device and dtype
        X = X.to(self.device).float()
        Y = Y.to(self.device).float()
        X_angles = X_angles.to(self.device).float()

        # Sliced Wasserstein Distance component
        swd = self.sliced_wasserstein_distance(X, Y, n_projections=self.n_projections)

        # Evaluate PDF at sample locations
        pdf_values = target_pdf_func(X_angles)

        # Penalize samples where PDF is below threshold
        support_violation = torch.relu(self.pdf_threshold - pdf_values).sum()

        # Penalize proportion of samples in invalid regions
        valid_ratio = (pdf_values > self.pdf_threshold).float().mean()
        coverage_penalty = 1.0 - valid_ratio

        total_loss = swd #+ self.support_weight * (support_violation + coverage_penalty)

        return total_loss, {
            'swd': swd.item(),
            # 'support_violation': support_violation.item(),
            # 'coverage_penalty': coverage_penalty.item(),
            # 'valid_ratio': valid_ratio.item()
        }
