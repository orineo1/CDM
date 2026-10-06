"""Coordinate maps applied to both sample sets before a distributional loss.

Any `loss(pred, target)` of this package compares vectors with Euclidean geometry. Two properties of the
data can make that the wrong geometry, independently of the loss:

* periodic coordinates (angles, in radians): -pi and pi are the same point. `periodic_embedding` replaces
  each such coordinate t by (cos t, sin t). Any kernel or distance on the embedded vectors respects the
  period; the RBF kernel on (cos t, sin t) is the von Mises kernel exp(-(1 - cos(t - t')) / h^2), which is
  positive definite, characteristic on the circle (all its Fourier coefficients are positive) and smooth in
  t, so MMD keeps its properties and its gradients.
* coordinates on different scales (e.g. stiff bond lengths next to angles): one kernel bandwidth then sees
  only the widest coordinates. With `standardize`, each coordinate is divided by its spread in the target
  sample: the standard deviation for a Euclidean coordinate, and for a periodic one the standard deviation
  of its (cos, sin) pair, sqrt(1 - R^2) with R the mean resultant length (about the angular standard
  deviation for concentrated angles, 1 for uniform ones).

`FeatureLoss` wraps a loss with these maps; with `embed=False` periodic coordinates are passed through
unchanged (for losses that wrap angle differences themselves, such as `kde_kl.KDEKLLoss`) and only the
Euclidean ones are standardized.
"""
import torch


def _split(d, periodic_dims):
    per = sorted({int(i) % d for i in periodic_dims})
    return [i for i in range(d) if i not in per], per


def periodic_embedding(y, periodic_dims):
    """[..., d] -> [..., d + len(periodic_dims)]: the Euclidean coordinates, then cos and sin of the periodic ones."""
    if not periodic_dims:
        return y
    lin, per = _split(y.shape[-1], periodic_dims)
    a = y[..., per]
    return torch.cat([y[..., lin], torch.cos(a), torch.sin(a)], dim=-1)


class FeatureLoss:
    """`loss(pred, target)` computed on mapped coordinates: periodic embedding of `periodic_dims` (unless
    `embed=False`) and, with `standardize`, division by per-coordinate scales fitted on a target sample
    (`fit`; on the first call's target if `fit` was not called)."""

    def __init__(self, loss, periodic_dims=(), standardize=False, embed=True, min_scale=1e-6):
        self.loss, self.periodic_dims = loss, list(periodic_dims or ())
        self.standardize, self.embed, self.min_scale = standardize, embed, min_scale
        self.scale = None

    @torch.no_grad()
    def fit(self, target):
        """Per-coordinate scales of the mapped target sample [n, d]."""
        target = target.reshape(target.shape[0], -1).float()
        lin, per = _split(target.shape[-1], self.periodic_dims)
        s_lin = target[:, lin].std(dim=0)
        if self.embed:
            a = target[:, per]
            R2 = torch.cos(a).mean(0) ** 2 + torch.sin(a).mean(0) ** 2
            s_per = (1.0 - R2).clamp_min(0.0).sqrt()
            scale = torch.cat([s_lin, s_per, s_per])            # the order of periodic_embedding
        else:
            scale = torch.ones(target.shape[-1], device=target.device)
            scale[lin] = s_lin                                    # angles keep their period
        self.scale = scale.clamp_min(self.min_scale)
        return self

    def transform(self, y):
        z = periodic_embedding(y, self.periodic_dims) if self.embed else y
        if self.standardize:
            if self.scale is None:
                raise RuntimeError("FeatureLoss: call fit(target) before transform with standardize=True")
            z = z / self.scale.to(device=z.device, dtype=z.dtype)
        return z

    def __call__(self, pred, target):
        if self.standardize and self.scale is None:
            self.fit(target)
        return self.loss(self.transform(pred), self.transform(target))
