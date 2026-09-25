"""Unconditional models over X: the score/diffusion prior P(X).

The class itself lives one level up, in `models.diffusion`, because the same
`DiffusionModel` also serves as the conditional sampler P(Y|X) when built with
`condition=True`.  This package is the P(X) view of it.
"""

from .. import diffusion  # noqa: F401

from ..diffusion import DiffusionModel  # noqa: F401
