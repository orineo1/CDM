"""Unconditional models over X: the score/diffusion prior P(X).

The classes themselves live one level up, in `models.diffusion` and
`models.diffusion_general`, because the same `DiffusionModel` also serves as the
conditional sampler P(Y|X) when built with `condition=True`.  This package is the
P(X) view of them.
"""

from .. import diffusion  # noqa: F401
from .. import diffusion_general  # noqa: F401

from ..diffusion import DiffusionModel  # noqa: F401
from ..diffusion_general import DiffusionModel as DiffusionModelCT  # noqa: F401
