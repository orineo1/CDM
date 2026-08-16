"""Compatibility shim -- this module moved to priors/diffusion.py.

Kept so existing notebooks (`import Diffusion`) and pickled checkpoints that
record this module path keep working. New code should import from priors.diffusion.
"""

import sys as _sys

from priors.diffusion import *  # noqa: F401,F403
from priors import diffusion as _moved

_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _moved.__dict__.items() if not k.startswith("__")}
)
