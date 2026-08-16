"""Compatibility shim -- this module moved to priors/diffusion_general.py.

Kept so existing notebooks (`import DiffusionGeneral`) and pickled checkpoints that
record this module path keep working. New code should import from priors.diffusion_general.
"""

import sys as _sys

from priors.diffusion_general import *  # noqa: F401,F403
from priors import diffusion_general as _moved

_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _moved.__dict__.items() if not k.startswith("__")}
)
