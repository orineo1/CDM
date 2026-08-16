"""Compatibility shim -- this module moved to losses/functions.py.

Kept so existing notebooks (`import LossFunctions`) and pickled checkpoints that
record this module path keep working. New code should import from losses.functions.
"""

import sys as _sys

from losses.functions import *  # noqa: F401,F403
from losses import functions as _moved

_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _moved.__dict__.items() if not k.startswith("__")}
)
