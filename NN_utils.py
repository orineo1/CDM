"""Compatibility shim -- this module moved to utils/nn.py.

Kept so existing notebooks (`import NN_utils`) and pickled checkpoints that
record this module path keep working. New code should import from utils.nn.
"""

import sys as _sys

from utils.nn import *  # noqa: F401,F403
from utils import nn as _moved

_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _moved.__dict__.items() if not k.startswith("__")}
)
