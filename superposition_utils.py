"""Compatibility shim -- this module moved to utils/superposition.py.

Kept so existing notebooks (`import superposition_utils`) and pickled checkpoints that
record this module path keep working. New code should import from utils.superposition.
"""

import sys as _sys

from utils.superposition import *  # noqa: F401,F403
from utils import superposition as _moved

_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _moved.__dict__.items() if not k.startswith("__")}
)
