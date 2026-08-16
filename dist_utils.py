"""Compatibility shim -- this module moved to utils/dist.py.

Kept so existing notebooks (`import dist_utils`) and pickled checkpoints that
record this module path keep working. New code should import from utils.dist.
"""

import sys as _sys

from utils.dist import *  # noqa: F401,F403
from utils import dist as _moved

_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _moved.__dict__.items() if not k.startswith("__")}
)
