"""Compatibility shim -- this module moved to samplers/consistency.py.

Kept so existing notebooks (`import ConsistencyModels`) and pickled checkpoints that
record this module path keep working. New code should import from samplers.consistency.
"""

import sys as _sys

from samplers.consistency import *  # noqa: F401,F403
from samplers import consistency as _moved

_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _moved.__dict__.items() if not k.startswith("__")}
)
