"""Compatibility shim -- this module moved to samplers/flow_matching.py.

Kept so existing notebooks (`import FlowMatching`) and pickled checkpoints that
record this module path keep working. New code should import from samplers.flow_matching.
"""

import sys as _sys

from samplers.flow_matching import *  # noqa: F401,F403
from samplers import flow_matching as _moved

_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _moved.__dict__.items() if not k.startswith("__")}
)
