"""Compatibility shim -- this module moved to samplers/fm_solver.py.

Kept so existing notebooks (`import FM_Solver_Extension_module`) and pickled checkpoints that
record this module path keep working. New code should import from samplers.fm_solver.
"""

import sys as _sys

from samplers.fm_solver import *  # noqa: F401,F403
from samplers import fm_solver as _moved

_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _moved.__dict__.items() if not k.startswith("__")}
)
