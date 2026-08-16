"""Compatibility shim -- this module moved to utils/evaluation.py.

Kept so existing notebooks (`import evalModels`) and pickled checkpoints that
record this module path keep working. New code should import from utils.evaluation.
"""

import sys as _sys

from utils.evaluation import *  # noqa: F401,F403
from utils import evaluation as _moved

_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _moved.__dict__.items() if not k.startswith("__")}
)
