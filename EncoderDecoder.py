"""Compatibility shim -- this module moved to utils/encoder_decoder.py.

Kept so existing notebooks (`import EncoderDecoder`) and pickled checkpoints that
record this module path keep working. New code should import from utils.encoder_decoder.
"""

import sys as _sys

from utils.encoder_decoder import *  # noqa: F401,F403
from utils import encoder_decoder as _moved

_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _moved.__dict__.items() if not k.startswith("__")}
)
