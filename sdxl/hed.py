"""
sdxl/hed.py -- HED scribble extraction.

SD-specific. The scribble that is optimised, and every baseline scribble, is an HED
edge map in scribble mode from `lllyasviel/Annotators` (the `controlnet_aux`
HEDdetector). Upstream calls this inline in scripts/eval_baselines.py and
scripts/run_mlgd_f.py @ cb8bc91; it is factored out here so both entry points and the
saliency analysis share one definition.
"""

import numpy as np
from PIL import Image

HED_ANNOTATOR_ID = "lllyasviel/Annotators"


def load_hed():
    """Load the HED detector. Requires `controlnet_aux` and a model download."""
    from controlnet_aux import HEDdetector
    return HEDdetector.from_pretrained(HED_ANNOTATOR_ID)


def extract_scribble(hed, image):
    """HED edge map in scribble mode, as a PIL image."""
    return hed(image, scribble=True)


def average_scribbles(scribbles, weights):
    """Weighted pixel-space average of HED scribbles, clipped to uint8."""
    arrs = [np.array(s).astype(np.float32) for s in scribbles]
    avg = sum(w * a for w, a in zip(weights, arrs)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(avg)
