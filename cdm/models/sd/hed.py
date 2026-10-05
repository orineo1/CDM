"""
cdm/models/sd/hed.py -- HED scribble extraction.

SD-specific. The scribble that is optimised, and every baseline scribble, is an HED
edge map in scribble mode from `lllyasviel/Annotators` (the `controlnet_aux`
HEDdetector), shared by both entry points and the saliency analysis.
"""

import numpy as np
from PIL import Image

HED_ANNOTATOR_ID = "lllyasviel/Annotators"


HED_FILENAME = "ControlNetHED.pth"   # what HEDdetector.from_pretrained loads by default


def load_hed(revision=None):
    """Load the HED detector. Requires `controlnet_aux` and a model download.

    `revision` pins the hub commit (None = main). HEDdetector.from_pretrained has no
    revision argument, but it accepts a local directory: download the weights file at that
    commit into the HF cache and hand it the file's snapshot directory.
    """
    from controlnet_aux import HEDdetector
    if not revision:
        return HEDdetector.from_pretrained(HED_ANNOTATOR_ID)
    import os
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HED_ANNOTATOR_ID, HED_FILENAME, revision=revision)
    return HEDdetector.from_pretrained(os.path.dirname(path), filename=HED_FILENAME)


def extract_scribble(hed, image):
    """HED edge map in scribble mode, as a PIL image."""
    return hed(image, scribble=True)


def average_scribbles(scribbles, weights):
    """Weighted pixel-space average of HED scribbles, clipped to uint8."""
    arrs = [np.array(s).astype(np.float32) for s in scribbles]
    avg = sum(w * a for w, a in zip(weights, arrs)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(avg)
