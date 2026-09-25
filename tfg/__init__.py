"""Generalised training-free guidance (TFG) for conditional distribution matching.

Modules
-------
``noise_tape``     Semantically-keyed RNG; makes two engines comparable.
``schedule``       float64 cosine diffusion schedule and rho/mu structures.
``n_schedule``     Component 1: adaptive conditional sample count n_t.
``config``         Configuration dataclasses; every extension off by default.
``reference_tfg``  Frozen transcription of TFG Algorithm 1 (test-only, NEW code).
``engine``         Generalised engine with switchable extensions.
``trace``          Passive intermediate-state recorder and trace comparison.
``gmm_l2``         Exact closed-form L2 distance between Gaussian mixtures.

Nothing here imports or modifies the legacy modules (``Optimization.py``,
``Diffusion.py``, ``LossFunctions.py``, ``dist_utils.py``); legacy code paths
are untouched.
"""

__all__ = [
    "noise_tape",
    "schedule",
    "n_schedule",
    "config",
    "reference_tfg",
    "engine",
    "trace",
    "gmm_l2",
]
