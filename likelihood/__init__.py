"""Likelihood-based optimizer comparison line (D-FLOW-style baseline).

These two modules used to live in ``scripts/`` and were imported by bare
sibling import, which only worked because of the ``sys.path`` hack in the
scripts.  They are libraries, not entry points, so they live here instead.

Exports the four functions the call sites use:

    from likelihood import optimize_likelihood, optimize_likelihood_direct
    from likelihood import get_representative_points, get_noised_representatives
"""

from .optimizer import optimize_likelihood, optimize_likelihood_direct
from .representative_points import (
    get_representative_points,
    get_noised_representatives,
)

__all__ = [
    "optimize_likelihood",
    "optimize_likelihood_direct",
    "get_representative_points",
    "get_noised_representatives",
]
