"""Historical optimizer entry points.

Optimization.py: the original functions (optimize_LGD, optimize_DPS, optimize_LGD_CM,
optimize_LGD_cfg) as thin wrappers over guidance.optimize_guided, plus optimize_DFLOW
and optimize_LGD_latent, which are separate algorithms.
Optimization_legacy.py: a frozen copy of the pre-refactor code, used only by
tests/test_equivalence.py and scripts/run_regression.py as the reference.
"""
