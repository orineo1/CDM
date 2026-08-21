"""Guard the Phase 2 package move: every legacy import must still resolve.

Phase 2 moved the flat top-level modules into packages (priors/, samplers/,
losses/, utils/) and left a compatibility shim at each old path. Notebooks use
the old names, and pickled checkpoints record the old module path, so the
shims are load-bearing -- not cosmetic.

This test asserts that for every moved module:
  * the old name still imports
  * the new name imports
  * every public symbol is present under BOTH names and is the *same object*

If a symbol is missing here, a notebook will fail at runtime.
"""

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOVES = {
    "NN_utils":                   "utils.nn",
    "dist_utils":                 "utils.dist",
    "superposition_utils":        "utils.superposition",
    "EncoderDecoder":             "utils.encoder_decoder",
    "evalModels":                 "utils.evaluation",
    "Diffusion":                  "priors.diffusion",
    "DiffusionGeneral":           "priors.diffusion_general",
    "ConsistencyModels":          "samplers.consistency",
    "FlowMatching":               "samplers.flow_matching",
    "FM_Solver_Extension_module": "samplers.fm_solver",
    "LossFunctions":              "losses.functions",
}

# Names every notebook is known to use directly; checked explicitly so a
# regression here fails loudly rather than at notebook runtime.
CRITICAL = {
    "Diffusion":         ["DiffusionModel"],
    "ConsistencyModels": ["ConsistencyModel", "ConsistencyModeliCT"],
    "LossFunctions":     ["MMDLoss", "RBF"],
    "dist_utils":        ["generate_mog_samples",
                          "generate_mog_samples_not_differentiable"],
}


@pytest.mark.parametrize("old,new", sorted(MOVES.items()))
def test_shim_exposes_same_objects(old, new):
    old_mod = importlib.import_module(old)
    new_mod = importlib.import_module(new)

    public = [n for n in dir(new_mod) if not n.startswith("_")]
    assert public, f"{new} exposes nothing -- did the move truncate it?"

    missing, mismatched = [], []
    for name in public:
        if not hasattr(old_mod, name):
            missing.append(name)
        elif getattr(old_mod, name) is not getattr(new_mod, name):
            mismatched.append(name)

    assert not missing, f"{old} shim is missing: {missing}"
    assert not mismatched, f"{old} shim rebinds (not the same object): {mismatched}"


@pytest.mark.parametrize("old,names", sorted(CRITICAL.items()))
def test_critical_symbols_importable_by_old_name(old, names):
    mod = importlib.import_module(old)
    for n in names:
        assert hasattr(mod, n), f"{old}.{n} missing -- notebooks depend on it"


def test_from_import_style_still_works():
    """Notebooks use `from X import Y`, which resolves differently from
    attribute access on the shim. Check that path explicitly."""
    from LossFunctions import MMDLoss, RBF          # noqa: F401
    from Diffusion import DiffusionModel            # noqa: F401
    from ConsistencyModels import ConsistencyModel  # noqa: F401
    from dist_utils import generate_mog_samples     # noqa: F401


def test_packages_import_cleanly():
    for pkg in ("priors", "samplers", "losses", "utils", "guidance", "gcio"):
        importlib.import_module(pkg)


def test_diffusion_models_are_distinct_not_duplicates():
    """priors.diffusion.DiffusionModel and priors.diffusion_general.DiffusionModel
    share a name but are different models (discrete-time DDPM vs continuous-time
    SDE). They must not be merged; see the naming note in diffusion_general.py."""
    from priors.diffusion import DiffusionModel as Discrete
    from priors.diffusion_general import DiffusionModel as SDE
    from priors.diffusion_general import DiffusionModelSDE

    assert Discrete is not SDE, "the two DiffusionModel classes were merged"
    assert DiffusionModelSDE is SDE, "DiffusionModelSDE alias is broken"

    methods = lambda c: {n for n in dir(c) if not n.startswith("_")}
    only_sde = methods(SDE) - methods(Discrete)
    only_discrete = methods(Discrete) - methods(SDE)

    # Both expose sample_ddim_step (with different signatures -- see
    # GuidanceConfig.ddim_step_style), so it is NOT a discriminator.
    assert "vector_field_SDE" in only_sde, only_sde
    assert "sm_loss" in only_sde, only_sde
    assert "sample_superdiff_and" in only_discrete, only_discrete
