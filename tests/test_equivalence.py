"""Equivalence gate: refactored optimizers must match the frozen originals.

This is THE test that must pass before the refactor is merged. It compares
`Optimization_legacy.<fn>` (a byte-for-byte freeze of the pre-refactor code)
against `Optimization.<fn>` (now a thin wrapper over guidance.core) on
identical fixed seeds.

Run on the cluster:
    cd GlobalConditional && python -m pytest tests/test_equivalence.py -v

Deliberately cheap: tiny untrained models, few steps, few samples. This tests
CODE equivalence, not scientific results -- untrained weights are fine because
both paths see the same weights. Scientific agreement is scripts/run_regression.py.

--------------------------------------------------------------------------
IF A TEST FAILS: the most likely cause is RNG CONSUMPTION ORDER, not wrong
math. The new core must draw random numbers in the same sequence as legacy.
Check, in order:
  1. the number of torch.randn_like calls per outer step (n_mc)
  2. whether the target sampler is called inside or outside the MC loop
  3. whether the conditional sampler is called before or after the target one
  4. whether an extra final-loss evaluation runs (cfg.compute_final_loss)
If the order genuinely cannot be preserved, fall back to
`assert_distributionally_equal` below and record that in revision/CHANGES.md.
--------------------------------------------------------------------------
"""

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legacy import Optimization
from legacy import Optimization_legacy
from priors.diffusion import DiffusionModel
from samplers.consistency import ConsistencyModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20260816

# Deliberately tiny -- this is a code test, not an experiment.
NFEATURES_X = 1
NFEATURES_Y = 1
N_COND = 16
N_MC = 2
DIFFUSION_STEPS = 6

# 2-component MoG target, matching the toy example in the paper's appendix.
MOG_MEANS = [torch.tensor([-3.0]), torch.tensor([3.0])]
MOG_VARIANCES = [torch.tensor([[0.5]]), torch.tensor([[0.5]])]
MOG_WEIGHTS = torch.tensor([0.5, 0.5])


@pytest.fixture(scope="module")
def models():
    """Build the two models ONCE so both code paths see identical weights."""
    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(SEED)

    model_uncond = DiffusionModel(
        nfeatures=NFEATURES_X, nunits=16, condition=False
    ).to(DEVICE)
    model_uncond.diffusion_steps = DIFFUSION_STEPS

    model_cond = ConsistencyModel(
        nfeatures=NFEATURES_Y, condition_on=NFEATURES_X, nunits=16, depth=2
    ).to(DEVICE)

    model_uncond.eval()
    model_cond.eval()
    return model_uncond, model_cond


def _seeded(fn, *args, **kwargs):
    """Run fn with a freshly reset global RNG state."""
    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(SEED)
    return fn(*args, **kwargs)


def _as_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0])
    return float(x)


def assert_exactly_equal(new, old, name, rtol=1e-5, atol=1e-7):
    """Strict comparison. Preferred -- use this whenever RNG order is preserved."""
    if isinstance(old, torch.Tensor):
        assert torch.allclose(
            new.detach().cpu().float(), old.detach().cpu().float(),
            rtol=rtol, atol=atol
        ), f"{name}: new={new} old={old}"
    else:
        assert math.isclose(_as_float(new), _as_float(old), rel_tol=rtol, abs_tol=atol), \
            f"{name}: new={_as_float(new)} old={_as_float(old)}"


def assert_distributionally_equal(new_vals, old_vals, name, n_sigma=3.0):
    """Fallback when RNG order cannot be preserved.

    Weaker: checks the two sample means agree within n_sigma pooled standard
    errors. Only use this if exact equivalence is genuinely unattainable, and
    say so explicitly in revision/CHANGES.md.
    """
    new = torch.tensor([_as_float(v) for v in new_vals])
    old = torch.tensor([_as_float(v) for v in old_vals])
    n = len(new)
    se = math.sqrt((new.var().item() + old.var().item()) / n)
    delta = abs(new.mean().item() - old.mean().item())
    assert delta <= n_sigma * max(se, 1e-9), (
        f"{name}: mean {new.mean():.6f} vs {old.mean():.6f}, "
        f"delta={delta:.6f} > {n_sigma}*se={n_sigma*se:.6f}"
    )


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

COMMON = dict(
    mu_list=None, Sigma_list=None, alpha=None,
    nsamples=N_COND, num_x_t=N_MC, loss="MMD", CM=True, device=DEVICE,
)


def test_optimize_LGD_equivalence(models):
    model_uncond, model_cond = models

    old = _seeded(
        Optimization_legacy.optimize_LGD,
        model_uncond, model_cond, MOG_MEANS, MOG_VARIANCES, MOG_WEIGHTS,
        FLAG=False, **COMMON
    )
    new = _seeded(
        Optimization.optimize_LGD,
        model_uncond, model_cond, MOG_MEANS, MOG_VARIANCES, MOG_WEIGHTS,
        FLAG=False, **COMMON
    )

    assert_exactly_equal(new[0], old[0], "optimize_LGD: x_final")
    assert_exactly_equal(new[2], old[2], "optimize_LGD: final_loss")


def test_optimize_DPS_equivalence(models):
    model_uncond, model_cond = models

    old = _seeded(
        Optimization_legacy.optimize_DPS,
        model_uncond, model_cond, MOG_MEANS, MOG_VARIANCES, MOG_WEIGHTS,
        **COMMON
    )
    new = _seeded(
        Optimization.optimize_DPS,
        model_uncond, model_cond, MOG_MEANS, MOG_VARIANCES, MOG_WEIGHTS,
        **COMMON
    )

    assert_exactly_equal(new[0], old[0], "optimize_DPS: x_final")
    assert_exactly_equal(new[2], old[2], "optimize_DPS: best_loss")


def test_untouched_optimizers_are_identical_source():
    """optimize_LGD_latent and optimize_DFLOW were deliberately NOT unified
    (different algorithms -- see guidance/core.py docstring). Guard that they
    were not edited by accident."""
    import inspect
    for name in ("optimize_LGD_latent", "optimize_DFLOW", "compute_loss",
                 "find_samples_to_kl"):
        new_src = inspect.getsource(getattr(Optimization, name))
        old_src = inspect.getsource(getattr(Optimization_legacy, name))
        assert new_src == old_src, f"{name} was modified but should be untouched"
