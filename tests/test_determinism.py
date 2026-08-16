"""Tier 2: same seed must give the same answer, twice in a row.

If this fails, the equivalence gate (Tier 1) is meaningless -- you cannot
compare two code paths when one path does not agree with itself. On CUDA,
some kernels are nondeterministic by default; if that is the cause, set
    torch.use_deterministic_algorithms(True)
    CUBLAS_WORKSPACE_CONFIG=:4096:8
and re-run, then record in revision/CHANGES.md that determinism requires it.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Diffusion import DiffusionModel
from ConsistencyModels import ConsistencyModel
from LossFunctions import MMDLoss, RBF
from guidance import GuidanceConfig, optimize_guided, init_randn, perturb_beta_ratio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20260816


def _build_cfg(seed):
    return GuidanceConfig(
        n_cond=16, n_mc=2,
        init_fn=init_randn,
        perturb_fn=perturb_beta_ratio,
        perturb_square=False,
        step_size_fn=lambda t, i: 1.0,
        timestep_source="range",
        ddim_step_style="condition_x",
        repeat_condition=True,
        condition_on_slice=False,
        target_generator=lambda n, dev: torch.randn(n, 1, device=dev),
        device=DEVICE,
        seed=seed,
    )


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(SEED)
    mu = DiffusionModel(nfeatures=1, nunits=16, condition=False).to(DEVICE)
    mu.diffusion_steps = 6
    mc = ConsistencyModel(nfeatures=1, condition_on=1, nunits=16, depth=2).to(DEVICE)
    mu.eval(); mc.eval()
    return mu, mc


def test_same_seed_same_result(models):
    model_uncond, model_cond = models
    loss_fn = MMDLoss(kernel=RBF(device=DEVICE))

    a = optimize_guided(model_uncond, model_cond, loss_fn, _build_cfg(SEED))
    b = optimize_guided(model_uncond, model_cond, loss_fn, _build_cfg(SEED))

    assert torch.allclose(a["x_final"], b["x_final"], rtol=0, atol=0), (
        "same seed gave different x_final -- nondeterminism present; "
        "see this module's docstring"
    )
    assert torch.allclose(a["final_loss"], b["final_loss"], rtol=0, atol=0)


def test_different_seed_different_result(models):
    """Sanity check on the seeding itself: if two seeds give identical output,
    cfg.seed is not actually being applied and Tier 1 would pass vacuously."""
    model_uncond, model_cond = models
    loss_fn = MMDLoss(kernel=RBF(device=DEVICE))

    a = optimize_guided(model_uncond, model_cond, loss_fn, _build_cfg(SEED))
    b = optimize_guided(model_uncond, model_cond, loss_fn, _build_cfg(SEED + 1))

    assert not torch.allclose(a["x_final"], b["x_final"]), (
        "different seeds gave identical output -- cfg.seed is not being applied"
    )
