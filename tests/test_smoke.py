"""Tier 0: does anything import and run at all?

Seconds, CPU-only. Catches import errors, typos and config mistakes before
burning GPU time on the equivalence gate.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cpu"
SEED = 20260816


def test_imports():
    """Every module the refactor touches must import cleanly."""
    import guidance
    import guidance.core
    import guidance.config
    import gcio.results
    from legacy import Optimization
    from legacy import Optimization_legacy
    from losses import functions as LossFunctions
    from models import diffusion as Diffusion
    from models.samplers import consistency as ConsistencyModels

    assert hasattr(guidance, "optimize_guided")
    assert hasattr(guidance, "GuidanceConfig")


def test_config_describe_is_jsonable():
    """cfg.describe() must survive json.dumps -- results.py depends on it."""
    import json
    from guidance import GuidanceConfig

    cfg = GuidanceConfig(target_generator=lambda n, d: None)
    json.dumps(cfg.describe())      # must not raise


def test_results_roundtrip(tmp_path):
    from gcio.results import save_run, load_runs
    from guidance import GuidanceConfig

    store = str(tmp_path / "runs.jsonl")
    cfg = GuidanceConfig(n_cond=7, target_generator=lambda n, d: None)

    save_run("unit", cfg, {"final_loss": 0.5}, seed=1, store=store)
    save_run("unit", cfg, {"final_loss": 0.4}, seed=2, store=store)
    save_run("other", cfg, {"final_loss": 9.9}, seed=3, store=store)

    recs = load_runs("unit", store=store)
    assert len(recs) == 2
    assert recs[0]["config"]["n_cond"] == 7
    assert recs[1]["metrics"]["final_loss"] == 0.4
    assert recs[0]["seed"] == 1


def test_aggregators_agree_on_single_value():
    """With one perturbation, logsumexp and mean must both reduce to L."""
    from guidance.config import aggregate_logsumexp, aggregate_mean, aggregate_single

    loss = torch.tensor(0.37)
    neg = [-loss]
    assert torch.allclose(aggregate_logsumexp(neg), loss, atol=1e-6)
    assert torch.allclose(aggregate_mean(neg), loss, atol=1e-6)
    assert torch.allclose(aggregate_single(neg), loss, atol=1e-6)


def test_logsumexp_is_a_softmin_not_a_mean():
    """Guard the documented property: logsumexp aggregation is <= the mean,
    i.e. it is dominated by the smallest loss. If this ever flips, the sign
    convention in aggregate_logsumexp has been broken."""
    from guidance.config import aggregate_logsumexp, aggregate_mean

    neg = [-torch.tensor(0.1), -torch.tensor(0.9)]
    assert aggregate_logsumexp(neg).item() < aggregate_mean(neg).item()


def test_end_to_end_tiny():
    """Run the core end to end on a tiny model. Values are meaningless
    (untrained); we only require it completes and returns finite numbers."""
    from models.diffusion import DiffusionModel
    from models.samplers.consistency import ConsistencyModel
    from losses.functions import MMDLoss, RBF
    from guidance import GuidanceConfig, optimize_guided, init_randn, perturb_beta_ratio

    torch.manual_seed(SEED)
    model_uncond = DiffusionModel(nfeatures=1, nunits=8, condition=False).to(DEVICE)
    model_uncond.diffusion_steps = 4
    model_cond = ConsistencyModel(nfeatures=1, condition_on=1, nunits=8, depth=2).to(DEVICE)

    cfg = GuidanceConfig(
        n_cond=8, n_mc=2,
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
        seed=SEED,
    )

    out = optimize_guided(model_uncond, model_cond, MMDLoss(kernel=RBF(device=DEVICE)), cfg)

    assert out["n_steps"] == 3          # range(3, 0, -1)
    assert torch.isfinite(out["x_final"]).all()
    assert torch.isfinite(out["final_loss"]).all()
