"""Particle population and post-refinement (guidance.core.optimize_guided_population / refine).

CPU, seconds. Untrained tiny models: values are meaningless, we check structure and the
reductions to the single-trajectory algorithm.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from priors.diffusion import DiffusionModel
from samplers.consistency import ConsistencyModel
from losses.functions import MMDLoss, RBF
from guidance import (GuidanceConfig, optimize_guided, optimize_guided_population, init_randn,
                      perturb_beta_ratio)
from guidance.core import _systematic_resample

DEVICE = "cpu"
SEED = 20260921


def _models():
    torch.manual_seed(SEED)
    mu = DiffusionModel(nfeatures=1, nunits=8, condition=False).to(DEVICE)
    mu.diffusion_steps = 5
    mc = ConsistencyModel(nfeatures=1, condition_on=1, nunits=8, depth=2).to(DEVICE)
    mu.eval(); mc.eval()
    return mu, mc


def _cfg(**kw):
    base = dict(
        n_cond=8, n_mc=2, init_fn=init_randn, perturb_fn=perturb_beta_ratio, perturb_square=False,
        step_size_fn=lambda t, i: 1.0, timestep_source="range", ddim_step_style="condition_x",
        repeat_condition=True, condition_on_slice=False,
        target_generator=lambda n, dev: torch.randn(n, 1, device=dev), device=DEVICE, seed=SEED,
    )
    base.update(kw)
    return GuidanceConfig(**base)


def test_population_of_one_equals_single_trajectory():
    mu, mc = _models()
    loss = MMDLoss(kernel=RBF(device=DEVICE))
    single = optimize_guided(mu, mc, loss, _cfg())
    pop = optimize_guided_population(mu, mc, loss, _cfg(n_particles=1))
    assert torch.allclose(single["x_final"], pop["x_final"])
    assert abs(float(single["final_loss"]) - float(pop["final_loss"])) < 1e-6
    assert pop["n_resamples"] == 0 and pop["x_particles"].shape == (1, 1)


def test_population_without_resampling_keeps_r_particles():
    mu, mc = _models()
    loss = MMDLoss(kernel=RBF(device=DEVICE))
    pop = optimize_guided_population(mu, mc, loss, _cfg(n_particles=4))
    assert pop["x_particles"].shape == (4, 1) and len(pop["final_losses"]) == 4
    assert pop["best_particle"] == min(range(4), key=lambda r: pop["final_losses"][r])
    assert all(len(s["loss"]) == 4 and s["ess"] == 4.0 for s in pop["trace"])


def test_population_resampling_fires_and_records_ancestry():
    mu, mc = _models()
    loss = MMDLoss(kernel=RBF(device=DEVICE))
    # a huge beta makes the weights degenerate, so resampling must fire at every step
    pop = optimize_guided_population(mu, mc, loss, _cfg(n_particles=4, resample_beta=1e6, ess_frac=0.99))
    assert pop["n_resamples"] == pop["n_steps"] > 0
    for s in pop["trace"]:
        assert s["resampled"] and len(s["ancestors"]) == 4 and all(0 <= a < 4 for a in s["ancestors"])
    assert pop["x_particles"].shape == (4, 1)


def test_systematic_resample_follows_weights():
    torch.manual_seed(0)
    w = torch.tensor([0.0, 0.0, 1.0, 0.0])
    assert _systematic_resample(w).tolist() == [2, 2, 2, 2]
    w = torch.tensor([0.5, 0.5])
    idx = _systematic_resample(w).tolist()
    assert sorted(idx) == [0, 1]


def test_refinement_runs_and_reports_before_after():
    mu, mc = _models()
    loss = MMDLoss(kernel=RBF(device=DEVICE))
    single = optimize_guided(mu, mc, loss, _cfg(refine_steps=3, refine_lr=0.01, refine_lambda=0.1))
    assert single["refine"] is not None and len(single["refine"]["trace"]) == 3
    assert single["refine"]["final_loss_before"] is not None
    assert torch.isfinite(single["x_final"]).all()
    pop = optimize_guided_population(mu, mc, loss, _cfg(n_particles=2, refine_steps=2))
    assert len(pop["refine"]["traces"]) == 2 and pop["refine"]["x_before"].shape == (2, 1)


def test_linear_resample_schedule_ramps_and_runs():
    mu, mc = _models()
    loss = MMDLoss(kernel=RBF(device=DEVICE))
    pop = optimize_guided_population(mu, mc, loss, _cfg(n_particles=3, resample_beta=2.0, resample_schedule="linear"))
    betas = [s["resample_beta"] for s in pop["trace"]]
    assert betas[0] < betas[-1] and abs(betas[-1] - 2.0) < 1e-9
    assert pop["x_particles"].shape == (3, 1)


def test_exact_conditional_sampler_matches_analytic_conditional():
    from samplers.exact import ExactConditionalSampler
    from utils.dist import compute_conditionals, compute_alpha
    torch.manual_seed(0)
    mu = [torch.tensor([0., 1., 0.5]), torch.tensor([2., -1., 1.])]
    S = [torch.tensor([[1., .3, .2], [.3, 1., .1], [.2, .1, 1.]]), torch.eye(3) * 0.5]
    w = torch.tensor([0.5, 0.5])
    smp = ExactConditionalSampler(mu, S, w, condition_on=1)
    x = torch.tensor([0.3], requires_grad=True)
    y, _, _ = smp.sample(20000, condition_x=x.view(1, -1).repeat(20000, 1), device="cpu")
    m, Sg = compute_conditionals(mu, S, x.detach().view(-1)); a = compute_alpha(mu, S, w, x.detach().view(-1))
    mean_true = sum(float(ai) * mi.view(-1) for ai, mi in zip(a, m))
    assert torch.allclose(y.mean(0), mean_true, atol=0.05)
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
