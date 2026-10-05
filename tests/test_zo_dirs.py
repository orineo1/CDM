"""Directions of the zeroth-order gradient (`--zo-dirs-mode`), CPU, seconds.

A deterministic stand-in sampler returns its conditioning, so the sample loss is an exact function of x
and the finite differences can be checked against the analytic gradient.
"""
from types import SimpleNamespace

import torch

from cdm.guidance import core


class _EchoSampler:
    """samples = the conditioning rows (no noise)"""
    condition_on = 0

    def sample(self, nsamples, condition_x, device="cpu", **kw):
        return condition_x.clone()


C = torch.tensor([0.3, -1.0, 2.0, 0.5, -0.7, 1.2, 0.0, 0.9])


def _quad(samples, target):          # L = 0.5 |mean(samples) - c|^2, gradient mean(samples) - c
    return 0.5 * ((samples.mean(0) - C[:samples.shape[1]]) ** 2).sum()


def _cfg(**kw):
    base = dict(zo_eps=1e-2, zo_dirs=0, zo_dirs_mode="auto", device="cpu", cond_sample_kwargs={},
                condition_on_slice=False, cond_output_transform=None, zo_jvp_delta=1e-3,
                target_generator=lambda n, dev: torch.zeros(n, 1))
    base.update(kw)
    return SimpleNamespace(**base)


def _grad(cfg, x, seed=0, tangent=None):
    dirs = core._zo_dirs(cfg, x.numel(), seed, x.device, x.dtype, tangent)
    return core._zo_grad(_EchoSampler(), _quad, cfg, cfg, x, 4, seed, dirs)[1]


def test_auto_is_the_original_rule():
    x = torch.randn(1, 6)
    assert core._zo_dirs(_cfg(), 6, 0, x.device, x.dtype) is None
    g_auto = core._zo_grad(_EchoSampler(), _quad, _cfg(), _cfg(), x, 4, 5)[1]
    g_coord = _grad(_cfg(zo_dirs_mode="coord"), x, 5)
    assert torch.equal(g_auto, g_coord)
    g_auto3 = core._zo_grad(_EchoSampler(), _quad, _cfg(zo_dirs=3), _cfg(zo_dirs=3), x, 4, 5)[1]
    g_gauss3 = _grad(_cfg(zo_dirs=3, zo_dirs_mode="gauss"), x, 5)
    assert torch.allclose(g_auto3, g_gauss3)


def test_orth_full_basis_equals_coordinates():
    x = torch.randn(1, 6)
    true = x.view(-1) - C[:6]
    for seed in range(5):
        g = _grad(_cfg(zo_dirs=6, zo_dirs_mode="orth"), x, seed).view(-1)
        assert torch.allclose(g, true, atol=1e-4), (g, true)


def test_orth_subset_is_unbiased():
    x = torch.randn(1, 6)
    true = x.view(-1) - C[:6]
    mean = torch.stack([_grad(_cfg(zo_dirs=2, zo_dirs_mode="orth"), x, s).view(-1) for s in range(3000)]).mean(0)
    assert (mean - true).norm() < 0.08 * true.norm(), (mean, true)


def test_tangent_directions_give_the_exact_pulled_back_gradient():
    """J low rank: directions spanning range(J) give J^T g = J^T grad exactly (quadratic loss)."""
    torch.manual_seed(0)
    d, r = 8, 3
    A = torch.randn(d, r)
    J = A @ A.T                                   # Tweedie Jacobian of a linear map, rank r
    x = torch.randn(1, d)
    true = x.view(-1) - C
    g = _grad(_cfg(zo_dirs=r, zo_dirs_mode="tangent"), x, 0, tangent=lambda V: V @ J.T).view(-1)
    assert torch.allclose(J.T @ g, J.T @ true, atol=1e-3), (J.T @ g, J.T @ true)
    assert not torch.allclose(g, true, atol=1e-2)      # g itself is only the projection onto range(J)


def test_tweedie_jvp_of_a_linear_map():
    d = 5
    J = torch.randn(d, d)
    cfg = _cfg(ddim_step_fn=lambda m, X, t, i: (X, X @ J.T))
    x_t = torch.randn(1, d)
    V = torch.randn(4, d)
    JV = core._tweedie_jvp(None, x_t, 10, 0, cfg, V)
    assert torch.allclose(JV, V @ J.T, atol=1e-3)
