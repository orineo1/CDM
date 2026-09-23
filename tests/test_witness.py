"""Gradient subsampling (utils/witness.py): the selection designs keep exactly k rows with the
stated inclusion probabilities, Horvitz-Thompson reweighting makes the subsampled gradient
unbiased for the full one, and --select uniform without rescaling is the old behaviour."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.functions import MMDLoss, RBF          # noqa: E402
from utils import witness as W                    # noqa: E402


def test_pps_inclusion_sums_to_k_and_caps_at_one():
    p = torch.tensor([0.7, 0.1, 0.1, 0.05, 0.05])
    pi = W.pps_inclusion(p, 3)
    assert abs(float(pi.sum()) - 3) < 1e-9
    assert float(pi.max()) <= 1.0 and float(pi[0]) == 1.0


def test_systematic_pps_keeps_k_with_the_stated_probabilities():
    torch.manual_seed(0)
    pi = W.pps_inclusion(torch.rand(12) + 0.05, 4)
    counts = torch.zeros(12)
    n_draws = 20000
    for _ in range(n_draws):
        keep = W.systematic_pps(pi)
        assert int(keep.sum()) == 4
        counts += keep
    assert torch.allclose(counts / n_draws, pi.float(), atol=0.015)


def _grads(rule, rescale, n_rep):
    """Mean gradient w.r.t. x of the MMD loss over n_rep subsamplings of one fixed sample set."""
    torch.manual_seed(1)
    x = torch.tensor([0.3, -0.2], requires_grad=True)
    eps = torch.randn(40, 2)
    target = torch.randn(40, 2) + 1.0
    loss = MMDLoss(kernel=RBF())
    y = eps * 0.5 + x                     # samples depend on x by reparametrisation
    full = torch.autograd.grad(loss(y, target), x, retain_graph=True)[0]
    acc = torch.zeros(2)
    for _ in range(n_rep):
        keep, pi = W.select(y, 10, rule, target=target, kernel=RBF(), floor=0.0)
        acc += torch.autograd.grad(loss(W.reweight(y, keep, pi, rescale), target), x, retain_graph=True)[0]
    return full, acc / n_rep


def test_horvitz_thompson_is_unbiased_for_both_rules():
    for rule in ("uniform", "witness"):
        full, mean = _grads(rule, "ht", 4000)
        assert torch.allclose(mean, full, rtol=0.05, atol=1e-3 * float(full.norm())), (rule, mean, full)


def test_uniform_without_rescaling_is_the_old_subsampler():
    torch.manual_seed(3)
    y = torch.randn(30, 2, requires_grad=True)
    torch.manual_seed(7)
    keep, pi = W.select(y, 12, "uniform")
    new = W.reweight(y, keep, pi, "none")
    torch.manual_seed(7)
    old_keep = torch.zeros(30, dtype=torch.bool)
    old_keep[torch.randperm(30)[:12]] = True
    old = torch.where(old_keep[:, None], y, y.detach())
    assert torch.equal(keep, old_keep) and torch.equal(new, old)
    g_new = torch.autograd.grad(new.sum(), y, retain_graph=True)[0]
    g_old = torch.autograd.grad(old.sum(), y)[0]
    assert torch.equal(g_new, g_old)
