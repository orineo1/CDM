"""Shared fixtures for the TFG equivalence and unit tests.

Two environment facts this file works around:

  * The repository has no packaging metadata, so the inner repo root is put on
    ``sys.path`` explicitly.
  * ``LossFunctions.py`` does ``import ot`` (POT) at module scope, and POT is
    not installed in this environment.  Rather than adding a dependency, we
    insert a stub module for ``ot`` before importing.  Nothing we use from
    ``LossFunctions`` touches POT -- only ``RBF`` and ``MMDLoss``, which are
    pure torch.  This keeps the tests running against the *real* repository
    MMD implementation instead of a re-typed copy.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "ot" not in sys.modules:
    sys.modules["ot"] = types.ModuleType("ot")

torch.use_deterministic_algorithms(True)


@pytest.fixture(autouse=True)
def _float64_cpu():
    """Force float64 on CPU for every test, and restore afterwards."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


class ToyDenoiser(torch.nn.Module):
    """Small deterministic ``eps_theta(x, t)`` that is *consistent* with the schedule.

    Deliberately not the repository's ``DiffusionModel``: for the equivalence
    tests the denoiser only has to be a fixed, differentiable, non-trivial
    function that both engines call identically.  Using a toy network avoids
    the float32 schedule buffers described in ``tfg/schedule.py`` and keeps the
    test independent of training.

    Crucially it is parameterised through a *bounded predicted clean sample*:

        m(x, t) = tanh(MLP([x, t/T]))          in (-1, 1)
        eps     = (x - sqrt(ab_t) * m) / sqrt(1 - ab_t)

    so that the engine's ``x_{0|t} = (x - sqrt(1-ab_t) eps)/sqrt(ab_t)``
    recovers ``m`` and therefore stays bounded.  An unconstrained random
    network is *not* consistent with the schedule: it makes ``x_{0|t}`` scale
    like ``1/sqrt(ab_t)``, the trajectory diverges to ~1e22, and the resulting
    chaotic amplification turns 1e-16 round-off into arbitrarily large
    differences.  That is a property of an untrained denoiser, not of the
    guidance algebra, and it would make every tolerance in the equivalence
    suite meaningless.
    """

    def __init__(self, d=2, hidden=16, T=100, seed=0, schedule=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.T = T
        self.schedule = schedule
        self.w1 = torch.nn.Parameter(torch.randn(d + 1, hidden, generator=g, dtype=torch.float64) * 0.5)
        self.b1 = torch.nn.Parameter(torch.randn(hidden, generator=g, dtype=torch.float64) * 0.1)
        self.w2 = torch.nn.Parameter(torch.randn(hidden, d, generator=g, dtype=torch.float64) * 0.5)
        self.b2 = torch.nn.Parameter(torch.randn(d, generator=g, dtype=torch.float64) * 0.1)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def predicted_x0(self, x, t):
        tt = torch.full((x.shape[0], 1), float(t) / self.T, dtype=x.dtype, device=x.device)
        h = torch.tanh(torch.cat([x, tt], dim=1) @ self.w1 + self.b1)
        return torch.tanh(h @ self.w2 + self.b2)

    def forward(self, x, t):
        m = self.predicted_x0(x, t)
        if self.schedule is None:
            return m
        ab = self.schedule.alphabar[t]
        return (x - torch.sqrt(ab) * m) / torch.sqrt(1.0 - ab)


@pytest.fixture
def denoiser():
    return ToyDenoiser(d=2, T=8, seed=0)


def make_quadratic_log_f(center, scale=1.0):
    """A plain TFG target predictor: ``f(x) = exp(-scale*||x - c||^2)``.

    This is the "same direct target predictor f" of Test A -- it has nothing to
    do with distributions or MMD, which is the point: the generalised engine
    must accept an ordinary TFG predictor.
    """
    c = torch.as_tensor(center, dtype=torch.float64)

    def log_f(x):
        return -scale * ((x - c) ** 2).sum()

    return log_f
