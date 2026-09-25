"""Tests for the consistency-model training/validation/estimator helpers."""

import math

import pytest
import torch

from tfg import oracle
from tfg import validate_cm as V
from tfg.run_cm_estimator_study import (
    conditional_samples_analytic,
    key_seed,
    stats_vs_reference,
    target_samples,
)


@pytest.fixture(scope="module")
def params():
    return oracle.load_params()


# -- metric helpers ---------------------------------------------------------

def test_w1_of_identical_samples_is_zero():
    a = torch.randn(500, 1, dtype=torch.float64)
    assert V.w1_1d(a, a.clone()) == pytest.approx(0.0, abs=1e-15)


def test_w1_of_a_pure_shift_equals_the_shift():
    """W1 between p and p shifted by c is exactly |c|."""
    a = torch.randn(2000, 1, dtype=torch.float64)
    assert V.w1_1d(a + 1.7, a) == pytest.approx(1.7, abs=1e-12)


def test_w1_is_symmetric():
    a = torch.randn(300, 1, dtype=torch.float64)
    b = torch.randn(300, 1, dtype=torch.float64) * 2 + 1
    assert V.w1_1d(a, b) == pytest.approx(V.w1_1d(b, a), abs=1e-15)


def test_mmd2_u_of_identical_distributions_is_near_zero():
    torch.manual_seed(0)
    a = torch.randn(600, 1, dtype=torch.float64)
    b = torch.randn(600, 1, dtype=torch.float64)
    val = V.mmd2_u(a, b, bandwidth=1.0)
    # Unbiased: expectation is exactly 0, so the value should be within a few
    # standard errors of 0 and must be able to go negative.
    assert abs(val) < 0.05, val


def test_mmd2_u_detects_a_shift():
    torch.manual_seed(0)
    a = torch.randn(600, 1, dtype=torch.float64)
    b = torch.randn(600, 1, dtype=torch.float64) + 3.0
    assert V.mmd2_u(a, b, bandwidth=1.0) > 0.3


# -- keyed randomness -------------------------------------------------------

def test_key_seed_is_deterministic_and_discriminating():
    assert key_seed("a", 1, 2) == key_seed("a", 1, 2)
    assert key_seed("a", 1, 2) != key_seed("a", 1, 3)
    assert key_seed("a", 1, 2) != key_seed("b", 1, 2)
    assert 0 <= key_seed("a", 1, 2) < 2 ** 31 - 1


def test_key_seed_survives_a_fresh_process():
    """blake2b, not hash(): Python salts string hashing per process."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = ("import sys; sys.path.insert(0, '.');"
            "from tfg.run_cm_estimator_study import key_seed;"
            "print(key_seed('role', -4.6, 8, 3))")
    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-W", "ignore", "-c", code],
                           capture_output=True, text=True, cwd=str(root))
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1]
    assert int(outs[0]) == key_seed("role", -4.6, 8, 3)


# -- the analytic reference sampler used for the model-error decomposition ---

def test_analytic_conditional_sampler_matches_the_conditional_moments(params):
    """The reference sampler must reproduce P(Y|X=x), not merely something close."""
    x = torch.tensor([-5.0], dtype=torch.float32)
    torch.manual_seed(0)
    y = conditional_samples_analytic(x, params, 20_000).double()

    m, c, w = oracle.conditional_params(
        x.double(), params["mu_list"], params["Sigma_list"], params["alpha"])
    exp_mean = float((w * m.reshape(-1)).sum())
    exp_var = float((w * (c.reshape(-1) + m.reshape(-1) ** 2)).sum() - exp_mean ** 2)

    assert float(y.mean()) == pytest.approx(exp_mean, abs=0.1)
    assert float(y.std()) == pytest.approx(math.sqrt(exp_var), rel=0.05)


def test_analytic_conditional_sampler_is_differentiable(params):
    x = torch.tensor([-3.0], dtype=torch.float32, requires_grad=True)
    y = conditional_samples_analytic(x, params, 64)
    g, = torch.autograd.grad(y.sum(), x, allow_unused=True)
    assert g is not None and torch.isfinite(g).all()


def test_target_samples_have_the_right_shape_and_moments(params):
    torch.manual_seed(0)
    t = target_samples(params, 20_000).double()
    assert t.shape == (20_000, 1)
    # G is 0.5 N(5, 0.124) + 0.5 N(-5, 0.124): mean 0, std ~= sqrt(25 + 0.124)
    assert abs(float(t.mean())) < 0.15
    assert float(t.std()) == pytest.approx(math.sqrt(25 + 0.12395), rel=0.02)


# -- statistics -------------------------------------------------------------

def test_stats_vs_reference_recovers_known_quantities():
    g = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    st = stats_vs_reference(g, 2.0)
    assert st["mean"] == pytest.approx(2.5)
    assert st["bias"] == pytest.approx(0.5)
    assert st["variance"] == pytest.approx(g.var(unbiased=True).item())
    assert st["mse"] == pytest.approx(((g - 2.0) ** 2).mean().item())
    assert st["sign_agreement"] == 1.0


def test_stats_sign_agreement_counts_correctly():
    g = torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert stats_vs_reference(g, 1.0)["sign_agreement"] == pytest.approx(0.75)


def test_mse_decomposes_into_bias_squared_plus_variance():
    torch.manual_seed(0)
    g = torch.randn(500, dtype=torch.float64) + 0.3
    st = stats_vs_reference(g, 0.0)
    biased_var = g.var(unbiased=False).item()
    assert st["mse"] == pytest.approx(st["bias"] ** 2 + biased_var, rel=1e-9)
