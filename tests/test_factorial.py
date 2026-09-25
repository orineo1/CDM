"""Correctness checks for the EXP-019 factorial harness."""
import pytest
import torch

from tfg.factorial_experiment import M_LGD, allocate


def test_no_lgd_spends_whole_budget_on_one_perturbation():
    a = allocate(16, "no_lgd")
    assert a == [16] and sum(a) == 16


def test_matched_budget_lgd_sums_to_the_budget():
    a = allocate(16, "lgd_matched")
    assert len(a) == M_LGD and sum(a) == 16


@pytest.mark.parametrize("B", [1, 2, 3, 4, 5, 7, 8, 10, 16, 17, 32, 33])
def test_matched_allocation_is_exact_and_deterministic(B):
    a = allocate(B, "lgd_matched")
    assert sum(a) == B, (B, a)
    assert a == allocate(B, "lgd_matched")
    assert max(a) - min(a) <= 1, "remainder must be spread one-per-perturbation"
    assert min(a) >= 1, "no perturbation may receive zero samples"


def test_remainders_go_to_the_first_perturbations():
    assert allocate(16, "lgd_matched") == [6, 5, 5]
    assert allocate(17, "lgd_matched") == [6, 6, 5]
    assert allocate(18, "lgd_matched") == [6, 6, 6]


def test_high_compute_lgd_costs_M_times_the_budget():
    a = allocate(16, "lgd_high")
    assert a == [16, 16, 16] and sum(a) == M_LGD * 16





def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown spatial mode"):
        allocate(10, "sideways")


def test_costs_are_ordered_as_designed():
    """A and B cost the same; C costs M_LGD times more."""
    cA = sum(allocate(16, "no_lgd"))
    cB = sum(allocate(16, "lgd_matched"))
    cC = sum(allocate(16, "lgd_high"))
    assert cA == cB == 16
    assert cC == M_LGD * 16


# -- n_max = 250 regime -----------------------------------------------------

@pytest.mark.parametrize("n_max", [4, 8, 16, 32])
def test_no_single_mmd_exceeds_the_cap(n_max):
    """The cap applies to each individual MMD evaluation, not just the total."""
    for mode in ("no_lgd", "lgd_matched", "lgd_high"):
        a = allocate(n_max, mode)
        assert max(a) <= n_max, (mode, a)


@pytest.mark.parametrize("n_max", [4, 8, 16, 32])
def test_costs_in_the_capped_regime(n_max):
    """A and B cost n_t; C costs 3 n_t. No single MMD exceeds n_max."""
    assert sum(allocate(n_max, "no_lgd")) == n_max
    assert sum(allocate(n_max, "lgd_matched")) == n_max
    assert sum(allocate(n_max, "lgd_high")) == M_LGD * n_max
    for mode in ("no_lgd", "lgd_matched", "lgd_high"):
        assert max(allocate(n_max, mode)) <= n_max


@pytest.mark.parametrize("n_max", [4, 8, 16, 32])
def test_n_t_protocols(n_max):
    from tfg.factorial_experiment import n_t_at
    from tfg.schedule import DiffusionSchedule
    s = DiffusionSchedule(T=100)
    assert all(n_t_at(t, s, "constant", n_max=n_max) == n_max for t in (99, 50, 1))
    assert n_t_at(1, s, "adaptive", n_max=n_max) == n_max      # data end
    assert n_t_at(99, s, "adaptive", n_max=n_max) == 1          # noisy end
    assert all(1 <= n_t_at(t, s, "adaptive", n_max=n_max) <= n_max
               for t in range(99, 0, -1))
    with pytest.raises(ValueError, match="unknown n_t protocol"):
        n_t_at(50, s, "sideways")


def test_lambda_zero_is_exactly_no_momentum():
    """A correctness gate: const lambda=0 must reproduce the no-momentum path bitwise."""
    import torch
    from tfg import oracle
    from tfg.factorial_experiment import optimize
    from tfg.run_estimator_v2 import make_fixed_target_set
    from tfg.train_cm import SEEDS, load_one
    from tfg.train_uncond import load as load_uncond

    p = oracle.load_params()
    S = make_fixed_target_set(p)
    mc, _ = load_one(SEEDS[0], "tfg_results/cm_checkpoints")
    mu, _ = load_uncond(out_dir="tfg_results/cm_checkpoints")

    x_none, i_none = optimize(mc, mu, S, "no_lgd", "none", 0, B=8, trace_out=True)
    x_zero, i_zero = optimize(mc, mu, S, "no_lgd", "const", 0, B=8,
                              lam_const=0.0, trace_out=True)
    assert x_none == x_zero, (x_none, x_zero)
    assert len(i_none["trajectory"]) == len(i_zero["trajectory"])
    for a, b in zip(i_none["trajectory"], i_zero["trajectory"]):
        assert a["x"] == b["x"] and a["grad"] == b["grad"] and a["grad_used"] == b["grad_used"]


def test_matched_lgd_never_allocates_zero_samples():
    """Regression: adaptive n_t starts at 1, and [1,0,0] makes an empty MMD (NaN)."""
    for B in range(1, 40):
        a = allocate(B, "lgd_matched")
        assert min(a) >= 1, (B, a)
        assert sum(a) == B, (B, a)
        assert len(a) == min(3, B), (B, a)


def test_small_budget_reduces_the_number_of_perturbations():
    assert allocate(1, "lgd_matched") == [1]
    assert allocate(2, "lgd_matched") == [1, 1]
    assert allocate(3, "lgd_matched") == [1, 1, 1]
    assert allocate(4, "lgd_matched") == [2, 1, 1]
