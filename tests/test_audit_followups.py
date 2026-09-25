"""Coverage added in response to the independent audit of Checkpoint 1.

The audit found four gaps that the original suite could not see:

  B4a  ``eta_keys`` -- the whole plumbing of Component 1 -- had ZERO
       behavioural coverage, because the only predictor accepting it discarded
       it. A mutation that re-keyed eta on every predictor evaluation was
       therefore invisible.
  B4b  ``structured_vector`` with ``increase``/``decrease`` was never
       exercised; only ``constant`` was.
  A2   ``NoiseTape`` served different values depending on which dtype was
       requested first.
  B3   ``compare_traces`` returned ok=True when it compared nothing.
"""

import math

import pytest
import torch

from conftest import ToyDenoiser, make_quadratic_log_f
from tfg.config import NScheduleConfig, TFGConfig
from tfg.engine import GeneralizedTFG
from tfg.noise_tape import NoiseTape
from tfg.schedule import DiffusionSchedule, structured_vector
from tfg.trace import Tracer, compare_traces

SHAPE = (1, 2)
T = 6


# ---------------------------------------------------------------------------
# B3: the comparator must not pass vacuously
# ---------------------------------------------------------------------------

def test_empty_traces_do_not_count_as_agreement():
    ok, report = compare_traces(Tracer(), Tracer(), atol=0.0)
    assert not ok, "comparing zero keys must not report agreement"
    assert report["n_keys_compared"] == 0


def test_disabled_tracers_do_not_count_as_agreement():
    ok, _ = compare_traces(Tracer(enabled=False), Tracer(enabled=False), atol=0.0)
    assert not ok


def test_tensor_vs_non_tensor_is_flagged():
    a, b = Tracer(), Tracer()
    a("x", 1, 1, None, torch.zeros(3))
    b("x", 1, 1, None, 0.0)
    ok, report = compare_traces(a, b, atol=0.0)
    assert not ok
    assert report["max_abs_err"] == float("inf")


# ---------------------------------------------------------------------------
# A2: dtype request order must not change served values
# ---------------------------------------------------------------------------

def test_dtype_request_order_does_not_change_values():
    """f32-then-f64 must agree with f64-then-f32.

    The original implementation cached the first-requested dtype, so asking for
    float32 first permanently degraded the key to float32 precision -- a
    2.9e-08 discrepancy, in the one class whose contract is that request order
    never matters.
    """
    a = NoiseTape(seed=42)
    a.randn(("delta", 3, 0), (4,), dtype=torch.float32)
    a64 = a.randn(("delta", 3, 0), (4,), dtype=torch.float64)

    b = NoiseTape(seed=42)
    b64 = b.randn(("delta", 3, 0), (4,), dtype=torch.float64)

    assert torch.equal(a64, b64), (
        "value served depends on which dtype was requested first: "
        f"max diff {(a64 - b64).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# B4b: rho/mu structures beyond "constant"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("structure", ["constant", "increase", "decrease"])
def test_structured_vector_shape_and_positivity(structure):
    sch = DiffusionSchedule(T=20)
    v = structured_vector(2.0, structure, sch)
    assert v.shape == (21,)
    assert torch.isfinite(v).all()
    assert (v[1:] >= 0).all()


def test_structured_vector_normalisation():
    """TFG Eq. 8 normalises so the structure averages to 1 over t = 1..T."""
    sch = DiffusionSchedule(T=20)
    for structure in ("increase", "decrease"):
        v = structured_vector(1.0, structure, sch)
        assert v[1:].mean().item() == pytest.approx(1.0, abs=1e-12)


def test_increase_and_decrease_are_opposites():
    sch = DiffusionSchedule(T=20)
    inc = structured_vector(1.0, "increase", sch)
    dec = structured_vector(1.0, "decrease", sch)
    # Indexing is by t, so inc[1] is the DATA end (t=1) and inc[T] the noisy
    # end (t=T). "increase" is proportional to alpha_t, which is near 1 at
    # t=1 and small at t=T, so guidance grows as sampling proceeds toward the
    # data -- which is the sense in which TFG calls it "increase".
    assert inc[1] > inc[sch.T], "increase must be strongest at the data end"
    assert dec[1] < dec[sch.T], "decrease must be strongest at the noisy end"


def test_unknown_structure_rejected():
    sch = DiffusionSchedule(T=8)
    with pytest.raises(ValueError, match="unknown structure"):
        structured_vector(1.0, "sigmoid", sch)


# ---------------------------------------------------------------------------
# B4a: eta_keys must actually reach the predictor and be honoured
# ---------------------------------------------------------------------------

class EtaConsumingPredictor:
    """A predictor that genuinely consumes ``eta_keys``.

    ``h_phi(x, eta_i)`` draws its conditional noise from the tape under the
    keys the engine supplies, so re-keying eta changes the answer. This is what
    the original ``CompositePredictor`` failed to do -- it accepted ``eta_keys``
    and discarded them, making the entire Component 1 plumbing untested.
    """

    def __init__(self, tape, S_G, beta=1.0, d_y=1, seed=3):
        self.tape = tape
        self.S_G = S_G
        self.beta = float(beta)
        self.d_y = d_y
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(SHAPE[1], d_y, generator=g, dtype=torch.float64)
        self.seen_keys = []

    def __call__(self, x, n_t=None, eta_keys=None):
        if eta_keys is None:
            eta_keys = [("eta", "frozen", 0)]
        self.seen_keys.extend(eta_keys)
        mean = torch.tanh(x @ self.W)
        etas = torch.cat([self.tape.randn(k, (1, self.d_y)) for k in eta_keys], dim=0)
        samples = mean + 0.3 * etas
        # Squared-distance stand-in for MMD; the point here is the eta plumbing.
        diff = samples.mean(0) - self.S_G.mean(0)
        return -self.beta * (diff ** 2).sum()


def _run_engine(predictor_cls, n_max, kind="constant", tape_seed=17, **cfg_kw):
    schedule = DiffusionSchedule(T=T)
    denoiser = ToyDenoiser(d=SHAPE[1], T=T, seed=0, schedule=schedule)
    eps_theta = lambda x, t: denoiser(x, t)          # noqa: E731
    tape = NoiseTape(seed=tape_seed)
    S_G = torch.full((8, 1), 0.4, dtype=torch.float64)
    predictor = predictor_cls(tape, S_G)

    cfg = TFGConfig(T=T, N_recur=2, N_iter=1, gamma_bar=0.0,
                    rho_scalar=0.4, mu_scalar=0.2, n_mc=1, **cfg_kw)
    cfg.n_schedule = NScheduleConfig(enabled=True, type=kind, n_max=n_max)
    tr = Tracer()
    engine = GeneralizedTFG(eps_theta, predictor, schedule, tape, cfg)
    engine.run(SHAPE, trace=tr)
    return tr, engine, predictor, tape


def test_eta_keys_reach_the_predictor():
    _, engine, predictor, _ = _run_engine(EtaConsumingPredictor, n_max=4, kind="time")
    assert predictor.seen_keys, "predictor never received eta_keys"
    assert all(k[0] == "eta" for k in predictor.seen_keys)


def test_eta_shared_within_an_outer_step_and_fresh_at_the_next():
    """The stated contract of Component 1's conditional seeding."""
    _, _, predictor, _ = _run_engine(EtaConsumingPredictor, n_max=4, kind="time")
    by_t = {}
    for kind, t, i in predictor.seen_keys:
        by_t.setdefault(t, set()).add(i)
    # Within one outer step the same indices recur across recurrences and
    # inner iterations -- never a growing set of distinct indices.
    for t, idxs in by_t.items():
        assert idxs == set(range(len(idxs))), f"t={t} used non-contiguous eta indices"
    # Different outer steps use different keys.
    assert len(by_t) > 1


def test_n_t_actually_varies_with_an_adaptive_schedule():
    _, engine, _, _ = _run_engine(EtaConsumingPredictor, n_max=4, kind="time")
    hist = engine.counter.n_t_history
    assert len(set(hist)) > 1, f"n_t never varied: {hist}"
    assert hist == sorted(hist), "n_t must be non-decreasing toward the data end"
    assert min(hist) >= 1 and max(hist) <= 4


def test_conditional_call_count_tracks_n_t():
    _, engine, _, _ = _run_engine(EtaConsumingPredictor, n_max=4, kind="time")
    c = engine.counter
    # Every predictor evaluation at outer step t draws n_t conditional samples.
    assert c.conditional_calls > c.predictor_evals, (
        "with n_max>1 the conditional-generator call count must exceed the "
        "number of predictor evaluations"
    )


# -- M7: the mutation the original harness could not see --------------------

class M7_FreshEtaPerEvaluation(GeneralizedTFG):
    """Violates the eta-sharing convention by re-keying eta on every call.

    This is a real, consequential bug: it makes the conditional samples fresh
    at every loss evaluation and every recurrence, which changes both the
    gradient estimator's variance and its value. The original harness could not
    detect it because no predictor consumed ``eta_keys``.
    """

    def _call_log_f(self, x, n_t, eta_keys):
        self._m7 = getattr(self, "_m7", 0) + 1
        if eta_keys is not None:
            eta_keys = [(k[0], k[1], k[2] + 1000 * self._m7) for k in eta_keys]
        return super()._call_log_f(x, n_t, eta_keys)


def test_M7_fresh_eta_per_evaluation_is_now_detected():
    base_tr, _, _, _ = _run_engine(EtaConsumingPredictor, n_max=2, kind="time")

    schedule = DiffusionSchedule(T=T)
    denoiser = ToyDenoiser(d=SHAPE[1], T=T, seed=0, schedule=schedule)
    eps_theta = lambda x, t: denoiser(x, t)          # noqa: E731
    tape = NoiseTape(seed=17)
    predictor = EtaConsumingPredictor(tape, torch.full((8, 1), 0.4, dtype=torch.float64))
    cfg = TFGConfig(T=T, N_recur=2, N_iter=1, gamma_bar=0.0,
                    rho_scalar=0.4, mu_scalar=0.2, n_mc=1)
    cfg.n_schedule = NScheduleConfig(enabled=True, type="time", n_max=2)
    mut_tr = Tracer()
    M7_FreshEtaPerEvaluation(eps_theta, predictor, schedule, tape, cfg).run(
        SHAPE, trace=mut_tr)

    ok, report = compare_traces(base_tr, mut_tr, atol=1e-12)
    assert not ok, (
        "M7 NOT DETECTED: re-keying eta on every predictor evaluation left the "
        "trajectory unchanged, so the Component 1 conditional-seeding contract "
        "is still untested."
    )
    assert report["max_abs_err"] > 1e-8


# ---------------------------------------------------------------------------
# Fix 2: experiment-facing schedules carry no alphabar floor
# ---------------------------------------------------------------------------

def test_schedule_has_no_floor_by_default():
    """The floor changed alpha_T from 0.001 to 0.41 at T=100 -- i.e. it changed
    the diffusion process, not just the conditioning. It must be opt-in."""
    for T in (8, 100):
        sch = DiffusionSchedule(T=T)
        assert sch.min_alphabar is None
        # beta-clipping alone determines the tail.
        assert float(sch.alphabar[T]) < 1e-4, (
            f"T={T}: alphabar_T={float(sch.alphabar[T]):.3e} looks floored"
        )
        assert float(sch.alphabar[T]) > 0.0


def test_betas_are_consistent_with_alphabar():
    for T in (8, 100):
        sch = DiffusionSchedule(T=T)
        recon = torch.cumprod(1.0 - sch.betas, dim=0)
        err = (recon - sch.alphabar[1:]).abs().max().item()
        assert err < 1e-15, f"T={T}: betas disagree with alphabar by {err:.3e}"


def test_test_only_floor_is_still_available_and_explicit():
    sch = DiffusionSchedule(T=100, min_alphabar=1e-4)
    assert sch.min_alphabar == 1e-4
    assert float(sch.alphabar[100]) == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# Fix 3: gmm_l2 fails loudly instead of returning a graph-breaking zero
# ---------------------------------------------------------------------------

def test_gmm_l2_raises_on_nonfinite_instead_of_returning_zero():
    """A NaN input must raise, not silently yield a graph-breaking zero."""
    from tfg.gmm_l2 import gmm_l2_squared
    with pytest.raises(FloatingPointError, match="non-finite log-density"):
        gmm_l2_squared([[float("nan")]], [[[1.0]]], [1.0],
                       [[0.0]], [[[1.0]]], [1.0])


def test_gmm_l2_tolerates_a_single_zero_weight():
    """A pruned component (weight 0) gives log_w = -inf for that ROW only.

    That is legitimate -- exp(-inf) = 0 contributes nothing -- and must NOT
    raise. Only a fully degenerate term, where the maximum itself is
    non-finite, is an error. This pins the boundary between the two.
    """
    from tfg.gmm_l2 import gmm_l2_squared
    with_zero = gmm_l2_squared([[0.0], [1.0]], [[[1.0]], [[1.0]]], [1.0, 0.0],
                               [[0.0]], [[[1.0]]], [1.0], normalize_weights=False)
    without = gmm_l2_squared([[0.0]], [[[1.0]]], [1.0],
                             [[0.0]], [[[1.0]]], [1.0], normalize_weights=False)
    assert torch.isfinite(with_zero)
    assert abs(float(with_zero) - float(without)) < 1e-15


def test_gmm_l2_error_carries_diagnostics():
    from tfg.gmm_l2 import gmm_l2_squared
    try:
        gmm_l2_squared([[0.0]], [[[1.0]]], [0.0],
                       [[0.0]], [[[1.0]]], [1.0], normalize_weights=False)
    except FloatingPointError as e:
        msg = str(e)
        for field in ("max log term", "n non-finite", "log N range",
                      "log weight range", "means A"):
            assert field in msg, f"diagnostic {field!r} missing from error"
    else:
        pytest.fail("expected FloatingPointError")
