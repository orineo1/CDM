"""Sampled gradient estimators for the conditional-GMM objective.

STATUS: PARKED, NOT USED IN THE ESTIMATOR STUDY.
------------------------------------------------
``pathwise_plus_score`` below was written and validated (its Monte-Carlo mean
converges to the exact oracle gradient on the diagonal-covariance isolation
case) before the study scope was corrected.  It is retained because that
validation is real and may be wanted later, but it is deliberately NOT the
estimator of record: MLGD-F's conditional sampler is a trained consistency
model, not this analytic GMM sampler, so studying this estimator would measure
something the paper does not do.  Nothing in the estimator study imports it.
See the call-graph finding in the Checkpoint report.


The objective is ``L(x) = MMD^2( P(Y|X=x), G )`` with a Gaussian kernel.  The
population value and gradient are exact and sampling-free (``tfg/gmm_mmd.py``),
so they serve as ground truth.  Here we build estimators that use only ``n``
conditional samples, which is what ``n_t`` controls.

The conditional is a mixture whose EVERY ingredient depends on x:

    P(Y|X=x) = sum_c w_c(x) N( m_c(x), S_c(x) )

so a correct gradient estimator needs two terms:

  1. **Pathwise (reparameterisation)** through ``m_c(x)`` and ``S_c(x)``:
     draw ``eps ~ N(0, I)`` and set ``y = m_c(x) + L_c(x) eps`` with
     ``L_c L_c^T = S_c``.  Differentiating ``y`` w.r.t. ``x`` is valid because
     ``eps`` does not depend on ``x``.

  2. **Score function (REINFORCE)** for the CATEGORICAL component choice, whose
     distribution ``w(x)`` depends on x and which cannot be reparameterised:

         d/dx E[h] = E[ d/dx h(y(x)) ] + E[ h(y) * sum_i d/dx log w_{c_i}(x) ]

Dropping term 2 is exactly the defect in ``dist_utils.generate_mog_samples``:
its Gumbel-softmax is applied to the global weight vector and is then discarded
by ``torch.multinomial``, so no gradient reaches ``w(x)`` at all.  On a
diagonal-covariance mixture, where term 1 vanishes identically, that path
recovers 0% of the true gradient.

Estimators provided
-------------------
``pathwise_plus_score``   both terms; unbiased. The estimator of record.
``pathwise_only``         term 1 only; biased. Ablation isolating the defect.
``gumbel``                the repository's current path. Ablation only, never
                          ground truth.

A note on the V- vs U-statistic, which matters for the bias table.  The
repository's MMD is the biased V-statistic, dividing the ``xx`` block by
``n^2`` including the diagonal.  Since ``k(y, y) = 1`` for every y, the
diagonal contributes exactly ``1/n`` -- a constant in x, so it does not bias
the gradient additively.  But it does change the normalisation: the V-statistic
gradient equals ``(n-1)/n`` times the U-statistic gradient on the ``xx`` term.
That is a genuine multiplicative ``O(1/n)`` gradient bias, and it is reported
separately from Monte-Carlo variance.
"""

import torch

from tfg.gmm_mmd import kernel_mean_embedding, kernel_matrix, population_mmd2
from tfg.oracle import conditional_params

ESTIMATORS = ("pathwise_plus_score", "pathwise_only", "gumbel")


# ---------------------------------------------------------------------------
# Exact reference
# ---------------------------------------------------------------------------

def exact_objective(x, params, sigma=1.0):
    """Population ``MMD^2(P(Y|X=x), G)``; exact, differentiable."""
    means, covs, weights = conditional_params(
        x, params["mu_list"], params["Sigma_list"], params["alpha"])
    tgt_cov = torch.stack([v for v in params["target_variances"]])
    return population_mmd2(means, covs, weights,
                           params["target_means"], tgt_cov,
                           params["target_weights"], sigma=sigma)


def exact_gradient(x, params, sigma=1.0):
    """Exact ``grad_x L(x)``. Zero variance -- this is the ground truth."""
    x = x.detach().clone().requires_grad_(True)
    val = exact_objective(x, params, sigma=sigma)
    g, = torch.autograd.grad(val, x)
    return val.detach(), g.detach()


# ---------------------------------------------------------------------------
# Sampled estimators
# ---------------------------------------------------------------------------

def _mixture_pieces(x, params):
    means, covs, weights = conditional_params(
        x, params["mu_list"], params["Sigma_list"], params["alpha"])
    chol = torch.linalg.cholesky(covs)
    return means, covs, weights, chol


def _sample_objective(y, params, sigma, unbiased_xx):
    """``MMD^2`` estimate from conditional samples ``y`` and the EXACT target.

    Using the exact target for the cross term isolates the randomness to the
    ``n`` conditional draws, which is the quantity under study. ``unbiased_xx``
    selects the U-statistic (diagonal excluded) or the V-statistic.
    """
    n = y.shape[0]
    K = kernel_matrix(y, sigma=sigma)
    if unbiased_xx:
        if n < 2:
            xx = torch.zeros((), dtype=y.dtype)
        else:
            xx = (K.sum() - K.diagonal().sum()) / (n * (n - 1))
    else:
        xx = K.sum() / (n * n)
    tgt_cov = torch.stack([v for v in params["target_variances"]])
    pq = kernel_mean_embedding(y, params["target_means"], tgt_cov,
                               params["target_weights"], sigma=sigma).mean()
    return xx - 2.0 * pq          # the qq term is constant in x


def sampled_gradient(x, params, n, tape_keys=None, generator=None,
                     sigma=1.0, estimator="pathwise_plus_score",
                     unbiased_xx=True, baseline=0.0, tau=7.5):
    """One realisation of a sampled gradient estimate at ``x``.

    Parameters
    ----------
    n:
        Number of conditional samples (the ``n_t`` of Component 1).
    estimator:
        One of :data:`ESTIMATORS`.
    unbiased_xx:
        U-statistic (True) or the repository's V-statistic (False).
    baseline:
        Constant control variate for the score term. Any constant leaves the
        estimator unbiased because ``E[sum_i grad log w_{c_i}] = 0``.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")

    x = x.detach().clone().requires_grad_(True)
    means, covs, weights, chol = _mixture_pieces(x, params)
    K, d_y = means.shape

    if estimator == "gumbel":
        # The repository's current path, reproduced for ablation: a
        # Gumbel-softmax over the weights that is then discarded by a hard
        # categorical draw, so no gradient reaches w(x).
        probs = torch.nn.functional.gumbel_softmax(
            torch.log(weights.clamp_min(1e-30)), tau=tau, hard=False)
        comp = torch.multinomial(probs.detach(), n, replacement=True)
    else:
        comp = torch.multinomial(weights.detach(), n, replacement=True,
                                 generator=generator)

    eps = torch.randn(n, d_y, dtype=means.dtype, generator=generator)
    # Pathwise: y depends on x through the component mean and Cholesky factor.
    y = means[comp] + torch.einsum("nij,nj->ni", chol[comp], eps)

    obj = _sample_objective(y, params, sigma, unbiased_xx)

    if estimator == "pathwise_plus_score":
        log_w = torch.log(weights.clamp_min(1e-300))
        score = log_w[comp].sum()
        surrogate = obj + (obj.detach() - baseline) * score
    else:
        surrogate = obj

    g, = torch.autograd.grad(surrogate, x, allow_unused=True)
    if g is None:
        g = torch.zeros_like(x)
    return obj.detach(), g.detach()


def replicate(x, params, n, reps, seed=0, **kw):
    """``reps`` independent estimates at ``x``. Returns ``(values, grads)``."""
    gen = torch.Generator().manual_seed(int(seed))
    vals, grads = [], []
    for _ in range(reps):
        v, g = sampled_gradient(x, params, n, generator=gen, **kw)
        vals.append(v)
        grads.append(g)
    return torch.stack(vals), torch.stack(grads)


def estimator_stats(x, params, n, reps, seed=0, sigma=1.0, **kw):
    """Bias / variance / MSE of a sampled estimator against the exact gradient.

    All quantities carry Monte-Carlo standard errors, since with ``reps``
    replications the reported bias is itself an estimate.
    """
    _, g_ref = exact_gradient(x, params, sigma=sigma)
    _, grads = replicate(x, params, n, reps, seed=seed, sigma=sigma, **kw)

    mean = grads.mean(0)
    bias = mean - g_ref
    centred = grads - mean
    # Per-replication squared error and squared deviation, so that both the
    # point estimates and their standard errors come from the same sample.
    sq_err = ((grads - g_ref) ** 2).sum(1)
    sq_dev = (centred ** 2).sum(1)

    reps_f = float(grads.shape[0])
    var = sq_dev.mean() * reps_f / max(1.0, reps_f - 1.0)
    mse = sq_err.mean()

    # SE of the mean gradient (per coordinate), combined into an SE on ||bias||.
    se_mean = grads.std(0, unbiased=True) / reps_f ** 0.5
    bias_norm = bias.norm()
    se_bias_norm = (se_mean.norm() if float(bias_norm) == 0.0
                    else (bias * se_mean).norm() / bias_norm)

    return {
        "n": int(n),
        "reps": int(reps),
        "g_ref": g_ref.tolist(),
        "g_mean": mean.tolist(),
        "bias": bias.tolist(),
        "bias_norm": float(bias_norm),
        "se_bias_norm": float(se_bias_norm),
        "variance": float(var),
        "se_variance": float(sq_dev.std(unbiased=True) / reps_f ** 0.5),
        "mse": float(mse),
        "se_mse": float(sq_err.std(unbiased=True) / reps_f ** 0.5),
        "rel_error": float((mse ** 0.5) / max(1e-30, float(g_ref.norm()))),
        "cosine": float(torch.nn.functional.cosine_similarity(
            mean.reshape(1, -1), g_ref.reshape(1, -1)).item()),
    }
