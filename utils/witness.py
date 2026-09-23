"""
Which conditional samples keep their gradient in the guidance loss, and how the kept
ones are reweighted.

The loss VALUE always uses all n samples; only k of them keep a gradient path to x
(the rest are detached). Two selection rules:

  uniform   k distinct rows uniformly at random (simple random sampling without
            replacement): inclusion probability pi_i = k / n.
  witness   rows drawn with probability proportional to the MMD witness score |w_i|,
            w(y_i) = mean_j k(y_i, y_j) - mean_j k(y_i, t_j) (y: conditional samples,
            t: target samples, k: the loss's kernel), i.e. large where the sampler puts
            too much or too little mass relative to G. A defensive floor mixes in the
            uniform distribution: p_i = floor / n + (1 - floor) |w_i| / sum |w|.
            Drawn by systematic probability-proportional-to-size sampling in a random
            order: exactly k distinct rows with inclusion probabilities
            pi_i = min(1, c p_i), c chosen so that sum pi_i = k.

Horvitz-Thompson rescaling (rescale="ht"): the gradient of each kept row is divided by
its inclusion probability pi_i, so the subsampled gradient is an unbiased estimate of
the full one (the gradient through all n samples), for either rule. Implemented as
y_i.detach() + (y_i - y_i.detach()) / pi_i: the value is unchanged, the gradient scales.
"""

import torch


@torch.no_grad()
def witness_scores(samples, target, kernel):
    """MMD witness w(y_i) = mean_j k(y_i, y_j) - mean_j k(y_i, t_j) for every row of samples,
    with the loss's kernel (a callable on the stacked [samples; target], as losses.functions.RBF)."""
    n = samples.shape[0]
    K = kernel(torch.vstack([samples.detach(), target.detach().to(samples.device)]))
    return K[:n, :n].mean(dim=1) - K[:n, n:].mean(dim=1)


def pps_inclusion(p, k):
    """Inclusion probabilities pi_i = min(1, c p_i) with sum pi_i = k (rows with c p_i > 1 are
    taken with certainty and c is recomputed on the rest)."""
    p = p.double()
    pi = torch.zeros_like(p)
    certain = torch.zeros_like(p, dtype=torch.bool)
    for _ in range(p.numel()):
        rest = ~certain
        m = k - int(certain.sum())
        mass = p[rest].sum()
        pi[rest] = m * p[rest] / mass if mass > 0 else float(m) / max(1, int(rest.sum()))
        pi[certain] = 1.0
        over = rest & (pi > 1.0)
        if not over.any():
            break
        certain |= over
    return pi.clamp(0.0, 1.0)


def systematic_pps(pi, generator=None):
    """Systematic sampling: exactly round(sum pi) distinct rows, row i kept with probability pi_i
    (pi_i <= 1). The rows are put in random order first, then a single uniform start u picks the
    rows whose cumulative interval contains one of u, u + 1, u + 2, ..."""
    n = pi.numel()
    k = int(round(float(pi.sum())))
    perm = torch.randperm(n, generator=generator)
    cum = torch.cumsum(pi[perm], 0)
    u = torch.rand(1, generator=generator, dtype=cum.dtype)
    points = u + torch.arange(k, dtype=cum.dtype)
    idx = torch.searchsorted(cum, points, right=True).clamp(max=n - 1)
    keep = torch.zeros(n, dtype=torch.bool)
    keep[perm[idx]] = True
    return keep


def select(samples, k, rule="uniform", target=None, kernel=None, floor=0.0):
    """Returns (keep [n] bool, pi [n] inclusion probabilities) for rule 'uniform' or 'witness'."""
    n = samples.shape[0]
    k = max(1, min(int(k), n))
    if rule == "uniform":
        keep = torch.zeros(n, dtype=torch.bool, device=samples.device)
        keep[torch.randperm(n, device=samples.device)[:k]] = True
        return keep, torch.full((n,), k / n, device=samples.device)
    if rule != "witness":
        raise ValueError(f"unknown selection rule {rule!r} (known: uniform, witness)")
    a = witness_scores(samples, target, kernel).abs().double().cpu()
    p = a / a.sum() if a.sum() > 0 else torch.full((n,), 1.0 / n, dtype=torch.float64)
    p = (1.0 - floor) * p + floor / n
    pi = pps_inclusion(p, k)
    keep = systematic_pps(pi)
    return keep.to(samples.device), pi.to(samples.device, samples.dtype)


def reweight(samples, keep, pi, rescale="ht"):
    """Kept rows keep their gradient (divided by pi_i under rescale='ht'), the rest are detached."""
    d = samples.detach()
    kept = samples if rescale == "none" else d + (samples - d) / pi[:, None].clamp_min(1e-12)
    return torch.where(keep[:, None], kept, d)
