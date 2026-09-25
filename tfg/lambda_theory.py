"""Generalised one-step mixing weight, without unbiasedness or independence.

Setup. Let g_t be the TRUE gradient at the current state and

    e_t     = ghat_t     - g_t          (current estimator error)
    e_{t+1} = ghat_{t+1} - g_t          (CACHED estimator error, measured
                                         against the CURRENT true gradient)

Note e_{t+1} is defined relative to g_t, so it absorbs the drift; nothing is
assumed about its mean or its correlation with e_t.

The aggregated estimate is  gtilde = (1-l) ghat_t + l ghat_{t+1}, hence

    gtilde - g_t = (1-l) e_t + l e_{t+1}

    MSE(l) = (1-l)^2 E||e_t||^2 + 2 l(1-l) E[e_t . e_{t+1}] + l^2 E||e_{t+1}||^2

    d/dl = -2(1-l) E||e_t||^2 + 2(1-2l) E[e_t . e_{t+1}] + 2l E||e_{t+1}||^2 = 0

    =>  l ( E||e_t||^2 + E||e_{t+1}||^2 - 2 E[e_t . e_{t+1}] )
            = E||e_t||^2 - E[e_t . e_{t+1}]

    =>  l* = clip_[0,1]  ( E||e_t||^2 - E[e_t . e_{t+1}] ) / E||e_t - e_{t+1}||^2

using E||e_t - e_{t+1}||^2 = E||e_t||^2 + E||e_{t+1}||^2 - 2E[e_t . e_{t+1}].
MSE(l) is convex in l (its second derivative is 2 E||e_t - e_{t+1}||^2 >= 0), so
the stationary point is the minimum and clipping to [0,1] is exact.

REDUCTION to the simplified rule. Assume
  (A1) independence of ghat_t and ghat_{t+1};
  (A2) unbiasedness, E[ghat_s] = g_s for each s.
Write eps_s = ghat_s - g_s and delta = g_{t+1} - g_t, so e_{t+1} = eps_{t+1} + delta
with ||delta||^2 = D_t^2 deterministic. Then
    E||e_t||^2      = V_t
    E[e_t . e_{t+1}] = E[eps_t . eps_{t+1}] + E[eps_t] . delta = 0        (A1, A2)
    E||e_{t+1}||^2  = V_{t+1} + 2 E[eps_{t+1}] . delta + D_t^2
                    = V_{t+1} + D_t^2                                     (A2)
and therefore
    l* = V_t / (V_t + V_{t+1} + D_t^2)
which is the previously used rule. It is exactly the special case in which the
cross term vanishes and the cached error decomposes into variance plus squared
drift -- both of which fail for a biased, correlated, heavy-tailed estimator.
"""

import torch


def lambda_general(e_t, e_next):
    """Generalised optimum from paired error samples. Shapes (R, d)."""
    Ett = float((e_t ** 2).sum(1).mean())
    Ecross = float((e_t * e_next).sum(1).mean())
    Ediff = float(((e_t - e_next) ** 2).sum(1).mean())
    if Ediff <= 0:
        return 0.0, {"E_et2": Ett, "E_cross": Ecross, "E_diff2": Ediff}
    lam = (Ett - Ecross) / Ediff
    return float(min(1.0, max(0.0, lam))), {
        "E_et2": Ett, "E_cross": Ecross, "E_diff2": Ediff, "lambda_raw": lam}


def lambda_simplified(g_t, g_next, ghat_t, ghat_next):
    """V_t / (V_t + V_{t+1} + D_t^2) -- the independence/unbiasedness special case."""
    V_t = float(((ghat_t - ghat_t.mean(0)) ** 2).sum(1).mean())
    V_n = float(((ghat_next - ghat_next.mean(0)) ** 2).sum(1).mean())
    D2 = float(((g_t - g_next) ** 2).sum())
    denom = V_t + V_n + D2
    return (0.0 if denom <= 0 else float(min(1.0, max(0.0, V_t / denom)))), \
        {"V_t": V_t, "V_next": V_n, "D2": D2}


def empirical_optimum(ghat_t, ghat_next, g_t, grid=None):
    """Direct grid minimisation of the paired MSE -- the ground truth here."""
    grid = torch.linspace(0, 1, 501, dtype=torch.float64) if grid is None else grid
    mses = torch.tensor([float((((1 - l) * ghat_t + l * ghat_next) - g_t)
                               .pow(2).sum(1).mean()) for l in grid])
    i = int(mses.argmin())
    return float(grid[i]), float(mses[i]), mses


def verify(model, S_G, bandwidth, x_t=-6.0, pairs=(("low", -5.98), ("med", -5.75),
                                                   ("high", -5.0)),
           n=32, reps=2000, ref_reps=300):
    """Numerical verification of both rules against the empirical optimum."""
    from tfg.run_estimator_v2 import reference, replicate
    out = []
    rt = reference(model, x_t, S_G, bandwidth, ref_reps, "thry-t")
    gt = replicate(model, x_t, n, reps, S_G, bandwidth, "thry-gt").reshape(-1, 1)
    g_t = torch.tensor([[rt["g_ref"]]], dtype=torch.float64)
    for lab, x_n in pairs:
        rn = reference(model, x_n, S_G, bandwidth, ref_reps, "thry-n")
        gn = replicate(model, x_n, n, reps, S_G, bandwidth, "thry-gn").reshape(-1, 1)
        g_n = torch.tensor([[rn["g_ref"]]], dtype=torch.float64)
        lg, ig = lambda_general(gt - g_t, gn - g_t)
        ls, isimp = lambda_simplified(g_t, g_n, gt, gn)
        le, mse_e, _ = empirical_optimum(gt, gn, g_t)
        mse = lambda l: float((((1 - l) * gt + l * gn) - g_t).pow(2).sum(1).mean())
        out.append({"pair": lab, "x_t": x_t, "x_next": x_n, "n": n, "reps": reps,
                    "g_ref_t": rt["g_ref"], "g_ref_next": rn["g_ref"],
                    "snr_t": rt["snr"], "snr_next": rn["snr"],
                    "lambda_empirical": le, "lambda_general": lg,
                    "lambda_simplified": ls,
                    "mse_at_empirical": mse_e, "mse_at_general": mse(lg),
                    "mse_at_simplified": mse(ls), "mse_at_zero": mse(0.0),
                    **ig, **isimp,
                    "cross_term_relative": ig["E_cross"] / ig["E_et2"]})
    return out
