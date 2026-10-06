"""Population loss and tempered-target masses of the Figure-1 'hard' toy, computed independently.

Rebuilds the mixture of fig1_toy_example.make_setting("hard") in plain numpy (no torch, no project
code) and prints
  - the population loss L(x) = ||P(Y|x) - P(Y|x*=2)||^2 (exact squared L2 between 1-D Gaussian
    mixtures, the quantity of cdm.utils.dist.gmm_l2_distance) at the cluster centres;
  - the mass of Q_beta(x) ∝ P(x) exp(-beta L(x)) in each cluster, for the beta of panel (d).
Used for the numbers in the paper's toy appendix (Setup and beta-sweep paragraphs).

    python experiments/synthetic/pipeline/toy_loss_check.py
"""
import math

import numpy as np

# (x-centre, y-means, y-variance, x-y correlations), as in fig1_toy_example.make_setting("hard")
CLUSTERS = [(-6., [2., 0., -2.], 0.15, [0.5, 0.5, 0.5]),
            (-2., [2.], 0.9, [-0.7]),
            (2., [2., -2.], 0.4, [0.4, -0.4]),
            (6., [0.], 3.0, [-0.5])]
VX, X_STAR, BETAS = 0.25, 2.0, (0, 2, 8, 16)

comps = [(cx, cy, vy, r * math.sqrt(VX * vy), 1 / (4 * len(ys)))
         for cx, ys, vy, rs in CLUSTERS for cy, r in zip(ys, rs)]


def npdf(x, m, v):
    return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)


def cond(x):
    """P(Y | X = x) as (means, variances, weights)."""
    m = [cy + cxy / VX * (x - cx) for cx, cy, vy, cxy, a in comps]
    v = [vy - cxy ** 2 / VX for cx, cy, vy, cxy, a in comps]
    w = np.array([a * npdf(x, cx, VX) for cx, cy, vy, cxy, a in comps])
    return m, v, w / w.sum()


def inner(p, q):
    return sum(wi * wj * npdf(mi, mj, vi + vj) for mi, vi, wi in zip(*p) for mj, vj, wj in zip(*q))


G = cond(X_STAR)


def loss(x):
    p = cond(x)
    return inner(p, p) - 2 * inner(p, G) + inner(G, G)


for x in (-6, -2, 2, 6):
    print(f"population loss at x = {x:+d}: {loss(x):.4f}")

xs = np.linspace(-9, 9, 3601)
px = sum(a * npdf(xs, cx, VX) for cx, cy, vy, cxy, a in comps)
L = np.array([loss(x) for x in xs])
print(f"argmin of the loss on the grid: x = {xs[L.argmin()]:.3f}")
for b in BETAS:
    q = px * np.exp(-b * L)
    q /= q.sum()
    mass = {c: q[(xs > c - 2) & (xs < c + 2)].sum() for c in (-6, -2, 2, 6)}
    print(f"beta = {b:2d}: Q mass by cluster " + "  ".join(f"x={c:+d}: {m:.3f}" for c, m in mass.items()))
