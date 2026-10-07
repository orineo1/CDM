"""Minimum-loss path between two minimizers (elastic band) with a DENSE check of the relaxed path:
the loss is evaluated on `sub` points of every straight segment between consecutive images, so a band that
jumps over a barrier between two images is not reported as barrier-free."""
import numpy as np
from scipy.optimize import minimize


def band_path(obj, xa, xb, kT, n=32, springs=(1e-3, 1e-1, 1.0), sub=12, maxiter=3000, bounds=None):
    """Returns dict(line_max, band_max_images, band_max_dense) where band_max_dense is the minimum over
    spring constants of the max loss along the densely evaluated relaxed path."""
    D = len(xa)
    t = np.linspace(0, 1, n)[1:-1, None]
    z0 = ((1 - t) * xa + t * xb).ravel()
    q = np.linspace(0, 1, sub + 1)[:-1]
    line = max(obj((1 - s) * xa + s * xb)[0] for s in np.linspace(0, 1, n * sub))
    best_img, best_dense = np.inf, np.inf
    for k in springs:
        def f(z):
            X = np.vstack([xa, z.reshape(-1, D), xb])
            vals, grads = zip(*[obj(x) for x in X[1:-1]])
            dX = np.diff(X, axis=0) / kT
            return sum(vals) + 0.5 * k * (dX ** 2).sum(), (np.array(grads) + k * (dX[:-1] - dX[1:]) / kT).ravel()
        r = minimize(f, z0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter),
                     bounds=None if bounds is None else bounds * (n - 2))
        X = np.vstack([xa, r.x.reshape(-1, D), xb])
        img = max(obj(x)[0] for x in X)
        dense = max(obj((1 - s) * X[i] + s * X[i + 1])[0] for i in range(len(X) - 1) for s in q)
        dense = max(dense, obj(xb)[0])
        best_img, best_dense = min(best_img, img), min(best_dense, dense)
    return dict(line_max=float(line), band_max_images=float(best_img), band_max_dense=float(best_dense))
