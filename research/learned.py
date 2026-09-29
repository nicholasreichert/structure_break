"""Nonlinear next-value predictor fitted on each series' history (vectorised research version).

Model: ridge regression of g_t (normal score of x_t) on [lags of g, random Fourier features of
the lag vector]. The random features approximate a Gaussian-kernel regression, so the fit can
pick up threshold / volatility / other nonlinear dependence that AR(p) cannot. The ridge penalty
is chosen per series by leave-one-out error (closed form from the SVD), and the history's
error stream uses LOO residuals, so it is on the same out-of-sample footing as the online errors.

A second ridge on the same features predicts the squared error (the conditional variance), so
e^2 / v_hat measures surprise relative to how noisy the model expected that step to be.
"""

import numpy as np
from scipy.special import ndtri


def normal_scores(x, ref_sorted):
    n = len(ref_sorted)
    rank = (np.searchsorted(ref_sorted, x, "left") + np.searchsorted(ref_sorted, x, "right")) / 2.0
    return ndtri((rank + 0.5) / (n + 1.0))


def lag_matrix(g, p):
    """Row t = (g[t-1], ..., g[t-p]); rows before p are padded with zeros."""
    L = np.zeros((len(g), p))
    for i in range(p):
        L[i + 1 :, i] = g[: len(g) - i - 1]
    return L


class RFF:
    def __init__(self, p, D, bandwidth, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(p, D)) / bandwidth
        self.b = rng.uniform(0, 2 * np.pi, size=D)
        self.c = np.sqrt(2.0 / D)

    def __call__(self, L):
        return np.hstack([L, self.c * np.cos(L @ self.W + self.b)])


LAMBDAS = np.array([1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0])


def ridge_loo(F, y):
    """Ridge with intercept; lambda (relative to n) by LOO. Returns predict fn and LOO residuals."""
    n = len(y)
    fm, ym = F.mean(0), y.mean()
    U, s, Vt = np.linalg.svd(F - fm, full_matrices=False)
    uy = U.T @ (y - ym)
    s2 = s * s
    best = None
    for lam in LAMBDAS * n:
        sh = s2 / (s2 + lam)
        fit = U @ (sh * uy) + ym
        h = (U * U) @ sh + 1.0 / n
        loo = (y - fit) / (1.0 - h)
        mse = float(loo @ loo)
        if best is None or mse < best[0]:
            best = (mse, lam, loo)
    _, lam, loo = best
    beta = Vt.T @ (s / (s2 + lam) * uy)
    icpt = ym - fm @ beta
    return beta, icpt, loo, lam / n


def learned_streams(x, nh, p=10, D=100, bandwidth=3.0, seed=0):
    """Error streams over x = hist+online (history part is LOO / out-of-sample)."""
    h = x[:nh]
    g = normal_scores(x, np.sort(h))
    L = lag_matrix(g, p)
    phi = RFF(p, D, bandwidth, seed)
    F = phi(L)
    tr = slice(p, nh)
    beta, icpt, loo, lam = ridge_loo(F[tr], g[tr])
    e = g - (F @ beta + icpt)
    e[tr] = loo
    # conditional variance of the error
    e2 = e * e
    bv, iv, loov, lamv = ridge_loo(F[tr], e2[tr])
    v = F @ bv + iv
    vfloor = 0.1 * e2[tr].mean()
    # history variance prediction must also be out-of-sample: v_loo = e2 - loov
    v[tr] = e2[tr] - loov
    v = np.maximum(v, vfloor)
    # linear AR(p) on g for a same-footing comparison
    ba, ia, looa, _ = ridge_loo(L[tr], g[tr])
    ea = g - (L @ ba + ia)
    ea[tr] = looa
    return dict(
        nl_e=e, nl_e2=e2, nl_abse=np.abs(e), nl_std2=e2 / v, nl_logv=np.log(v),
        nl_gain=ea * ea - e2,  # how much the nonlinear model beats AR at this step
        ar_e2=ea * ea,
    ), dict(lam=lam, lamv=lamv)
