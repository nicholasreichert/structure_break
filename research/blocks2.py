"""Candidate feature blocks for v8, each horizon-calibrated like calfeat (stride-aligned).

  garch : normal scores whitened by a GARCH(1,1) fitted per series on the history
          (variance-targeted to 1, Gaussian QMLE), instead of a fixed-lambda EWMA.
  ord   : one-hot order-3 ordinal patterns of (x_{t-2}, x_{t-1}, x_t) -- dependence that is
          invariant to the marginal distribution.

usage: python research/blocks2.py garch ord   -> cache/<name>_{train,test_reduced}.npy
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import ndtri

from calfeat import BURN, EWMA_ALPHA, KINDS, STATS, WINDOWS, _horizon_sd, _sd_at
from common import CACHE, CACHE_STRIDE, load_test_reduced, load_train


def _g(x, nh):
    sh = np.sort(x[:nh])
    u = ((np.searchsorted(sh, x, "left") + np.searchsorted(sh, x, "right")) / 2 + 0.5) / (nh + 1)
    return ndtri(u)


def _garch_var(g2m1, a, b):
    """Conditional variance v_t (of g_t, given the past) with unit unconditional variance."""
    u = lfilter([0.0, a], [1.0, -b], g2m1)
    return np.maximum(1.0 + u, 1e-3)


def fit_garch(gh):
    g2m1 = gh * gh - 1.0

    def nll(p):
        a, b = p
        if a + b >= 0.999:
            return 1e10
        v = _garch_var(g2m1, a, b)
        return 0.5 * float(np.sum(np.log(v) + (g2m1 + 1.0) / v))

    best = None
    for x0 in ((0.05, 0.90), (0.02, 0.5)):
        r = minimize(nll, x0, method="L-BFGS-B", bounds=[(1e-4, 0.5), (0.0, 0.998)])
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def _kinds(b):
    b1 = np.r_[0.0, b[:-1]]
    a, a1 = np.abs(b), np.abs(b1)
    return [b, b * b, a, b * b1, a * a1]


def streams_garch(x, nh):
    g = _g(x, nh)
    a, b = fit_garch(g[:nh])
    v = _garch_var(g * g - 1.0, a, b)
    return np.array(_kinds(g / np.sqrt(v)))


def streams_ord(x, nh):
    x2, x1 = np.r_[x[0], x[0], x[:-2]], np.r_[x[0], x[:-1]]
    # pattern id from the three pairwise comparisons (6 reachable codes out of 8)
    code = (x1 > x2).astype(int) * 4 + (x > x1).astype(int) * 2 + (x > x2).astype(int)
    return np.array([(code == c).astype(float) for c in (0, 1, 3, 4, 6, 7)])


BLOCKS = {
    "garch": (streams_garch, [f"gw_{k}" for k in KINDS]),
    "ord": (streams_ord, [f"op{c}" for c in (0, 1, 3, 4, 6, 7)]),
}


def calibrate(S, nh, T):
    E = S - S[:, BURN:nh].mean(1, keepdims=True)
    eh, eo = E[:, BURN:nh], E[:, nh:]
    sdg = _horizon_sd(eh)
    n = np.arange(1, T + 1)
    cs = np.cumsum(eo, 1)
    feats = [cs / _sd_at(sdg, n)]
    for w in WINDOWS:
        ws = cs.copy()
        if T > w:
            ws[:, w:] -= cs[:, :-w]
        feats.append(ws / _sd_at(sdg, np.minimum(n, w)))
    al = EWMA_ALPHA
    eh_ew = lfilter([al], [1, -(1 - al)], eh, axis=1)[:, int(3 / al):]
    ew = lfilter([al], [1, -(1 - al)], eo, axis=1)
    feats.append(ew / np.maximum(eh_ew.std(1, keepdims=True), 1e-9))
    return np.stack(feats, 2).transpose(1, 0, 2).reshape(T, -1).astype(np.float32)


def series_block(name, x_hist, x_online):
    x = np.concatenate([x_hist, x_online]).astype(np.float64)
    return calibrate(BLOCKS[name][0](x, len(x_hist)), len(x_hist), len(x_online))


def names(name):
    return [f"{name}_{s}_{st}" for s in BLOCKS[name][1] for st in STATS]


def _one(args):
    name, h, o = args
    return series_block(name, h, o)[::CACHE_STRIDE]


if __name__ == "__main__":
    for name in sys.argv[1:]:
        for split, fn in (("train", load_train), ("test_reduced", load_test_reduced)):
            ds = fn()
            t0 = time.time()
            with Pool(20) as pool:
                F = pool.map(_one, [(name, *ds.segments(k)) for k in range(len(ds))], chunksize=20)
            np.save(os.path.join(CACHE, f"{name}_{split}.npy"), np.concatenate(F))
            print(f"{name}/{split}: {len(ds)} series in {time.time() - t0:.1f}s", flush=True)
