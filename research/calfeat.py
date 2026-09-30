"""Per-series calibrated + volatility-whitened evidence block (vectorised, causal).

Why: windows *inside* a history give a variance z-stat with sd ~1.9 instead of 1 -- the
series have volatility clustering, so an iid null (or a short-bandwidth Bartlett LRV) is
badly miscalibrated, and miscalibrated differently per series, which costs cross-sectional
AUC. Two fixes, both measured on the true-tau oracle (GBDT 0.547 -> 0.607):

  * whitening: divide the normal scores by an EWMA volatility (lambda 0.97 / 0.99) carried
    from the start of the history, so the online points are compared with *current* vol;
  * calibration: scale every window/sum/EWMA statistic by the sd of the same statistic,
    at the same horizon, computed over rolling windows of the series' own history.

Output rows follow research/common.py's CACHE_STRIDE, so they align with cache/train.npz.
usage: python research/calfeat.py
"""

from __future__ import annotations

import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtri

from common import CACHE, CACHE_STRIDE, load_test_reduced, load_train

BURN = 200
LAMBDAS = (0.97, 0.99)
WINDOWS = (16, 64, 256)
EWMA_ALPHA = 0.03
GRID = 2 ** np.arange(0, 11)  # horizons 1..1024 for the variance-time curve
BASES = ["g"] + [f"w{int(l * 100)}" for l in LAMBDAS]
KINDS = ["lvl", "sq", "abs", "ac1", "vol1"]
STATS = ["sum"] + [f"w{w}" for w in WINDOWS] + ["ewma"]
FEATURE_NAMES = [f"c_{b}_{k}_{s}" for b in BASES for k in KINDS for s in STATS]


def _ewma_prev(v2, lam, v0=1.0):
    """v_t = lam v_{t-1} + (1-lam) v2_t; returns v_{t-1} aligned to t (causal)."""
    v, _ = lfilter([1 - lam], [1, -lam], v2, zi=[lam * v0])
    return np.r_[v0, v[:-1]]


def _streams(x, nh):
    h = x[:nh]
    sh = np.sort(h)
    u = ((np.searchsorted(sh, x, "left") + np.searchsorted(sh, x, "right")) / 2 + 0.5) / (nh + 1)
    g = ndtri(u)
    bases = [g] + [g / np.sqrt(_ewma_prev(g * g, lam)) for lam in LAMBDAS]
    out = []
    for b in bases:
        b1 = np.r_[0.0, b[:-1]]
        a, a1 = np.abs(b), np.abs(b1)
        out += [b, b * b, a, b * b1, a * a1]
    return np.array(out)  # (n_streams, N)


def _horizon_sd(eh):
    """sd of rolling k-sums over the history, for k in GRID (rows: streams)."""
    c = np.concatenate([np.zeros((eh.shape[0], 1)), np.cumsum(eh, 1)], 1)
    n = eh.shape[1]
    sds = []
    for k in GRID:
        if k < n // 2:
            sds.append((c[:, k:] - c[:, :-k]).std(1))
        else:
            sds.append(sds[-1] * np.sqrt(k / GRID[len(sds) - 1]))  # extend at constant VR
    return np.maximum(np.array(sds).T, 1e-9)  # (streams, len(GRID))


def _sd_at(sdg, k):
    """log-log interpolation of the horizon sd at horizons k (array)."""
    lk = np.log2(np.clip(k, 1, GRID[-1]))
    lo = np.floor(lk).astype(int)
    hi = np.minimum(lo + 1, len(GRID) - 1)
    f = lk - lo
    out = np.exp((1 - f) * np.log(sdg[:, lo]) + f * np.log(sdg[:, hi]))
    big = k > GRID[-1]
    if big.any():
        out[:, big] *= np.sqrt(k[big] / GRID[-1])
    return out


def series_block(x_hist, x_online) -> np.ndarray:
    x = np.concatenate([x_hist, x_online]).astype(np.float64)
    nh, T = len(x_hist), len(x_online)
    S = _streams(x, nh)
    ctr = S[:, BURN:nh].mean(1, keepdims=True)
    E = S - ctr
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
    # EWMA from 0 at the online start; null sd from the same EWMA run over the history
    a = EWMA_ALPHA
    eh_ew = lfilter([a], [1, -(1 - a)], eh, axis=1)[:, int(3 / a):]
    ew = lfilter([a], [1, -(1 - a)], eo, axis=1)
    feats.append(ew / np.maximum(eh_ew.std(1, keepdims=True), 1e-9))
    F = np.stack(feats, 2)  # (streams, T, stats)
    return F.transpose(1, 0, 2).reshape(T, -1).astype(np.float32)


def _one(args):
    return series_block(*args)[::CACHE_STRIDE]


def compute(ds, processes=10):
    jobs = [ds.segments(k) for k in range(len(ds))]
    t0 = time.time()
    with Pool(processes) as pool:
        F = pool.map(_one, jobs, chunksize=20)
    print(f"cal block: {len(ds)} series in {time.time() - t0:.1f}s", flush=True)
    return np.concatenate(F)


if __name__ == "__main__":
    from common import ts_auc

    for name, fn in (("train", load_train), ("test_reduced", load_test_reduced)):
        np.save(os.path.join(CACHE, f"cal_{name}.npy"), compute(fn()))
    ds = load_train()
    F = np.load(os.path.join(CACHE, "cal_train.npy"))
    y = np.concatenate([ds.labels(k)[::CACHE_STRIDE] for k in range(len(ds))])
    step = np.concatenate([np.arange(0, n, CACHE_STRIDE) for n in ds.online_len])
    rows = sorted(((ts_auc(y, np.abs(F[:, j]), step), nm) for j, nm in enumerate(FEATURE_NAMES)), reverse=True)
    for auc, nm in rows[:25]:
        print(f"{auc:.4f}  |{nm}|")
