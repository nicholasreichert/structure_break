"""Second calibrated block: AR(20)-residual normal scores (raw + vol-whitened), longer lag
products and tail/centre rates, all with the same per-series horizon calibration as calfeat.
usage: python research/calfeat2.py   -> cache/cal2_{train,test_reduced}.npy
"""

from __future__ import annotations

import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtri

import calfeat
from calfeat import BURN, EWMA_ALPHA, WINDOWS, _ewma_prev, _horizon_sd, _sd_at
from common import CACHE, CACHE_STRIDE, load_test_reduced, load_train

AR_P = 20
TAIL_Z = 1.6448536269514722
CENTER_G = 0.2533471031357997


def _ns(ref_sorted, v):
    n = len(ref_sorted)
    u = ((np.searchsorted(ref_sorted, v, "left") + np.searchsorted(ref_sorted, v, "right")) / 2 + 0.5) / (n + 1)
    return ndtri(u)


def _lag(b, L):
    return np.r_[np.zeros(L), b[:-L]]


def _streams(x, nh):
    h = x[:nh]
    g = _ns(np.sort(h), x)
    z = (x - h.mean()) / h.std()
    X = np.column_stack([_lag(z, i + 1) for i in range(AR_P)])
    phi, *_ = np.linalg.lstsq(X[AR_P:nh], z[AR_P:nh], rcond=None)
    r = z - X @ phi
    r[:AR_P] = 0.0
    rg = _ns(np.sort(r[AR_P:nh]), r)
    out = []
    for b in (rg, rg / np.sqrt(_ewma_prev(rg * rg, 0.99))):
        b1 = _lag(b, 1)
        a, a1 = np.abs(b), np.abs(b1)
        out += [b, b * b, a, b * b1, a * a1]
    for b in (g, g / np.sqrt(_ewma_prev(g * g, 0.99))):
        a = np.abs(b)
        out += [b * _lag(b, 2), b * _lag(b, 5), a * np.abs(_lag(b, 2)), a * np.abs(_lag(b, 5))]
    w = g / np.sqrt(_ewma_prev(g * g, 0.99))
    out += [(np.abs(w) > TAIL_Z).astype(float), (np.abs(w) < CENTER_G).astype(float)]
    return np.array(out)


STREAM_NAMES = (
    [f"{b}_{k}" for b in ("rg", "wrg") for k in calfeat.KINDS]
    + [f"{b}_{k}" for b in ("g", "w99") for k in ("ac2", "ac5", "vol2", "vol5")]
    + ["w99_tail", "w99_center"]
)
FEATURE_NAMES = [f"c2_{s}_{st}" for s in STREAM_NAMES for st in calfeat.STATS]


def series_block(x_hist, x_online) -> np.ndarray:
    x = np.concatenate([x_hist, x_online]).astype(np.float64)
    nh, T = len(x_hist), len(x_online)
    S = _streams(x, nh)
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
    a = EWMA_ALPHA
    eh_ew = lfilter([a], [1, -(1 - a)], eh, axis=1)[:, int(3 / a):]
    ew = lfilter([a], [1, -(1 - a)], eo, axis=1)
    feats.append(ew / np.maximum(eh_ew.std(1, keepdims=True), 1e-9))
    F = np.stack(feats, 2)
    return F.transpose(1, 0, 2).reshape(T, -1).astype(np.float32)


def _one(args):
    return series_block(*args)[::CACHE_STRIDE]


def compute(ds, processes=20):
    jobs = [ds.segments(k) for k in range(len(ds))]
    t0 = time.time()
    with Pool(processes) as pool:
        F = pool.map(_one, jobs, chunksize=20)
    print(f"cal2 block: {len(ds)} series in {time.time() - t0:.1f}s", flush=True)
    return np.concatenate(F)


if __name__ == "__main__":
    assert len(FEATURE_NAMES) == 100
    for name, fn in (("train", load_train), ("test_reduced", load_test_reduced)):
        np.save(os.path.join(CACHE, f"cal2_{name}.npy"), compute(fn(), processes=8))
