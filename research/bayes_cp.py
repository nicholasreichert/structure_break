"""Exact Bayesian change-point evidence with nonparametric post-change models.

For every candidate break start k <= t we keep log BF_k(t): the Bayes factor of
"x_k..x_t come from an unknown new distribution" against "they come from the history's
distribution". The post-change model is a Dirichlet over bins centred on the null, so
it adapts to whatever the new distribution is (mean, scale, shape, dependence) without
choosing a parametric family. The prior over k given survival to t is flat, so ranking
series by log sum_k BF_k is ranking by the Shiryaev posterior P(tau <= t | data).

Models
  marg : B equiprobable bins of the history PIT; null = uniform.
  pair : transition bins (b_{t-1}, b_t) on a coarser grid; null = history transitions.

Usage: python research/bayes_cp.py   (standalone TS-AUC of each output, train set)
"""

from __future__ import annotations

import math
import time
from multiprocessing import Pool

import numpy as np

MARG_B = 16
MARG_C = (4.0, 32.0, 256.0)
PAIR_B = 4
PAIR_C = (8.0, 64.0)
RECENT = (16, 64)


def _pit_bins(h_sorted: np.ndarray, x: np.ndarray, B: int) -> np.ndarray:
    n = len(h_sorted)
    r = (np.searchsorted(h_sorted, x, "left") + np.searchsorted(h_sorted, x, "right")) / 2.0
    u = (r + 0.5) / (n + 1.0)
    return np.minimum((u * B).astype(np.int64), B - 1)


class BayesCP:
    """Streaming state; `update(x)` returns a list of features."""

    def __init__(self, x_hist, cap: int = 1024):
        h = np.sort(np.asarray(x_hist, dtype=np.float64))
        self.h_sorted = h
        hb = _pit_bins(h, np.asarray(x_hist, dtype=np.float64), PAIR_B)
        P = np.full((PAIR_B, PAIR_B), 0.5)
        np.add.at(P, (hb[:-1], hb[1:]), 1.0)
        self.P0 = P / P.sum(1, keepdims=True)
        self.log_P0 = np.log(self.P0)
        self.prev_pb = int(hb[-1])
        self.t = 0
        self._alloc(cap)

    def _alloc(self, cap):
        self.cap = cap
        nm, npair = len(MARG_C), len(PAIR_C)
        self.m_cnt = np.zeros((cap, MARG_B))          # counts of each bin since k
        self.m_lbf = np.zeros((nm, cap))              # log BF per concentration
        self.p_cnt = np.zeros((cap, PAIR_B, PAIR_B))  # transition counts since k
        self.p_row = np.zeros((cap, PAIR_B))
        self.p_lbf = np.zeros((npair, cap))

    def _grow(self):
        old = (self.m_cnt, self.m_lbf, self.p_cnt, self.p_row, self.p_lbf)
        c = self.cap
        self._alloc(2 * c)
        self.m_cnt[:c], self.m_lbf[:, :c], self.p_cnt[:c], self.p_row[:c], self.p_lbf[:, :c] = old

    def update(self, x: float):
        if self.t >= self.cap:
            self._grow()
        t = self.t
        n = t + 1  # candidate starts k = 0..t
        mb = int(_pit_bins(self.h_sorted, np.array([x]), MARG_B)[0])
        pb = int(_pit_bins(self.h_sorted, np.array([x]), PAIR_B)[0])

        # marginal Dirichlet-multinomial, null uniform 1/B
        cnt = self.m_cnt[:n, mb]
        tot = np.arange(t, -1, -1, dtype=np.float64)  # points already in [k, t-1]
        logB = math.log(MARG_B)
        for j, c in enumerate(MARG_C):
            self.m_lbf[j, :n] += np.log((c / MARG_B + cnt) / (c + tot)) + logB
        self.m_cnt[:n, mb] += 1.0

        # transition model, prior centred on history transitions
        a = self.prev_pb
        pc = self.p_cnt[:n, a, pb]
        pr = self.p_row[:n, a]
        p0 = self.P0[a, pb]
        for j, c in enumerate(PAIR_C):
            self.p_lbf[j, :n] += np.log((c * p0 + pc) / (c + pr)) - math.log(p0)
        self.p_cnt[:n, a, pb] += 1.0
        self.p_row[:n, a] += 1.0
        self.prev_pb = pb
        self.t = n

        feats = []
        for L in (self.m_lbf[:, :n], self.p_lbf[:, :n]):
            for row in L:
                mx = row.max()
                lse = mx + math.log(np.exp(row - mx).sum())
                feats.append(lse - math.log(n))       # log mean BF = log SR / (t+1)
                feats.append(mx)                       # GLR-style max over starts
                for R in RECENT:                       # evidence the break is recent
                    r = row[max(0, n - R):]
                    rm = r.max()
                    feats.append(rm + math.log(np.exp(r - rm).sum()) - lse)
        return feats


FEATURE_NAMES = [
    f"{m}{c:g}_{s}"
    for m, cs in (("bm", MARG_C), ("bp", PAIR_C))
    for c in cs
    for s in ["lsr", "max"] + [f"rec{R}" for R in RECENT]
]


def series_features(x_hist, x_online) -> np.ndarray:
    st = BayesCP(x_hist)
    return np.array([st.update(float(x)) for x in x_online], dtype=np.float32)


def _one(args):
    return series_features(*args)


def compute(ds, processes=20):
    jobs = [ds.segments(k) for k in range(len(ds))]
    t0 = time.time()
    with Pool(processes) as pool:
        F = pool.map(_one, jobs, chunksize=20)
    print(f"bayes features: {len(ds)} series in {time.time() - t0:.1f}s")
    return np.concatenate(F)


if __name__ == "__main__":
    import os
    from common import CACHE, load_train, ts_auc, load_test_reduced

    for name, fn in (("train", load_train), ("test_reduced", load_test_reduced)):
        path = os.path.join(CACHE, f"bayes_{name}.npy")
        if not os.path.exists(path):
            np.save(path, compute(fn()))
    ds = load_train()
    F = np.load(os.path.join(CACHE, "bayes_train.npy"))
    y = np.concatenate([ds.labels(k) for k in range(len(ds))])
    step = np.concatenate([np.arange(n) for n in ds.online_len])
    for j, nm in enumerate(FEATURE_NAMES):
        print(f"{ts_auc(y, F[:, j], step):.4f}  {nm}")
