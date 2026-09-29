"""
ADIA Lab Structural Break Challenge, real-time edition.

Pipeline
--------
1. From the historical segment, fit a null model of each series: its mean/sd, its
   empirical distribution (for rank-based "normal scores"), and an AR(20) fit whose
   residuals capture any change in dependence (a changed process predicts worse).
2. Turn every observation into *streams*, one per kind of break (variance, shape,
   tails, dependence, predictability). Each stream is standardised by its historical
   mean and long-run sd, so under "no break" it is roughly mean 0 / variance 1.
3. Accumulate evidence on each stream with O(1) running statistics: the full-window
   z-test, sums over the last 16/64/256 steps (a cheap stand-in for the
   unknown-change-point GLR), two-sided Page CUSUMs, and an EWMA.
4. A LightGBM classifier, trained on labelled (series, step) pairs, maps that feature
   vector to P(break has already happened | data so far).

Historical and online streams go through the same `_raw` function, and training and
inference share `FeatureState.update`, so features cannot drift between the two.
"""

import math
import os
from bisect import bisect_left, bisect_right
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.special import ndtri

# @crunch/keep:on
INFER_PARALLELISM = 4

AR_ORDER = 20
MAX_LAG = AR_ORDER
CUSUM_K = (0.1, 0.5)
WINDOWS = (16, 64, 256)
EWMA_ALPHA = 0.03
G4_CAP = 50.0
TAIL_Z = 1.6448536269514722  # two-sided 10% tail of N(0,1)

STREAMS = (
    "z", "z2", "g", "g2", "absg", "g4", "tail", "gg1", "gg2", "gg3", "zz1",
    "absdz", "dz2", "vol1", "r", "absr", "r2", "rg2", "absrg",
)
STAT_NAMES = ["sum"] + [f"cusum{k}" for k in CUSUM_K] + [f"w{w}" for w in WINDOWS] + ["ewma"]
META_NAMES = ["log_n"]
FEATURE_NAMES = META_NAMES + [f"{s}_{st}" for s in STREAMS for st in STAT_NAMES]
N_FEATURES = len(FEATURE_NAMES)


def _long_run_sd(s: np.ndarray) -> float:
    """Bartlett-kernel long-run sd: the sd of the *mean* of s, times sqrt(n)."""
    n = len(s)
    d = s - s.mean()
    g0 = float(d @ d) / n
    if g0 <= 0:
        return 1e-6
    L = int(math.ceil(n ** (1.0 / 3.0)))
    lrv = g0
    for j in range(1, L + 1):
        lrv += 2.0 * (1.0 - j / (L + 1.0)) * float(d[j:] @ d[:-j]) / n
    # a near-zero lrv (anti-persistent series) would explode the scale
    return math.sqrt(max(lrv, 0.1 * g0))


def _normal_score_table(sorted_vals: np.ndarray) -> list:
    """ndtri of the mid-rank, indexed by bisect_left + bisect_right."""
    n = len(sorted_vals)
    half_ranks = np.arange(2 * n + 1) / 2.0
    return ndtri((half_ranks + 0.5) / (n + 1.0)).tolist()


class FeatureState:
    """Running state for one series. `update(x)` returns the feature vector (list)."""

    def __init__(self, x_hist):
        xh = np.asarray(x_hist, dtype=np.float64)
        n = len(xh)
        self.mu = float(xh.mean())
        self.sd = max(float(xh.std()), 1e-8)
        z = (xh - self.mu) / self.sd

        self.x_sorted = np.sort(xh).tolist()
        self.gtab = _normal_score_table(np.asarray(self.x_sorted))

        # AR(p) on standardised history (least squares, no intercept: z has mean 0)
        p = AR_ORDER
        X = np.column_stack([z[p - i - 1 : n - i - 1] for i in range(p)])
        phi, *_ = np.linalg.lstsq(X, z[p:], rcond=None)
        self.phi = phi.tolist()
        r = z[p:] - X @ phi
        self.r_sd = max(float(r.std()), 1e-8)
        self.r_sorted = np.sort(r / self.r_sd).tolist()
        self.rtab = _normal_score_table(np.asarray(self.r_sorted))

        # replay the history through the same stream function to get null statistics
        self.z_lags = z[:MAX_LAG][::-1].tolist()  # most recent first
        self.g_lags = [self._g(v) for v in xh[:MAX_LAG][::-1]]
        H = np.array([self._raw(float(v)) for v in xh[MAX_LAG:]])
        self.center = H.mean(0).tolist()
        self.scale = [1.0 / _long_run_sd(H[:, i]) for i in range(H.shape[1])]
        # the online segment continues the history: lags now hold the last MAX_LAG points

        ns = len(STREAMS)
        self.n = 0
        self.sums = [0.0] * ns
        self.cum = [[0.0] * ns]  # cumulative sums at every step, for windowed sums
        self.up = [[0.0] * ns for _ in CUSUM_K]
        self.dn = [[0.0] * ns for _ in CUSUM_K]
        self.ew = [0.0] * ns
        self.ew_norm = math.sqrt((2.0 - EWMA_ALPHA) / EWMA_ALPHA)

    def _g(self, x: float) -> float:
        xs = self.x_sorted
        return self.gtab[bisect_left(xs, x) + bisect_right(xs, x)]

    def _raw(self, x: float) -> tuple:
        """Stream values for observation x; advances the lag buffers."""
        zt = (x - self.mu) / self.sd
        gt = self._g(x)
        zl, gl = self.z_lags, self.g_lags
        pred = 0.0
        for c, v in zip(self.phi, zl):
            pred += c * v
        rt = (zt - pred) / self.r_sd
        rs = self.r_sorted
        rg = self.rtab[bisect_left(rs, rt) + bisect_right(rs, rt)]
        d = zt - zl[0]
        g2 = gt * gt
        ag = abs(gt)
        out = (
            zt,
            zt * zt,
            gt,
            g2,
            ag,
            min(g2 * g2, G4_CAP),
            1.0 if ag > TAIL_Z else 0.0,
            gt * gl[0],
            gt * gl[1],
            gt * gl[2],
            zt * zl[0],
            abs(d),
            d * d,
            ag * abs(gl[0]),
            rt,
            abs(rt),
            rt * rt,
            rg * rg,
            abs(rg),
        )
        zl.insert(0, zt)
        zl.pop()
        gl.insert(0, gt)
        gl.pop()
        return out

    def update(self, x: float) -> List[float]:
        raw = self._raw(x)
        center, scale, sums = self.center, self.scale, self.sums
        ns = len(raw)
        e = [(raw[i] - center[i]) * scale[i] for i in range(ns)]

        self.n += 1
        n = self.n
        for i in range(ns):
            sums[i] += e[i]
        cum = self.cum
        cum.append(sums[:])
        inv_sqrt_n = 1.0 / math.sqrt(n)

        win = []
        for w in WINDOWS:
            if n > w:
                base = cum[n - w]
                c = 1.0 / math.sqrt(w)
                win.append([abs(sums[i] - base[i]) * c for i in range(ns)])
            else:
                win.append([abs(s) * inv_sqrt_n for s in sums])

        cus = []
        for k, up, dn in zip(CUSUM_K, self.up, self.dn):
            row = []
            for i in range(ns):
                u = up[i] + e[i] - k
                up[i] = u if u > 0.0 else 0.0
                v = dn[i] - e[i] - k
                dn[i] = v if v > 0.0 else 0.0
                row.append(up[i] if up[i] > dn[i] else dn[i])
            cus.append(row)

        a, b, ew = EWMA_ALPHA, 1.0 - EWMA_ALPHA, self.ew
        for i in range(ns):
            ew[i] = b * ew[i] + a * e[i]

        feats = [math.log(n)]
        for i in range(ns):
            feats.append(abs(sums[i]) * inv_sqrt_n)
            for row in cus:
                feats.append(row[i])
            for row in win:
                feats.append(row[i])
            feats.append(abs(ew[i]) * self.ew_norm)
        return feats


def series_features(x_hist, x_online) -> np.ndarray:
    st = FeatureState(x_hist)
    return np.array([st.update(float(x)) for x in x_online], dtype=np.float32)


# --------------------------------------------------------------------------- model

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=2000,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=0,
    deterministic=True,
    force_col_wise=True,
)
NUM_ROUNDS = 600
TRAIN_ROW_STRIDE = 2  # keep every other online step; neighbouring steps are near-duplicates


def build_training_matrix(datasets, stride: int = TRAIN_ROW_STRIDE):
    Xs, ys = [], []
    for _, x_hist, x_online, tau in datasets:
        F = series_features(x_hist, x_online)
        y = np.zeros(len(F), dtype=np.float32)
        if tau is not None:
            y[int(tau):] = 1.0
        Xs.append(F[::stride])
        ys.append(y[::stride])
    return np.concatenate(Xs), np.concatenate(ys)


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
):
    import lightgbm as lgb

    X, y = build_training_matrix(datasets)
    booster = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X, y, feature_name=FEATURE_NAMES, free_raw_data=True),
        num_boost_round=NUM_ROUNDS,
    )
    booster.save_model(os.path.join(model_directory_path, "model.txt"))


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    import lightgbm as lgb
    from threadpoolctl import threadpool_limits

    booster = lgb.Booster(model_file=os.path.join(model_directory_path, "model.txt"))

    with threadpool_limits(limits=1):
        yield  # ready

        buf = np.empty((1, N_FEATURES), dtype=np.float64)
        for x_hist, x_online in datasets:
            st = FeatureState(x_hist)
            for point in x_online:
                buf[0, :] = st.update(float(point))
                yield float(booster.predict(buf, num_threads=1)[0])
