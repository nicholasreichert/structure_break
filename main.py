"""
ADIA Lab Structural Break Challenge, real-time edition.

Pipeline
--------
1. From the historical segment, fit a null model of each series: its mean/sd, its
   empirical distribution (for rank-based "normal scores"), and an AR(20) fit whose
   residuals capture any change in dependence (a changed process predicts worse). A second,
   nonlinear predictor (ridge on random Fourier features of the last 10 normal scores, with
   its own conditional-variance model) adds error streams for dependence AR cannot express.
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
INFER_PARALLELISM = 12

AR_ORDER = 20
MAX_LAG = AR_ORDER
CUSUM_K = (0.1, 0.3)
WINDOWS = (16, 64, 256)
EWMA_ALPHA = 0.03
G4_CAP = 50.0
TAIL_Z = 1.6448536269514722  # two-sided 10% tail of N(0,1)

STREAMS = (
    "z", "z2", "g", "g2", "absg", "g4", "tail", "gg1", "gg2", "gg3", "zz1",
    "absdz", "dz2", "vol1", "r", "absr", "r2", "rg2", "absrg",
    "zf2", "gg5", "gg10", "sgn1", "g3", "vol5", "upper", "center",
    "nl_e2", "nl_std2", "nl_logv", "nl_gain",
)
N_LEARNED = 4  # the nl_* streams come last; their history values are leave-one-out
VOL_LAMBDA = 0.94  # EWMA volatility filter for zf2
CENTER_G = 0.2533471031357997  # |g| below this <=> middle 20% of the historical distribution
# every statistic keeps its sign: breaks and no-break drift push streams in different directions
STAT_NAMES = (
    ["sum"]
    + [f"cusum{d}{k}" for k in CUSUM_K for d in ("up", "dn")]
    + [f"w{w}" for w in WINDOWS]
    + ["ewma", "glr", "glr_age"]
)
META_NAMES = ["log_n"]

# Calibrated block (v7). The histories have volatility clustering, so an iid null is wrong
# by a different amount for every series: inside a history, a 100-point variance z-stat has
# sd ~1.9, not 1. These streams are (a) whitened by an EWMA volatility carried from the start
# of the history and (b) accumulated with scales measured on the series' own history *at the
# same horizon* (sd of rolling k-sums), instead of one long-run sd for every horizon.
CAL_BURN = 200  # history points skipped before measuring the null (EWMA start-up)
CAL_LAMBDAS = (0.97, 0.99)
CAL_WINDOWS = (16, 64, 256)
CAL_ALPHA = 0.03
CAL_GRID = 1 << np.arange(11)  # horizons 1, 2, 4, ..., 1024
CAL_KINDS = ("lvl", "sq", "abs", "ac1", "vol1")
# (AR-residual bases, lag-2/5 products and tail/centre rates were tried too: no CV gain)
# (a per-series GARCH(1,1) whitening base was tried too: +0.0003 over 3 seeds, i.e. noise)
CAL_STREAMS = [f"{b}_{k}" for b in ("g", "w97", "w99") for k in CAL_KINDS]
CAL_STATS = ["sum"] + [f"w{w}" for w in CAL_WINDOWS] + ["ewma"]
CAL_NAMES = [f"c_{s}_{st}" for s in CAL_STREAMS for st in CAL_STATS]

FEATURE_NAMES = META_NAMES + [f"{s}_{st}" for s in STREAMS for st in STAT_NAMES] + CAL_NAMES
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


# ------------------------------------------------------------- learned next-value predictor
# Ridge regression of g_t on [its last NL_P lags, random Fourier features of those lags]: an
# approximate Gaussian-kernel regression, so it can learn nonlinear dependence AR(p) cannot.
# The penalty is picked per series by leave-one-out error (closed form from one SVD), and the
# history's error stream uses the LOO residuals, so it is out-of-sample like the online errors.
NL_P = 10
NL_D = 100
NL_BANDWIDTH = 3.0
NL_LAMBDAS = (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)  # multiples of n
_rff = np.random.default_rng(0)
NL_W = _rff.normal(size=(NL_P, NL_D)) / NL_BANDWIDTH
NL_B = _rff.uniform(0.0, 2.0 * math.pi, size=NL_D)
NL_C = math.sqrt(2.0 / NL_D)
del _rff


def _nl_features(L: np.ndarray) -> np.ndarray:
    """Rows of lag vectors (most recent first) -> [lags, random Fourier features]."""
    return np.hstack([L, NL_C * np.cos(L @ NL_W + NL_B)])


def _ridge_loo(F: np.ndarray, y: np.ndarray):
    """Ridge with an unpenalised intercept. Returns (beta, intercept, LOO residuals)."""
    n = len(y)
    fm, ym = F.mean(0), float(y.mean())
    U, s, Vt = np.linalg.svd(F - fm, full_matrices=False)
    uy = U.T @ (y - ym)
    s2 = s * s
    U2 = U * U
    best = None
    for lam in NL_LAMBDAS:
        sh = s2 / (s2 + lam * n)
        loo = (y - U @ (sh * uy) - ym) / (1.0 - U2 @ sh - 1.0 / n)
        mse = float(loo @ loo)
        if best is None or mse < best[0]:
            best = (mse, lam, loo)
    _, lam, loo = best
    beta = Vt.T @ (s / (s2 + lam * n) * uy)
    return beta, ym - float(fm @ beta), loo


class CalBlock:
    """Whitened streams with horizon-calibrated running sums (see CAL_* above).

    `step(g)` turns one normal score into stream values;
    the history is replayed through it, `fit_null` measures the per-horizon scales on that
    replay, and `update` accumulates the online stream values into features.
    """

    def __init__(self):
        self.v = [1.0] * len(CAL_LAMBDAS)  # EWMA of g^2 per lambda (variance before this step)
        self.prev = [0.0] * (1 + len(CAL_LAMBDAS))  # last value of each base

    def step(self, g: float) -> list:
        bases = [g]
        g2 = g * g
        for j, lam in enumerate(CAL_LAMBDAS):
            bases.append(g / math.sqrt(self.v[j]))
            self.v[j] = lam * self.v[j] + (1.0 - lam) * g2
        out = []
        for b, b1 in zip(bases, self.prev):
            a = abs(b)
            out += [b, b * b, a, b * b1, a * abs(b1)]
        self.prev = bases
        return out

    def fit_null(self, C: np.ndarray):
        """C: replayed stream values over the history (rows = time), burn-in included."""
        C = C[CAL_BURN - MAX_LAG :]  # rows start at history index MAX_LAG
        self.ctr = C.mean(0)
        eh = (C - self.ctr).T
        ns, n = eh.shape
        c = np.concatenate([np.zeros((ns, 1)), np.cumsum(eh, 1)], 1)
        sds = []
        for k in CAL_GRID:
            if k < n // 2:
                sds.append((c[:, k:] - c[:, :-k]).std(1))
            else:  # history too short for this horizon: extend at constant variance ratio
                sds.append(sds[-1] * math.sqrt(k / CAL_GRID[len(sds) - 1]))
        lsd = np.log(np.maximum(np.array(sds).T, 1e-9))
        # per-horizon table 1..1024 by log-log interpolation between grid points
        lk = np.log2(np.arange(1, CAL_GRID[-1] + 1))
        lo = np.floor(lk).astype(int)
        hi = np.minimum(lo + 1, len(CAL_GRID) - 1)
        f = lk - lo
        self.inv_sd = np.ascontiguousarray((1.0 / np.exp((1 - f) * lsd[:, lo] + f * lsd[:, hi])).T)
        a = CAL_ALPHA
        ew = np.zeros(ns)
        ews = []
        for row in eh.T:
            ew = (1 - a) * ew + a * row
            ews.append(ew)
        self.inv_ew_sd = 1.0 / np.maximum(np.array(ews)[int(3 / a):].std(0), 1e-9)
        self.n = 0
        self.cum = np.zeros((64, ns))
        self.ew = np.zeros(ns)

    def _inv_sd(self, k: int) -> np.ndarray:
        if k <= len(self.inv_sd):
            return self.inv_sd[k - 1]
        return self.inv_sd[-1] * math.sqrt(len(self.inv_sd) / k)

    def update(self, vals: list) -> list:
        e = np.asarray(vals) - self.ctr
        self.n += 1
        n = self.n
        if n >= len(self.cum):
            self.cum = np.concatenate([self.cum, np.zeros_like(self.cum)])
        cs = self.cum[n - 1] + e
        self.cum[n] = cs
        out = [cs * self._inv_sd(n)]
        for w in CAL_WINDOWS:
            if n > w:
                out.append((cs - self.cum[n - w]) * self._inv_sd(w))
            else:
                out.append(cs * self._inv_sd(n))
        self.ew = (1 - CAL_ALPHA) * self.ew + CAL_ALPHA * e
        out.append(self.ew * self.inv_ew_sd)
        return np.stack(out, 1).ravel().tolist()  # stream-major, matching CAL_NAMES


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

        nl_hist = self._fit_learned(xh)

        self.vol = 1.0  # EWMA of z^2, carried from the history into the online segment
        # replay the history through the same stream function to get null statistics
        self.z_lags = z[:MAX_LAG][::-1].tolist()  # most recent first
        self.g_lags = [self._g(v) for v in xh[:MAX_LAG][::-1]]
        self.cal = CalBlock()
        self.replay = True  # skip the online predictor; nl_hist holds the LOO values
        rows, cal_rows = [], []
        for v in xh[MAX_LAG:]:
            rows.append(self._raw(float(v)))
            cal_rows.append(self.cal_vals)
        self.replay = False
        H = np.array(rows)
        self.cal.fit_null(np.array(cal_rows))
        H[:, -N_LEARNED:] = nl_hist[MAX_LAG - NL_P :]
        self.center = H.mean(0).tolist()
        self.scale = [1.0 / _long_run_sd(H[:, i]) for i in range(H.shape[1])]
        # the online segment continues the history: lags now hold the last MAX_LAG points

        ns = len(STREAMS)
        self.n = 0
        self.sums = [0.0] * ns
        # cumulative sums at every step (row n = after n points), for windows and the GLR
        self.cum = np.zeros((64, ns))
        self.up = [[0.0] * ns for _ in CUSUM_K]
        self.dn = [[0.0] * ns for _ in CUSUM_K]
        self.ew = [0.0] * ns
        self.ew_norm = math.sqrt((2.0 - EWMA_ALPHA) / EWMA_ALPHA)

    _NL_ZERO = (0.0,) * N_LEARNED

    def _fit_learned(self, xh: np.ndarray) -> np.ndarray:
        """Fit the nonlinear predictor on the history; return its LOO streams, rows NL_P.."""
        xs = np.asarray(self.x_sorted)
        idx = np.searchsorted(xs, xh, "left") + np.searchsorted(xs, xh, "right")
        g = np.asarray(self.gtab)[idx]
        n = len(g)
        L = np.column_stack([g[NL_P - i - 1 : n - i - 1] for i in range(NL_P)])
        F = _nl_features(L)
        y = g[NL_P:]
        beta, self.nl_c0, e = _ridge_loo(F, y)
        e2 = e * e
        self.vfloor = 0.1 * float(e2.mean())
        bv, self.nl_v0, loov = _ridge_loo(F, e2)
        v = np.maximum(e2 - loov, self.vfloor)  # LOO prediction of e2
        ba, self.nl_a0, ea = _ridge_loo(L, y)  # linear AR(NL_P) on the same footing
        self.nl_beta, self.nl_bv, self.nl_ba = beta, bv, ba
        return np.column_stack([e2, e2 / v, np.log(v), ea * ea - e2])

    def _learned(self, gt: float, gl: list) -> tuple:
        u = np.array(gl[:NL_P])
        f = np.concatenate([u, NL_C * np.cos(u @ NL_W + NL_B)])
        e = gt - float(f @ self.nl_beta) - self.nl_c0
        e2 = e * e
        v = max(float(f @ self.nl_bv) + self.nl_v0, self.vfloor)
        ea = gt - float(u @ self.nl_ba) - self.nl_a0
        return (e2, e2 / v, math.log(v), ea * ea - e2)

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
        self.cal_vals = self.cal.step(gt)
        d = zt - zl[0]
        g2 = gt * gt
        ag = abs(gt)
        z2 = zt * zt
        zf2 = z2 / self.vol
        self.vol = VOL_LAMBDA * self.vol + (1.0 - VOL_LAMBDA) * z2
        g1 = gl[0]
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
            zf2,
            gt * gl[4],
            gt * gl[9],
            (1.0 if gt > 0 else -1.0 if gt < 0 else 0.0) * (1.0 if g1 > 0 else -1.0 if g1 < 0 else 0.0),
            g2 * gt,
            ag * abs(gl[4]),
            1.0 if gt > 0.0 else 0.0,
            1.0 if ag < CENTER_G else 0.0,
        ) + (self._NL_ZERO if self.replay else self._learned(gt, gl))
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
        if n >= len(cum):
            cum = self.cum = np.concatenate([cum, np.zeros_like(cum)])
        cum[n] = sums
        inv_sqrt_n = 1.0 / math.sqrt(n)

        win = []
        for w in WINDOWS:
            if n > w:
                c = 1.0 / math.sqrt(w)
                win.append(((cum[n] - cum[n - w]) * c).tolist())
            else:
                win.append([s * inv_sqrt_n for s in sums])

        # GLR for a mean shift at an unknown point j: max_j (S_n - S_j)^2 / (n - j)
        seg = cum[n] - cum[:n]
        stat = seg * seg * (1.0 / np.arange(n, 0, -1, dtype=np.float64))[:, None]
        jstar = stat.argmax(0)
        cols = np.arange(ns)
        glr = (seg[jstar, cols] / np.sqrt(n - jstar)).tolist()  # signed shift at the best split
        glr_age = np.log(n - jstar).tolist()

        cus = []
        for k, up, dn in zip(CUSUM_K, self.up, self.dn):
            for i in range(ns):
                u = up[i] + e[i] - k
                up[i] = u if u > 0.0 else 0.0
                v = dn[i] - e[i] - k
                dn[i] = v if v > 0.0 else 0.0
            cus.append(up[:])
            cus.append(dn[:])

        a, b, ew = EWMA_ALPHA, 1.0 - EWMA_ALPHA, self.ew
        for i in range(ns):
            ew[i] = b * ew[i] + a * e[i]

        feats = [math.log(n)]
        for i in range(ns):
            feats.append(sums[i] * inv_sqrt_n)
            for row in cus:
                feats.append(row[i])
            for row in win:
                feats.append(row[i])
            feats.append(ew[i] * self.ew_norm)
            feats.append(glr[i])
            feats.append(glr_age[i])
        feats += self.cal.update(self.cal_vals)
        return feats


def series_features(x_hist, x_online) -> np.ndarray:
    st = FeatureState(x_hist)
    return np.array([st.update(float(x)) for x in x_online], dtype=np.float32)


# --------------------------------------------------------------------------- model

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.025,
    num_leaves=15,
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
NUM_ROUNDS = 1200
N_SEEDS = 3  # models averaged (raw scores) at inference; each uses its own bagging/feature seed
TRAIN_ROW_STRIDE = 2  # keep every other online step; neighbouring steps are near-duplicates

# Shifted-start copies (v9). The model is data-limited (2/5 -> 4/5 of the series: +0.011
# TS-AUC), and breaks are the scarce part. Each copy moves the first k online points, all
# pre-break (k < tau), into the history: tau' = tau - k, same post-break data, seen from a
# different start (history calibration, accumulated sums). Copy c draws k with seed c.
# 2-fold CV (lr 0.05, seeds 1-2): none 0.6195, 1 copy 0.6236, 2 copies 0.6271,
# 4 copies at stride 4 (same rows as 2 at stride 2) 0.6269.
SHIFT_COPIES = 2
SHIFT_STRIDE = 2
SHIFT_MIN_ONLINE = 10


def shift_amounts(taus: np.ndarray, online_lens: np.ndarray, seed: int) -> np.ndarray:
    """k per series; taus = -1 for no break (then k < T/2)."""
    u = np.random.default_rng(seed).uniform(0.0, 1.0, len(online_lens))
    lim = np.where(taus >= 0, taus, online_lens // 2)
    lim = np.minimum(lim, online_lens - SHIFT_MIN_ONLINE)
    return np.where(lim >= 1, np.floor(u * lim).astype(int), 0)


def build_training_matrix(datasets, stride: int = TRAIN_ROW_STRIDE):
    datasets = list(datasets)
    taus = np.array([-1 if tau is None else int(tau) for *_, tau in datasets])
    lens = np.array([len(x_online) for _, _, x_online, _ in datasets])
    ks = [np.zeros(len(datasets), dtype=int)] + [
        shift_amounts(taus, lens, c) for c in range(SHIFT_COPIES)
    ]
    Xs, ys = [], []
    for c, k_all in enumerate(ks):
        st = stride if c == 0 else SHIFT_STRIDE
        for (_, x_hist, x_online, _), tau, k in zip(datasets, taus, k_all):
            xh = np.concatenate([np.asarray(x_hist, dtype=np.float64), np.asarray(x_online[:k], dtype=np.float64)])
            F = series_features(xh, x_online[k:])
            y = np.zeros(len(F), dtype=np.float32)
            if tau >= 0:
                y[tau - k :] = 1.0
            Xs.append(F[::st])
            ys.append(y[::st])
    return np.concatenate(Xs), np.concatenate(ys)


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
):
    import lightgbm as lgb

    X, y = build_training_matrix(datasets)
    ds = lgb.Dataset(X, y, feature_name=FEATURE_NAMES, free_raw_data=False)
    for s in range(N_SEEDS):
        booster = lgb.train(seed_params(s), ds, num_boost_round=NUM_ROUNDS)
        booster.save_model(os.path.join(model_directory_path, f"model_{s}.txt"))


def seed_params(s: int) -> dict:
    return dict(LGB_PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s)


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    import lightgbm as lgb
    from threadpoolctl import threadpool_limits

    # The upload can rewrite model.txt with CRLF line endings. LightGBM seeks to each tree by
    # the byte offsets in its `tree_sizes=` header, so one extra byte per line breaks loading.
    boosters = []
    for s in range(N_SEEDS):
        with open(os.path.join(model_directory_path, f"model_{s}.txt"), "rb") as f:
            boosters.append(lgb.Booster(model_str=f.read().decode("utf-8").replace("\r\n", "\n")))

    with threadpool_limits(limits=1):
        yield  # ready

        buf = np.empty((1, N_FEATURES), dtype=np.float64)
        for x_hist, x_online in datasets:
            st = FeatureState(x_hist)
            for point in x_online:
                buf[0, :] = st.update(float(point))
                # mean raw score (log-odds) over the seeds; only the per-step ranking matters
                raw = 0.0
                for b in boosters:
                    raw += b.predict(buf, raw_score=True, num_threads=1)[0]
                yield float(raw / N_SEEDS)
