"""Streaming twin of calfeat.series_block (O(1) per step). Draft for main.py; verified
against the vectorised version by `python research/calstream.py`."""

import math

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtri

CAL_BURN = 200
CAL_LAMBDAS = (0.97, 0.99)
CAL_WINDOWS = (16, 64, 256)
CAL_ALPHA = 0.03
CAL_GRID_MAX = 1024
CAL_NS = 5 * (1 + len(CAL_LAMBDAS))


class CalState:
    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        nh = len(h)
        self.h_sorted = np.sort(h)
        self.inv = 1.0 / (nh + 1.0)
        g = self._g_vec(h)
        v0 = []
        bases = [g]
        for lam in CAL_LAMBDAS:
            v, _ = lfilter([1 - lam], [1, -lam], g * g, zi=[lam * 1.0])
            vprev = np.r_[1.0, v[:-1]]
            bases.append(g / np.sqrt(vprev))
            v0.append(float(v[-1]))
        self.v = v0
        self.prev = [float(b[-1]) for b in bases]
        S = []
        for b in bases:
            b1 = np.r_[0.0, b[:-1]]
            a, a1 = np.abs(b), np.abs(b1)
            S += [b, b * b, a, b * b1, a * a1]
        S = np.array(S)
        eh = S[:, CAL_BURN:] - S[:, CAL_BURN:].mean(1, keepdims=True)
        self.ctr = S[:, CAL_BURN:].mean(1)
        # horizon sd on the dyadic grid, then a per-horizon table 1..CAL_GRID_MAX
        c = np.concatenate([np.zeros((CAL_NS, 1)), np.cumsum(eh, 1)], 1)
        n = eh.shape[1]
        grid = 2 ** np.arange(0, 11)
        sds = []
        for k in grid:
            if k < n // 2:
                sds.append((c[:, k:] - c[:, :-k]).std(1))
            else:
                sds.append(sds[-1] * math.sqrt(k / grid[len(sds) - 1]))
        lsd = np.log(np.maximum(np.array(sds).T, 1e-9))
        ks = np.arange(1, CAL_GRID_MAX + 1)
        lk = np.log2(ks)
        lo = np.floor(lk).astype(int)
        hi = np.minimum(lo + 1, len(grid) - 1)
        f = lk - lo
        self.inv_sd = 1.0 / np.exp((1 - f) * lsd[:, lo] + f * lsd[:, hi])  # (NS, 1024)
        a = CAL_ALPHA
        ew_h = lfilter([a], [1, -(1 - a)], eh, axis=1)[:, int(3 / a):]
        self.inv_ew_sd = 1.0 / np.maximum(ew_h.std(1), 1e-9)
        self.n = 0
        self.cum = np.zeros((64, CAL_NS))
        self.ew = np.zeros(CAL_NS)

    def _g_vec(self, x):
        hs = self.h_sorted
        u = ((np.searchsorted(hs, x, "left") + np.searchsorted(hs, x, "right")) / 2 + 0.5) * self.inv
        return ndtri(u)

    def _inv_sd_at(self, k):
        if k <= CAL_GRID_MAX:
            return self.inv_sd[:, k - 1]
        return self.inv_sd[:, -1] * math.sqrt(CAL_GRID_MAX / k)

    def update(self, gt: float) -> np.ndarray:
        """gt: normal score of the new point against the history (as main.FeatureState._g)."""
        vals = []
        bs = [gt]
        for j, lam in enumerate(CAL_LAMBDAS):
            bs.append(gt / math.sqrt(self.v[j]))
            self.v[j] = lam * self.v[j] + (1 - lam) * gt * gt
        for b, b1 in zip(bs, self.prev):
            a, a1 = abs(b), abs(b1)
            vals += [b, b * b, a, b * b1, a * a1]
        self.prev = bs
        e = np.asarray(vals) - self.ctr
        self.n += 1
        n = self.n
        if n >= len(self.cum):
            self.cum = np.concatenate([self.cum, np.zeros_like(self.cum)])
        cs = self.cum[n - 1] + e
        self.cum[n] = cs
        out = [cs * self._inv_sd_at(n)]
        for w in CAL_WINDOWS:
            if n > w:
                out.append((cs - self.cum[n - w]) * self._inv_sd_at(w))
            else:
                out.append(cs * self._inv_sd_at(n))
        self.ew = (1 - CAL_ALPHA) * self.ew + CAL_ALPHA * e
        out.append(self.ew * self.inv_ew_sd)
        return np.stack(out, 1).ravel()  # stream-major, stats minor (as calfeat)


if __name__ == "__main__":
    import time
    from common import load_train
    import calfeat

    ds = load_train()
    for k in (0, 1, 7):
        h, o = ds.segments(k)
        ref = calfeat.series_block(h, o)
        st = CalState(h)
        g = st._g_vec(np.asarray(o, dtype=np.float64))
        t0 = time.time()
        got = np.array([st.update(float(v)) for v in g])
        print(k, got.shape, "max abs diff", float(np.abs(got - ref).max()), f"{(time.time() - t0) / len(o) * 1e6:.0f} us/step")
