"""Item 4: horizon-calibrated copies of v6's sum/window features for every v6 stream.

v6 scales stream e by one Bartlett long-run sd, so its sum at step n is S_n/sqrt(n). The
calibrated version is S_n / sd_h(n), with sd_h the sd of rolling h-sums of e over this
series' history -- i.e. the old feature times sqrt(h)/sd_h(h), h = n for the sum and
min(n, w) for window w. We only need that factor per series and horizon.

usage: python research/rcal.py   -> cache/rcal_{train,test_reduced}.npy
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from calfeat import _horizon_sd, _sd_at
from common import CACHE, CACHE_STRIDE, load_test_reduced, load_train, main

SKIP = 180  # replay rows start at history index MAX_LAG; skip to index 200 like calfeat
STATS = ["sum"] + [f"w{w}" for w in main.WINDOWS]
NAMES = [f"{s}_{st}" for s in main.STREAMS for st in STATS]


class Probe(main.FeatureState):
    def __init__(self, x_hist):
        self._rows = []
        super().__init__(x_hist)

    def _fit_learned(self, xh):
        self._nl = super()._fit_learned(xh)
        return self._nl

    def _raw(self, x):
        out = super()._raw(x)
        if self.replay:
            self._rows.append(out)
        return out


def factors(args):
    x_hist, T = args
    st = Probe(x_hist)
    H = np.array(st._rows)
    H[:, -main.N_LEARNED:] = st._nl[main.MAX_LAG - main.NL_P:]
    e = (H - np.array(st.center)) * np.array(st.scale)
    sdg = _horizon_sd(e[SKIP:].T)  # (streams, grid)
    n = np.arange(1, T + 1)[::CACHE_STRIDE]
    out = []
    for h in [n] + [np.minimum(n, w) for w in main.WINDOWS]:
        out.append(np.sqrt(h)[None, :] / _sd_at(sdg, h))  # (streams, rows)
    return np.stack(out, 2).transpose(1, 0, 2).reshape(len(n), -1).astype(np.float32)


if __name__ == "__main__":
    for split, fn, cache in (("test_reduced", load_test_reduced, "test_reduced"), ("train", load_train, "train")):
        ds = fn()
        t0 = time.time()
        with Pool(20) as pool:
            Fr = pool.map(factors, [(ds.segments(k)[0], int(ds.online_len[k])) for k in range(len(ds))], chunksize=20)
        R = np.concatenate(Fr)
        del Fr
        print(f"{split}: factors in {time.time() - t0:.0f}s", flush=True)
        d = np.load(os.path.join(CACHE, f"{cache}.npz"))
        names = list(d["names"])
        idx = [names.index(nm) for nm in NAMES]
        X = d["X"]
        assert X.shape[0] == R.shape[0]
        out = X[:, idx] * R
        del X, d
        np.save(os.path.join(CACHE, f"rcal_{split}.npy"), out.astype(np.float32))
        print(f"{split}: saved {out.shape}", flush=True)
