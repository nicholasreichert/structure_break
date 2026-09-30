"""Two gap diagnostics on the v8 feature set (fast settings: lr 0.05, 600 rounds, 1 seed).

1. Learning curve: train on 1/4 of the series instead of 1/2 (same validation fold).
   A steep curve means more (or augmented) training series would pay.
2. Hidden-T probe (DIAGNOSTIC ONLY, never shipped): add log(T) and (t+1)/T.
   If this reaches ~0.687, the leaderboard gap is information about T, not a better detector.

usage: python research/diag_gap.py [--t-only]   (run from the repo root)
"""

import sys
import time

import lightgbm as lgb
import numpy as np

from common import folds, load_train, main, ts_auc
from cv2 import load


def fit_eval(X, y, trn, val, step, names):
    params = dict(main.LGB_PARAMS, num_threads=20, learning_rate=0.05)
    b = lgb.train(params, lgb.Dataset(X[trn], y[trn], feature_name=names), num_boost_round=600)
    p = b.predict(X[val])
    return ts_auc(y[val], p, step[val]), p


def run():
    X, y, series, step, names = load([])
    fold_of = folds(series.max() + 1)[series]
    val = np.flatnonzero(fold_of == 0)
    trn = np.flatnonzero(fold_of != 0)
    rng = np.random.default_rng(1)
    half = rng.permutation(np.unique(series[trn]))[: len(np.unique(series[trn])) // 2]
    trn_q = trn[np.isin(series[trn], half)]

    if "--t-only" not in sys.argv:
        t0 = time.time()
        a_full, _ = fit_eval(X, y, trn, val, step, names)
        print(f"train 1/2 of series: {a_full:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        a_q, _ = fit_eval(X, y, trn_q, val, step, names)
        print(f"train 1/4 of series: {a_q:.4f}", flush=True)

    T = load_train().online_len.astype(np.float32)[series]
    XT = np.hstack([X, np.log(T)[:, None], ((step + 1) / T)[:, None]])
    del X
    a_T, _ = fit_eval(XT, y, trn, val, step, names + ["logT", "frac"])
    print(f"+ hidden T (diagnostic): {a_T:.4f}", flush=True)
    a_T0, _ = fit_eval(XT[:, -2:], y, trn, val, step, ["logT", "frac"])
    print(f"T features alone: {a_T0:.4f}", flush=True)


if __name__ == "__main__":
    run()
