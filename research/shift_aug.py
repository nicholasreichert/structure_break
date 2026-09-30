"""Augmentation: one extra copy of every training series with the online start moved later.

The first k online points (all pre-break, k < tau) are appended to the history, so the copy
has tau' = tau - k and T' = T - k. The post-break data is the same, but the detector sees it
from a different start: different history calibration, different accumulated sums.

  python research/shift_aug.py build [c]   # copy c (shift seed c) -> cache/shift[c]_train.npz
  SHIFT_COPIES=2 COPY_STRIDE=4 python research/shift_aug.py cv ...   # copies 0..1, every 4th step
  python research/shift_aug.py cv [aug|base] [seeds]   # e.g. cv base 1,2 ; validates on real rows
"""

import os
import sys
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np

from common import CACHE, CACHE_STRIDE, folds, load_train, main, ts_auc
from cv2 import load

N_COPIES = int(os.environ.get("SHIFT_COPIES", "1"))  # copy c uses shift seed c
COPY_STRIDE = int(os.environ.get("COPY_STRIDE", "2"))  # keep copy rows with step % this == 0


def path(c: int) -> str:
    return os.path.join(CACHE, "shift_train.npz" if c == 0 else f"shift{c}_train.npz")
MIN_ONLINE = 10


def shift_of(ds, seed=0):
    rng = np.random.default_rng(seed)
    T = ds.online_len
    lim = np.where(ds.has_break, ds.tau, T // 2)  # never move a post-break point into history
    lim = np.minimum(lim, T - MIN_ONLINE)
    return np.where(lim >= 1, np.floor(rng.uniform(0, 1, len(T)) * lim).astype(int), 0)


def _one(args):
    xh, xo, k = args
    return main.series_features(np.r_[xh, xo[:k]], xo[k:])[::CACHE_STRIDE]


def build(c: int = 0):
    ds = load_train()
    k = shift_of(ds, seed=c)
    jobs = [(*ds.segments(i), int(k[i])) for i in range(len(ds))]
    t0 = time.time()
    with Pool(20) as pool:
        feats = pool.map(_one, jobs, chunksize=20)
    print(f"shift features: {len(ds)} series in {time.time() - t0:.0f}s")
    X = np.concatenate(feats)
    series = np.concatenate([np.full(len(f), i, dtype=np.int32) for i, f in enumerate(feats)])
    step = np.concatenate([np.arange(len(f), dtype=np.int32) * CACHE_STRIDE for f in feats])
    tau = np.where(ds.has_break, ds.tau - k, -1)[series]
    y = ((tau >= 0) & (step >= tau)).astype(np.int8)
    np.savez(path(c), X=X, series=series, step=step, y=y)


def cv():
    X, y, series, step, names = load([])
    AX, ay, aseries = [], [], []
    for c in range(N_COPIES):
        A = np.load(path(c))
        keep = A["step"] % COPY_STRIDE == 0
        AX.append(A["X"][keep])
        ay.append(A["y"][keep])
        aseries.append(A["series"][keep])
        del A
    AX, ay, aseries = np.concatenate(AX), np.concatenate(ay), np.concatenate(aseries)
    fold_of = folds(series.max() + 1)
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
    use_aug = sys.argv[2] != "base" if len(sys.argv) > 2 else True
    oof = np.zeros(len(y))
    for f, s in [(f, s) for s in seeds for f in range(2)]:
        params = dict(main.LGB_PARAMS, num_threads=20, learning_rate=float(os.environ.get("LR", "0.05")),
                      seed=s, bagging_seed=s, feature_fraction_seed=s)
        t0 = time.time()
        trn = np.flatnonzero(fold_of[series] != f)
        atrn = np.flatnonzero((fold_of[aseries] != f) & use_aug)
        ds = lgb.Dataset(np.vstack([X[trn], AX[atrn]]), np.r_[y[trn], ay[atrn]], feature_name=names)
        b = lgb.train(params, ds, num_boost_round=int(os.environ.get("ROUNDS", "600")))
        del ds
        val = np.flatnonzero(fold_of[series] == f)
        p = b.predict(X[val], raw_score=True)
        oof[val] += p / len(seeds)
        print(f"seed {s} fold {f}: {ts_auc(y[val], p, step[val]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    m = fold_of[series] < 2  # folds() makes 5 folds; only 0 and 1 are validated
    print(f"OOF {'with shifted copies' if use_aug else 'baseline'} seeds {seeds}: {ts_auc(y[m], oof[m], step[m]):.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        cv()
