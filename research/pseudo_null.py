"""Model-level per-series calibration from a pseudo-online segment cut from the history.

Why: CalBlock calibrates each stream per series, but the final score is a nonlinear mix of
v6 CUSUMs/GLRs/windows that are still on an iid-ish null, and the miscalibration differs by
series. Cut the last m points of the history off as a *known no-break* pseudo-online segment,
run the full pipeline on it, and the model's scores there are this series' own null score
distribution. A stage-2 model then ranks the real online score against that null.

  python research/pseudo_null.py build   # pseudo features -> cache/pseudo_train.npz
  python research/pseudo_null.py cv      # stage 1 (2-fold OOF) + stage 2 (2-fold) TS-AUC
  python research/pseudo_null.py stage2  # stage 2 only, from the saved stage-1 scores
"""

import os
import sys
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np

from common import CACHE, folds, load_train, main, ts_auc
from cv2 import load

PSEUDO_MAX = 500
PSEUDO_STRIDE = 4
PATH = os.path.join(CACHE, "pseudo_train.npz")


def pseudo_len(nh: int) -> int:
    return min(PSEUDO_MAX, nh // 3)


def _one(xh):
    m = pseudo_len(len(xh))
    return main.series_features(xh[:-m], xh[-m:])[::PSEUDO_STRIDE]


def build():
    ds = load_train()
    jobs = [ds.segments(k)[0] for k in range(len(ds))]
    t0 = time.time()
    with Pool(20) as pool:
        feats = pool.map(_one, jobs, chunksize=20)
    print(f"pseudo features: {len(ds)} series in {time.time() - t0:.0f}s")
    X = np.concatenate(feats)
    series = np.concatenate([np.full(len(f), k, dtype=np.int32) for k, f in enumerate(feats)])
    step = np.concatenate([np.arange(len(f), dtype=np.int32) * PSEUDO_STRIDE for f in feats])
    np.savez(PATH, X=X, series=series, step=step)


def stage2_features(s, step, series, ps, pstep, pseries, n_series):
    """Per real row: its score plus summaries of the same series' pseudo-null scores."""
    order = np.lexsort((pstep, pseries))
    ps, pstep, pseries = ps[order], pstep[order], pseries[order]
    start = np.searchsorted(pseries, np.arange(n_series))
    cnt = np.bincount(pseries, minlength=n_series)
    csum = np.r_[0.0, np.cumsum(ps)]
    mean_all = (csum[start + cnt] - csum[start]) / cnt
    sd_all = np.array([ps[a : a + c].std() for a, c in zip(start, cnt)])
    q90 = np.array([np.quantile(ps[a : a + c], 0.9) for a, c in zip(start, cnt)])
    # pseudo score at the matching online step (clipped to the pseudo segment's length)
    j = np.minimum(step // PSEUDO_STRIDE, cnt[series] - 1)
    at = ps[start[series] + j]
    mean_to = (csum[start[series] + j + 1] - csum[start[series]]) / (j + 1)
    ma, sa = mean_all[series], np.maximum(sd_all[series], 1e-3)
    F = np.column_stack([s, np.log1p(step), at, mean_to, ma, sa, q90[series],
                         s - at, s - mean_to, (s - ma) / sa])
    names = ["s", "logt", "p_at", "p_mean_to", "p_mean", "p_sd", "p_q90",
             "s_m_at", "s_m_meanto", "s_z"]
    return F.astype(np.float32), names


def cv():
    X, y, series, step, names = load([])
    P = np.load(PATH)
    PX, pseries, pstep = P["X"], P["series"], P["step"]
    n_series = series.max() + 1
    fold_of = folds(n_series)
    s = np.zeros(len(y))
    ps = np.zeros(len(pseries))
    params = dict(main.LGB_PARAMS, num_threads=20, learning_rate=0.05)
    for f in range(2):
        t0 = time.time()
        trn = np.flatnonzero(fold_of[series] != f)
        b = lgb.train(params, lgb.Dataset(X[trn], y[trn], feature_name=names), num_boost_round=600)
        val = np.flatnonzero(fold_of[series] == f)
        s[val] = b.predict(X[val], raw_score=True)
        pv = np.flatnonzero(fold_of[pseries] == f)
        ps[pv] = b.predict(PX[pv], raw_score=True)
        print(f"stage1 fold {f}: {ts_auc(y[val], s[val], step[val]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    del X, PX
    np.save(os.path.join(CACHE, "pseudo_s1.npy"), s.astype(np.float32))
    np.save(os.path.join(CACHE, "pseudo_ps1.npy"), ps.astype(np.float32))
    m = fold_of[series] < 2  # folds() makes 5 folds; only 0 and 1 are validated
    print(f"stage1 OOF: {ts_auc(y[m], s[m], step[m]):.4f}")
    stage2(y, step, series, s, ps, pstep, pseries, n_series, fold_of)


def stage2(y, step, series, s, ps, pstep, pseries, n_series, fold_of):
    F, fn = stage2_features(s, step, series, ps, pstep, pseries, n_series)
    # stage-1 scores exist only for folds 0 and 1: stage 2 trains on one, validates on the other
    keep = fold_of[series] < 2
    F, y, step, series = F[keep], y[keep], step[keep], series[keep]
    for name, cols in [("s,logt only", [0, 1]), ("+ pseudo null", list(range(len(fn))))]:
        oof = np.zeros(len(y))
        for f in range(2):
            trn = np.flatnonzero(fold_of[series] == 1 - f)
            val = np.flatnonzero(fold_of[series] == f)
            p2 = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=2000,
                      verbose=-1, num_threads=20, seed=0)
            b = lgb.train(p2, lgb.Dataset(F[trn][:, cols], y[trn]), num_boost_round=300)
            oof[val] = b.predict(F[val][:, cols])
        print(f"stage2 [{name}]: {ts_auc(y, oof, step):.4f}", flush=True)
    for k, n in enumerate(fn[2:], 2):
        print(f"  alone {n}: {ts_auc(y, F[:, k], step):.4f}")


def stage2_only():
    d = np.load(os.path.join(CACHE, "train.npz"), mmap_mode="r")
    y, series, step = np.asarray(d["y"]), np.asarray(d["series"]), np.asarray(d["step"])
    P = np.load(PATH, mmap_mode="r")
    pseries, pstep = np.asarray(P["series"]), np.asarray(P["step"])
    s = np.load(os.path.join(CACHE, "pseudo_s1.npy")).astype(np.float64)
    ps = np.load(os.path.join(CACHE, "pseudo_ps1.npy")).astype(np.float64)
    n = series.max() + 1
    stage2(y, step, series, s, ps, pstep, pseries, n, folds(n))


if __name__ == "__main__":
    {"build": build, "cv": cv, "stage2": stage2_only}[sys.argv[1]]()
