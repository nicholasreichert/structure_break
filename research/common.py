"""Shared research helpers: feature caching, TS-AUC, grouped CV."""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402
from data_loader.load import Dataset, load, load_test_reduced, load_train  # noqa: E402

CACHE = os.path.join(ROOT, "cache")


def _feat_one(args):
    x_hist, x_online = args
    return main.series_features(x_hist, x_online)


def compute_features(ds: Dataset, processes: int = 20) -> dict:
    """Features for every online step of every series, plus labels and bookkeeping."""
    jobs = [ds.segments(k) for k in range(len(ds))]
    t0 = time.time()
    with Pool(processes) as pool:
        feats = pool.map(_feat_one, jobs, chunksize=20)
    print(f"features: {len(ds)} series in {time.time() - t0:.1f}s")
    X = np.concatenate(feats)
    series = np.concatenate([np.full(len(f), k, dtype=np.int32) for k, f in enumerate(feats)])
    step = np.concatenate([np.arange(len(f), dtype=np.int32) for f in feats])
    y = np.concatenate([ds.labels(k) for k in range(len(ds))]).astype(np.int8)
    return dict(X=X, y=y, series=series, step=step, names=np.array(main.FEATURE_NAMES))


def cached(name: str, ds_fn, rebuild: bool = False) -> dict:
    """Load cache/<name>.npz, rebuilding if missing, stale (feature names changed) or forced."""
    path = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(path) and not rebuild:
        d = dict(np.load(path, allow_pickle=False))
        if list(d["names"]) == main.FEATURE_NAMES:
            return d
        print(f"{name}: feature set changed, rebuilding")
    os.makedirs(CACHE, exist_ok=True)
    d = compute_features(ds_fn())
    np.savez(path, **d)
    return d


def ts_auc(y: np.ndarray, score: np.ndarray, step: np.ndarray) -> float:
    """Time-Stratified AUC: per-step cross-sectional AUC, weighted by n_pos * n_neg."""
    order = np.lexsort((score, step))
    y, score, step = y[order], score[order], step[order]
    bounds = np.flatnonzero(np.diff(step)) + 1
    num = den = 0.0
    for ys, ss in zip(np.split(y, bounds), np.split(score, bounds)):
        npos = int(ys.sum())
        nneg = len(ys) - npos
        if npos == 0 or nneg == 0:
            continue
        # midranks handle ties; scores are already sorted within the step
        _, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
        cum = np.cumsum(cnt)
        ranks = (cum - (cnt - 1) / 2.0)[inv]
        auc = (ranks[ys == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
        w = npos * nneg
        num += w * auc
        den += w
    return num / den if den else 0.5


def folds(n_series: int, k: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n_series) % k
