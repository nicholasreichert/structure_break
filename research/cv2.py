"""Grouped CV on the v7 feature set (main.FEATURE_NAMES picked out of cache/train.npz by name),
optionally plus extra blocks (cache/<name>_train.npy) and training-side variants.

usage: python research/cv2.py [block ...] [--folds 2] [--rounds 600] [--lr 0.05]
                              [--leaves 15] [--weight] [--xendcg] [--seeds 1] [--tag name]
Saves OOF predictions to cache/oof_<tag>.npy for paired comparisons / blending.
"""

import argparse
import os
import time

import lightgbm as lgb
import numpy as np

from common import CACHE, folds, main, ts_auc


def step_weights(y, step):
    n = np.bincount(step).astype(np.float64)
    pos = np.bincount(step, weights=y.astype(np.float64))
    w = (pos * (n - pos) / np.maximum(n, 1))[step]
    return w / w.mean()


def load(blocks):
    d = np.load(os.path.join(CACHE, "train.npz"))
    names = list(d["names"])
    base = [n for n in main.FEATURE_NAMES if n in names]  # blocks not in the cache come via args
    idx = [names.index(n) for n in base]
    y, series, step = d["y"], d["series"], d["step"]
    X = d["X"]
    del d
    parts = [X[:, idx]]
    del X
    fnames = list(base)
    for b in blocks:
        F = np.load(os.path.join(CACHE, f"{b}_train.npy"))
        assert F.shape[0] == len(y), b
        parts.append(F)
        fnames += [f"{b}{j}" for j in range(F.shape[1])]
    X = np.hstack(parts) if len(parts) > 1 else parts[0]
    return X, y, series, step, fnames


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("blocks", nargs="*")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=main.NUM_ROUNDS)
    ap.add_argument("--lr", type=float, default=main.LGB_PARAMS["learning_rate"])
    ap.add_argument("--leaves", type=int, default=main.LGB_PARAMS["num_leaves"])
    ap.add_argument("--weight", action="store_true")
    ap.add_argument("--xendcg", action="store_true", help="rank_xendcg with the online step as query")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    X, y, series, step, fnames = load(a.blocks)
    print(f"{X.shape[1]} features, {len(y)} rows", flush=True)
    fold_of = folds(series.max() + 1)[series]
    oof = np.full(len(y), np.nan)
    for f in range(a.folds):
        t0 = time.time()
        trn = np.flatnonzero(fold_of != f)
        val = np.flatnonzero(fold_of == f)
        params = dict(main.LGB_PARAMS, num_threads=20, learning_rate=a.lr, num_leaves=a.leaves)
        kw = {}
        if a.xendcg:
            trn = trn[np.argsort(step[trn], kind="stable")]  # queries must be contiguous
            params.update(objective="rank_xendcg")
            kw["group"] = np.bincount(step[trn])[np.unique(step[trn])]
        if a.weight:
            kw["weight"] = step_weights(y[trn], step[trn])
        p = np.zeros(len(val))
        for s in range(a.seeds):
            params.update(seed=s, bagging_seed=s, feature_fraction_seed=s)
            b = lgb.train(params, lgb.Dataset(X[trn], y[trn], feature_name=fnames, **kw),
                          num_boost_round=a.rounds)
            p += b.predict(X[val])
        oof[val] = p / a.seeds
        print(f"fold {f}: TS-AUC {ts_auc(y[val], oof[val], step[val]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    m = ~np.isnan(oof)
    print(f"OOF TS-AUC {ts_auc(y[m], oof[m], step[m]):.4f}")
    if a.tag:
        np.save(os.path.join(CACHE, f"oof_{a.tag}.npy"), oof.astype(np.float32))


if __name__ == "__main__":
    run()
