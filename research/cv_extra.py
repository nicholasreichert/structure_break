"""cv.py, but with extra per-row feature blocks (cache/<name>_train.npy) appended.

usage: python research/cv_extra.py bayes [more ...] [--folds N]
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np

from common import CACHE, CACHE_STRIDE, cached, folds, load_train, main, ts_auc


def run():
    argv = sys.argv[1:]
    nf = 2
    if "--folds" in argv:
        i = argv.index("--folds")
        nf = int(argv[i + 1])
        del argv[i : i + 2]
    base_only = "--base" in argv
    args = [a for a in argv if not a.startswith("--")]
    tr = cached("train", load_train)
    X, y, series, step = tr["X"], tr["y"], tr["series"], tr["step"]
    names = list(tr["names"])
    blocks = [X] if not base_only else [X]
    full_keep = None
    for a in args:
        F = np.load(os.path.join(CACHE, f"{a}_train.npy"), mmap_mode="r")
        if F.shape[0] != len(y):  # block saved at every step: keep the cache's steps
            if full_keep is None:
                T = load_train().online_len
                full_keep = np.concatenate([np.arange(n) % CACHE_STRIDE == 0 for n in T])
            F = F[full_keep]
        F = np.asarray(F)
        blocks.append(F)
        names += [f"{a}{j}" for j in range(F.shape[1])]
    X = np.hstack(blocks) if len(blocks) > 1 else X
    print(f"{X.shape[1]} features, {len(y)} rows")
    fold_of = folds(series.max() + 1)[series]
    stride = main.TRAIN_ROW_STRIDE
    num = den = 0.0
    oof = np.full(len(y), np.nan)
    for f in range(nf):
        t0 = time.time()
        trn = np.flatnonzero((fold_of != f) & (step % stride == 0))
        val = np.flatnonzero(fold_of == f)
        b = lgb.train(dict(main.LGB_PARAMS, num_threads=20),
                      lgb.Dataset(X[trn], y[trn], feature_name=names), num_boost_round=main.NUM_ROUNDS)
        oof[val] = b.predict(X[val])
        print(f"fold {f}: TS-AUC {ts_auc(y[val], oof[val], step[val]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    m = ~np.isnan(oof)
    print(f"OOF TS-AUC {ts_auc(y[m], oof[m], step[m]):.4f}")
    imp = b.feature_importance("gain")
    top = np.argsort(imp)[::-1][:15]
    print("top gain:", ", ".join(f"{names[i]}={imp[i] / imp.sum():.3f}" for i in top))


if __name__ == "__main__":
    run()
