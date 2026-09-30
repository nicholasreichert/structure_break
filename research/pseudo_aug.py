"""Augmentation test: add the pseudo-online segments (history tails, known no-break) to the
training folds as extra negative series. The learning curve says the model is data-limited
(1/4 -> 1/2 of the series: fold-0 TS-AUC 0.6203 -> 0.6310), and these are free labels.

Compare with the stage-1 OOF of research/pseudo_null.py (same params, same folds).
usage (repo root): python research/pseudo_aug.py [--weight w]
"""

import os
import sys
import time

import lightgbm as lgb
import numpy as np

from common import CACHE, folds, main, ts_auc
from cv2 import load


def run():
    w = float(sys.argv[sys.argv.index("--weight") + 1]) if "--weight" in sys.argv else 1.0
    X, y, series, step, names = load([])
    P = np.load(os.path.join(CACHE, "pseudo_train.npz"))
    PX, pseries = P["X"], P["series"]
    fold_of = folds(series.max() + 1)
    params = dict(main.LGB_PARAMS, num_threads=20, learning_rate=0.05)
    oof = np.zeros(len(y))
    for f in range(2):
        t0 = time.time()
        trn = np.flatnonzero(fold_of[series] != f)
        ptrn = np.flatnonzero(fold_of[pseries] != f)
        # pseudo rows are at stride 4, real rows at stride 2: weight 2 per pseudo row
        # puts one pseudo series on the same footing as one real series
        Xt = np.vstack([X[trn], PX[ptrn]])
        yt = np.r_[y[trn], np.zeros(len(ptrn), dtype=y.dtype)]
        wt = np.r_[np.ones(len(trn)), np.full(len(ptrn), 2.0 * w)]
        b = lgb.train(params, lgb.Dataset(Xt, yt, weight=wt, feature_name=names), num_boost_round=600)
        del Xt
        val = np.flatnonzero(fold_of[series] == f)
        oof[val] = b.predict(X[val], raw_score=True)
        print(f"fold {f}: {ts_auc(y[val], oof[val], step[val]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    m = fold_of[series] < 2  # folds() makes 5 folds; only 0 and 1 are validated
    print(f"OOF with pseudo negatives (w={w}): {ts_auc(y[m], oof[m], step[m]):.4f}")
    np.save(os.path.join(CACHE, f"oof_pseudoaug_w{w}.npy"), oof.astype(np.float32))


if __name__ == "__main__":
    run()
