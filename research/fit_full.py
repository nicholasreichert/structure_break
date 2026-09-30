"""Fit the production models on ALL training series -> resources/model_{s}.txt.

Equivalent to main.train() (same features, stride, params, seeds), but reuses the cached
features in cache/train.npz. Columns are picked by name, so the matrix matches
main.FEATURE_NAMES exactly even if the cache holds extra (research) columns.
"""
import os
import time

import lightgbm as lgb
import numpy as np

from common import CACHE, ROOT, main


def load_matrix():
    d = np.load(os.path.join(CACHE, "train.npz"))
    names = list(d["names"])
    y, step = d["y"], d["step"]
    X = d["X"]
    del d
    keep = np.flatnonzero(step % main.TRAIN_ROW_STRIDE == 0)
    X = X[keep][:, [names.index(n) for n in main.FEATURE_NAMES]]
    return X, y[keep].astype(np.float32)


def run():
    X, y = load_matrix()
    ds = lgb.Dataset(X, y, feature_name=main.FEATURE_NAMES, free_raw_data=False)
    for s in range(main.N_SEEDS):
        t0 = time.time()
        b = lgb.train(dict(main.seed_params(s), num_threads=20), ds, num_boost_round=main.NUM_ROUNDS)
        out = os.path.join(ROOT, "resources", f"model_{s}.txt")
        b.save_model(out)
        print(f"saved {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    run()
