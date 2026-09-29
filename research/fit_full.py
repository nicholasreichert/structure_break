"""Fit the production model on ALL training series (cached features) -> resources/model.txt.

Equivalent to main.train() (same features, same stride, same params), just parallel.
"""
import os
import time

import lightgbm as lgb
import numpy as np

from common import ROOT, cached, load_train, main


def run():
    tr = cached("train", load_train)
    keep = tr["step"] % main.TRAIN_ROW_STRIDE == 0
    t0 = time.time()
    b = lgb.train(
        dict(main.LGB_PARAMS, num_threads=20),
        lgb.Dataset(tr["X"][keep], tr["y"][keep].astype(np.float32), feature_name=main.FEATURE_NAMES),
        num_boost_round=main.NUM_ROUNDS,
    )
    out = os.path.join(ROOT, "resources", "model.txt")
    b.save_model(out)
    print(f"saved {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    run()
