"""Grouped 5-fold CV of the LightGBM model on cached train features, scored by TS-AUC.

usage: python research/cv.py [--rebuild] [--folds N] [--rounds R] [--drop PREFIX ...]
"""

import argparse
import time

import lightgbm as lgb
import numpy as np

from common import cached, folds, load_test_reduced, load_train, main, ts_auc


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--folds", type=int, default=5, help="how many of the 5 folds to run")
    ap.add_argument("--rounds", type=int, default=main.NUM_ROUNDS)
    ap.add_argument("--drop", nargs="*", default=[], help="drop features starting with these")
    ap.add_argument("--param", nargs="*", default=[], help="LightGBM overrides key=value")
    ap.add_argument("--keep", nargs="*", default=None, help="keep only features starting with these")
    args = ap.parse_args()

    tr = cached("train", load_train, args.rebuild)
    te = cached("test_reduced", load_test_reduced, args.rebuild)
    names = list(tr["names"])
    cols = [
        i for i, n in enumerate(names)
        if not any(n.startswith(p) for p in args.drop)
        and (args.keep is None or n in main.META_NAMES or any(n.startswith(p) for p in args.keep))
    ]
    print(f"{len(cols)} features, {len(tr['y'])} train rows")

    X, y, series, step = tr["X"][:, cols], tr["y"], tr["series"], tr["step"]
    fold_of = folds(series.max() + 1)[series]
    oof = np.full(len(y), np.nan)
    stride = main.TRAIN_ROW_STRIDE
    for f in range(args.folds):
        t0 = time.time()
        trn = np.flatnonzero((fold_of != f) & (step % stride == 0))
        val = np.flatnonzero(fold_of == f)
        b = lgb.train(
            dict(main.LGB_PARAMS, num_threads=20, **{k: type(main.LGB_PARAMS.get(k, 0.0))(v) for k, v in (p.split("=") for p in args.param)}),
            lgb.Dataset(X[trn], y[trn], feature_name=[names[i] for i in cols]),
            num_boost_round=args.rounds,
        )
        oof[val] = b.predict(X[val])
        print(f"fold {f}: TS-AUC {ts_auc(y[val], oof[val], step[val]):.4f}  ({time.time() - t0:.0f}s)")

    m = ~np.isnan(oof)
    print(f"OOF TS-AUC {ts_auc(y[m], oof[m], step[m]):.4f}")
    p_te = b.predict(te["X"][:, cols])
    print(f"reduced-test TS-AUC (last fold model, 100 series, noisy) {ts_auc(te['y'], p_te, te['step']):.4f}")

    imp = b.feature_importance("gain")
    top = np.argsort(imp)[::-1][:20]
    print("top gain:", ", ".join(f"{names[cols[i]]}={imp[i] / imp.sum():.3f}" for i in top))


if __name__ == "__main__":
    run()
